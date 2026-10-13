"""Fine-tuning mixto 5A desde época 19. CPU/FP32 reproducible, sin Test.

python -B entrenar_sft.py --smoke-test
python -B entrenar_sft.py --train
python -B entrenar_sft.py --train --resume
"""
import argparse
import copy
import json
import math
import random
import time
from collections import Counter

import torch
import torch.nn.functional as F

from dataset_sft import (BASE, SFT, TASKS, DatasetSFT, causal_dataset, load_model,
                         frozen_hashes, write_json, sha256, require, mixed_schedule, prompt,
                         install_test_guard)
from evaluar_sft import evaluate_tokens, generation_suite, final_report

CONFIG = dict(max_epochs=8, batch_size=16, learning_rate=5e-5, optimizer='AdamW', betas=[.9, .95],
              eps=1e-8, weight_decay=.01, gradient_clip_norm=1., scheduler='none', seed=42,
              precision='FP32', num_workers=0, cpu_threads=2, device='cpu',
              early_stopping_patience=2, early_stopping_min_delta=.001,
              target_step_percent=dict(grounded_instruction=45, direct_instruction=10, abstention=15,
                                       causal_language_model_retention=30))
RESULTS = BASE / 'resultados'
CODE_FILES = ('preparar_sft.py', 'dataset_sft.py', 'entrenar_sft.py', 'evaluar_sft.py')
PROTECTED = ('modelos/checkpoint_best.pt', 'modelos/checkpoint_last.pt', 'modelo.py', 'tokenizador.py',
             'dataset.py', 'entrenar.py', 'evaluar.py', 'datos/dataset/train.pt',
             'datos/dataset/validation.pt', 'datos/tokenizer/tokenizer.json')
# No hash/read de test.pt: el guard impide cualquier acceso a su contenido.


def protected_hashes():
    return {p: sha256(BASE / p) for p in PROTECTED}


def setup():
    torch.set_num_threads(CONFIG['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], betas=tuple(CONFIG['betas']),
                            eps=CONFIG['eps'], weight_decay=CONFIG['weight_decay'])


def batch(ds, indices):
    rows = [ds[i] for i in indices]
    return torch.stack([x for x, _ in rows]), torch.stack([y for _, y in rows])


def train_step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(x)
    count = (y != -100).sum().item()
    require(count > 0, 'Batch sin targets.')
    summed = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction='sum')
    loss = summed / count
    require(torch.isfinite(loss).item(), 'Loss no finita.')
    loss.backward()
    require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()), 'Gradientes ausentes/no finitos.')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip_norm'], error_if_nonfinite=True)
    optimizer.step()
    return dict(ce_sum=summed.item(), targets=count, gradient_norm=norm.item())


def rng_state():
    return dict(torch=torch.get_rng_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])


def atomic_checkpoint(path, cp):
    require(path.name in ('checkpoint_sft_best.pt', 'checkpoint_sft_last.pt'), 'Salida checkpoint no autorizada.')
    temporary = path.with_suffix('.pt.tmp')
    torch.save(cp, temporary)
    temporary.replace(path)


def preflight(train, validation, hashes, protected):
    begin = time.perf_counter()
    initial_rng = rng_state()
    # Checks de índices independientes para TODOS los ejemplos finales.
    for ds in (train, validation):
        for r, (x, y) in zip(ds.records, ds.encoded):
            require(x.dtype == y.dtype == torch.long and x.shape == y.shape == (128,), 'Shape/dtype incorrectos.')
            a = ds.tokenizer.encode(r['answer'])
            prefix = [2] + ds.tokenizer.encode(prompt(r))
            first = len(prefix) - 1
            expected = torch.tensor(a + [3], dtype=torch.long)
            require(torch.equal(y[first:first+len(expected)], expected), 'Targets no equivalen a respuesta+EOS.')
            require((y[:first] == -100).all().item() and (y[first+len(expected):] == -100).all().item(), 'Prompt/PAD participan en loss.')
            require((y == 3).sum().item() == 1 and (y[x == 0] == -100).all().item(), 'EOS/PAD fallaron.')
            require(((y == -100) | ((y >= 0) & (y < 6000))).all().item(), 'Target fuera de vocabulario.')
    causal = causal_dataset('train')
    cx, cy = batch(causal, [0, len(causal)//2, len(causal)-1])
    require(((cy == -100) | ((cy >= 0) & (cy < 6000))).all().item(), 'Target causal inválido.')
    # Padding a derecha no cambia predicción de tokens previos (máscara causal).
    model, _ = load_model()
    model.eval()
    x, y = batch(train, list(range(4)))
    with torch.no_grad():
        full, _ = model(x)
        short, _ = model(x[:, :int((x[0] != 0).sum().item())])
        require(torch.allclose(full[:, :short.shape[1]], short, atol=2e-5, rtol=1e-5), 'Padding cambia logits reales.')
        positions = y != -100
        ignored = y == -100
        original = F.cross_entropy(full.flatten(0, 1), y.flatten(), ignore_index=-100)
        altered = full.clone()
        altered[ignored] = torch.randn_like(altered[ignored]) * 100
        require(torch.equal(original, F.cross_entropy(altered.flatten(0, 1), y.flatten(), ignore_index=-100)), 'Logits de PAD/prompt alteran loss.')
        require(positions.any().item(), 'No se supervisa la respuesta.')
    # Overfit sobre un único batch Train desechable que incluye las tres tareas.
    indices = []
    for task in TASKS:
        indices.extend([i for i, r in enumerate(train.records) if r['task_type'] == task][:2])
    x, y = batch(train, indices)
    opt = optimizer_for(model)
    def assistant_loss():
        model.eval()
        with torch.no_grad():
            logits, _ = model(x)
            return F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100).item()
    initial = assistant_loss()
    timings = []
    for _ in range(60):
        step_start = time.perf_counter()
        train_step(model, opt, x, y)
        timings.append(time.perf_counter()-step_start)
    final = assistant_loss()
    require(final < initial * .8, 'Overfit no redujo claramente assistant loss.')
    # Reanudar RNG/model/AdamW produce el mismo siguiente paso, sin archivo temporal.
    checkpoint = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()), rng=rng_state())
    first_result = train_step(model, opt, x, y)
    expected_state = copy.deepcopy(model.state_dict())
    restored, _ = load_model()
    restored.load_state_dict(checkpoint['model'], strict=True)
    restored_opt = optimizer_for(restored)
    restored_opt.load_state_dict(checkpoint['optimizer'])
    restore_rng(checkpoint['rng'])
    repeat_result = train_step(restored, restored_opt, x, y)
    require(first_result == repeat_result and all(torch.equal(v, restored.state_dict()[k]) for k, v in expected_state.items()), 'Reanudación no reproducible.')
    # Un paso causal comprueba objetivo original y gradientes finitos.
    causal_result = train_step(restored, restored_opt, cx, cy)
    del model, opt, restored, restored_opt, checkpoint, expected_state
    restore_rng(initial_rng)
    require(frozen_hashes() == hashes and protected_hashes() == protected, 'Preflight alteró entradas.')
    schedule = mixed_schedule(train)
    require(schedule == mixed_schedule(train), 'Agenda no determinista.')
    seen = [i for task, ids in schedule if ids is not None for i in ids]
    require(sorted(seen) == list(range(len(train))), 'Época no recorre instrucciones exactamente una vez.')
    counts = Counter(t for t, _ in schedule)
    report = dict(version='5A', passed=True, hashes=hashes, protected_hashes=protected,
                  training_config=CONFIG, deviations_before_official_training=[],
                  checked_instruction_examples=len(train)+len(validation), shapes_masking_EOS_PAD_vocab=True,
                  finite_gradients=True, deterministic_resume_exact=True,
                  overfit=dict(records=len(indices), steps=60, initial_assistant_loss=initial, final_assistant_loss=final,
                               relative_reduction=1-final/initial, model_discarded=True, temporary_checkpoints_created=0),
                  causal_smoke=causal_result, step_counts_per_epoch=dict(counts),
                  step_percent_per_epoch={t: 100*n/len(schedule) for t, n in counts.items()},
                  mean_small_batch_step_seconds=sum(timings)/len(timings), duration_seconds=time.perf_counter()-begin,
                  official_training_started=False, test_used=False,
                  code_hashes={p: sha256(BASE / p) for p in CODE_FILES})
    write_json(RESULTS / 'reporte_preflight_sft_5A.json', report)
    print(json.dumps(report['overfit'], ensure_ascii=True), flush=True)
    return report


def baseline(validation, retention, hashes):
    model, _ = load_model()
    fixed = json.loads((SFT / 'prompts_validation_5A.json').read_text(encoding='utf-8'))
    instruction = evaluate_tokens(model, validation)
    lm = evaluate_tokens(model, retention)
    require(abs(lm['loss'] - 4.725705095024419) < 1e-5, 'Baseline causal no reproduce Validation histórica.')
    suite = generation_suite(model, validation, fixed['records'], fixed['generation_config'])
    negatives = generation_suite(model, validation, [r for r in validation.records if r['task_type'] == 'abstention'])
    result = dict(version='5A', hashes=hashes, instruction=instruction, retention=lm,
                  generation=suite, all_validation_abstention=negatives, test_used=False)
    write_json(RESULTS / 'baseline_sft_5A.json', result)
    print(f"Baseline: assistant loss={instruction['loss']:.6f}; LM loss={lm['loss']:.6f}", flush=True)
    return result


def train_official(train, validation, causal_train, causal_validation, hashes, protected, resume=False):
    preflight_path = RESULTS / 'reporte_preflight_sft_5A.json'
    require(preflight_path.exists(), 'Ejecutar --smoke-test antes del entrenamiento.')
    smoke = json.loads(preflight_path.read_text(encoding='utf-8'))
    require(smoke['passed'] and smoke['hashes'] == hashes and smoke['training_config'] == CONFIG, 'Preflight incompatible.')
    require(smoke['code_hashes'] == {p: sha256(BASE / p) for p in CODE_FILES}, 'Código cambió después del preflight.')
    last_path = BASE / 'modelos/checkpoint_sft_last.pt'
    best_path = BASE / 'modelos/checkpoint_sft_best.pt'
    require(resume or not (last_path.exists() or best_path.exists() or (SFT / 'training_lock_5A.json').exists()), 'Run existente: usar --resume; no sobrescribir silenciosamente.')
    saved_baseline = RESULTS / 'baseline_sft_5A.json'
    if resume:
        require(last_path.exists() and saved_baseline.exists(), 'No hay estado oficial para reanudar.')
        base_metrics = json.loads(saved_baseline.read_text(encoding='utf-8'))
        require(base_metrics['hashes'] == hashes, 'Baseline incompatible.')
    else:
        base_metrics = baseline(validation, causal_validation, hashes)
    if resume:
        model, cp = load_model(last_path)
        require(cp['hashes'] == hashes and cp['training_config'] == CONFIG and cp['protected_hashes'] == protected, 'Checkpoint incompatible.')
        require(cp['code_hashes'] == smoke['code_hashes'], 'Código cambió desde el checkpoint.')
        opt = optimizer_for(model)
        opt.load_state_dict(cp['optimizer_state_dict'])
        restore_rng(cp['rng_state'])
        start = cp['epoch'] + 1
        history, global_step, best = cp['history'], cp['global_step'], cp['best_validation_loss']
        early_best, counter = cp['early_best'], cp['early_stopping_counter']
    else:
        model, base_cp = load_model()
        opt = optimizer_for(model)
        start, history, global_step, best = 1, [], 0, math.inf
        early_best, counter = math.inf, 0
        write_json(SFT / 'training_lock_5A.json', dict(version='5A', hashes=hashes, training_config=CONFIG, code_hashes=smoke['code_hashes']))
    mc = copy.deepcopy((cp if resume else base_cp)['model_config'])
    if counter >= CONFIG['early_stopping_patience'] or start > CONFIG['max_epochs']:
        print('Entrenamiento ya terminado; se recalcula el reporte final.', flush=True)
    else:
        for epoch in range(start, CONFIG['max_epochs']+1):
            begin = time.perf_counter()
            schedule = mixed_schedule(train, CONFIG['batch_size'], CONFIG['seed'], epoch)
            counts, sums = Counter(), {}
            causal_rng = random.Random(CONFIG['seed'] + 10000 + epoch)
            causal_indices = list(range(len(causal_train)))
            causal_rng.shuffle(causal_indices)
            cursor = 0
            for task, indices in schedule:
                if indices is None:
                    indices = causal_indices[cursor:cursor+CONFIG['batch_size']]
                    require(len(indices) == CONFIG['batch_size'], 'No hay suficientes ejemplos causal Train.')
                    cursor += len(indices)
                    x, y = batch(causal_train, indices)
                else:
                    x, y = batch(train, indices)
                result = train_step(model, opt, x, y)
                bucket = sums.setdefault(task, dict(ce_sum=0., targets=0))
                bucket['ce_sum'] += result['ce_sum']
                bucket['targets'] += result['targets']
                counts[task] += 1
                global_step += 1
            instruction = evaluate_tokens(model, validation)
            retention = evaluate_tokens(model, causal_validation)
            improved = instruction['loss'] < best
            best = min(best, instruction['loss'])
            if instruction['loss'] < early_best - CONFIG['early_stopping_min_delta']:
                early_best, counter = instruction['loss'], 0
            else:
                counter += 1
            row = dict(epoch=epoch, global_step=global_step, step_counts=dict(counts),
                       step_percent={t: 100*n/len(schedule) for t, n in counts.items()},
                       train_by_task_type={t: b | dict(loss=b['ce_sum']/b['targets']) for t, b in sums.items()},
                       validation_instruction=instruction, original_validation_lm=retention,
                       learning_rate=CONFIG['learning_rate'], best=improved, early_stopping_counter=counter,
                       duration_seconds=time.perf_counter()-begin)
            history.append(row)
            cp = dict(version='5A', epoch=epoch, epoch_completed=epoch, global_step=global_step,
                      model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(), model_config=mc,
                      base_checkpoint_hash=hashes['base_checkpoint'], tokenizer_hash=hashes['tokenizer'],
                      dataset_hashes={k: hashes[k] for k in ('train', 'validation', 'causal_train', 'causal_validation')},
                      hashes=hashes, protected_hashes=protected, training_config=CONFIG, history=copy.deepcopy(history),
                      best_validation_loss=best, early_best=early_best, early_stopping_counter=counter,
                      rng_state=rng_state(), code_hashes=smoke['code_hashes'], seed=42,
                      pytorch_version=str(torch.__version__), device_type='cpu', test_used=False)
            require(frozen_hashes() == hashes and protected_hashes() == protected, 'Entradas base cambiaron durante entrenamiento.')
            atomic_checkpoint(last_path, cp)
            if improved:
                atomic_checkpoint(best_path, cp)
            write_json(RESULTS / 'historial_sft_5A.json', history)
            print(f"Época {epoch}/8: assistant={instruction['loss']:.5f}; LM={retention['loss']:.5f}; EOS={instruction['EOS_accuracy_at_reference_end']:.3f}; best={improved}", flush=True)
            if counter >= CONFIG['early_stopping_patience']:
                print('Early stopping exclusivamente por Validation.', flush=True)
                break
    best_model, best_cp = load_model(best_path)
    # El reporte conserva todas las épocas, aunque best sea anterior a last.
    best_cp['history'] = history
    instruction = evaluate_tokens(best_model, validation)
    retention = evaluate_tokens(best_model, causal_validation)
    fixed = json.loads((SFT / 'prompts_validation_5A.json').read_text(encoding='utf-8'))
    suite = generation_suite(best_model, validation, fixed['records'], fixed['generation_config'])
    report = final_report(best_cp, base_metrics['instruction'], base_metrics['retention'], base_metrics['generation'], instruction, retention, suite)
    require(frozen_hashes() == hashes and protected_hashes() == protected, 'Entradas protegidas cambiaron.')
    write_json(RESULTS / 'reporte_entrenamiento_sft_5A.json', report)
    print(report['verdict'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke-test', action='store_true')
    mode.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    require(not args.resume or args.train, '--resume requiere --train.')
    accesses = install_test_guard()
    setup()
    hashes, protected = frozen_hashes(), protected_hashes()
    train, validation = DatasetSFT('train'), DatasetSFT('validation')
    if args.smoke_test:
        preflight(train, validation, hashes, protected)
    else:
        print('Entrenamiento OFICIAL 5A: hasta 8 épocas, LR=5e-5, FP32/CPU; datasets congelados.', flush=True)
        train_official(train, validation, causal_dataset('train'), causal_dataset('validation'), hashes, protected, args.resume)
    write_json(RESULTS / 'auditoria_accesos_sft_5A.json', dict(version='5A', mode='smoke' if args.smoke_test else 'official',
               test_guard_active=True, test_documents_used=[], test_used=False, workspace_files_opened=sorted(accesses)))


if __name__ == '__main__':
    main()
