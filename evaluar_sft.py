"""Métricas 5A separadas por objetivo y comparación fija de generación."""
import argparse
import json
import math
import re
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_sft import (BASE, SFT, ABSTENTION, DatasetSFT, causal_dataset, prompt,
                         load_model, frozen_hashes, write_json, require, install_test_guard)
from dataset_sft import sha256


@torch.no_grad()
def evaluate_tokens(model, ds, batch_size=16):
    model.eval()
    ce, targets, eos_top1, eos_top5, eos_count = 0., 0, 0, 0, 0
    task_sums = {}
    offset = 0
    for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
        logits, _ = model(x)
        losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction='none').reshape_as(y)
        require(torch.isfinite(losses).all().item(), 'Loss de evaluación no finita.')
        ce += losses.double().sum().item()
        targets += (y != -100).sum().item()
        if isinstance(ds, DatasetSFT):
            eos_logits = logits[y == 3]
            eos_count += len(eos_logits)
            eos_top1 += (eos_logits.argmax(-1) == 3).sum().item()
            eos_top5 += (eos_logits.topk(5, -1).indices == 3).any(-1).sum().item()
            for j, r in enumerate(ds.records[offset:offset + len(x)]):
                bucket = task_sums.setdefault(r['task_type'], dict(cross_entropy_sum=0., targets=0))
                bucket['cross_entropy_sum'] += losses[j].double().sum().item()
                bucket['targets'] += (y[j] != -100).sum().item()
            offset += len(x)
    require(targets > 0, 'No hay targets.')
    loss = ce / targets
    result = dict(loss=loss, perplexity=math.exp(min(loss, 700)), cross_entropy_sum=ce, targets=targets)
    if isinstance(ds, DatasetSFT):
        result.update(assistant_token_cross_entropy=loss, assistant_token_perplexity=result['perplexity'],
                      EOS_accuracy_at_reference_end=eos_top1 / eos_count, EOS_top5_at_reference_end=eos_top5 / eos_count,
                      EOS_targets=eos_count, mean_answer_tokens=sum(r['answer_token_count'] for r in ds.records) / len(ds),
                      by_task_type={t: b | dict(loss=b['cross_entropy_sum']/b['targets']) for t, b in task_sums.items()})
    return result


@torch.no_grad()
def generate(model, tokenizer, r, max_new_tokens=48):
    model.eval()
    prefix = [2] + tokenizer.encode(prompt(r))
    require(len(prefix) < 128, 'Prompt no cabe.')
    tokens, eos = [], False
    budget = min(max_new_tokens, 128 - len(prefix) + 1)
    for _ in range(budget):
        ids = torch.tensor([prefix + tokens], dtype=torch.long)
        logits, _ = model(ids)
        scores = logits[0, -1].clone()
        scores[:3] = -torch.inf
        next_id = scores.argmax().item()
        if next_id == 3:
            eos = True
            break
        tokens.append(next_id)
    raw = tokenizer.decode(tokens)
    stripped = raw.strip()
    initial_fragment = bool(re.match(r'^(?:a la|a los|de la|de los|del |en la|en los|y |que |para |con |por )', stripped, re.I))
    words = re.findall(r'\w+', stripped.lower())
    trigrams = [tuple(words[i:i+3]) for i in range(max(0, len(words)-2))]
    repetition = any(c >= 3 for c in Counter(trigrams).values())
    return dict(response=raw, generated_token_ids=tokens, EOS_generated=eos, answer_tokens=len(tokens),
                generation_budget=budget, stop_reason='EOS' if eos else 'token_budget',
                initial_fragment=initial_fragment, natural_close=bool(re.search(r'[.!?][”"\)]?$', stripped)),
                syntactically_complete_start_heuristic=bool(stripped and stripped[0].isupper() and not initial_fragment),
                obvious_repetition=repetition,
                direct_answer_vs_continuation='fragmentary_continuation' if initial_fragment else 'requires_semantic_review',
                abstention_exact=stripped == ABSTENTION,
                abstention_prefix=stripped.startswith('No tengo información suficiente'))


def generation_suite(model, ds, records=None, config=None):
    config = config or dict(max_new_tokens=48, greedy=True, seed=42)
    records = ds.records if records is None else records
    rows = [dict(id=r['id'], task_type=r['task_type'], instruction_category=r['instruction_category'],
                 source_doc=r['source_doc'], source_split=r['source_split'], question=r['question'],
                 context=r['context'], reference_answer=r['answer'], **generate(model, ds.tokenizer, r, config['max_new_tokens'])) for r in records]
    negatives = [r for r in rows if r['task_type'] == 'abstention']
    positives = [r for r in rows if r['task_type'] != 'abstention']
    def rate(items, key):
        return sum(r[key] for r in items)/len(items) if items else None
    metrics = dict(examples=len(rows), negative_examples=len(negatives),
                   EOS_generated_rate=rate(rows, 'EOS_generated'), natural_close_rate=rate(rows, 'natural_close'),
                   complete_start_heuristic_rate=rate(rows, 'syntactically_complete_start_heuristic'),
                   mean_generated_answer_tokens=sum(r['answer_tokens'] for r in rows)/len(rows),
                   abstention_exact_rate=rate(negatives, 'abstention_exact'), abstention_prefix_rate=rate(negatives, 'abstention_prefix'),
                   false_abstention_prefix_rate_on_answerable=rate(positives, 'abstention_prefix'))
    return dict(generation_config=config, metrics=metrics, examples=rows,
                limitations='Cierre e inicio son heurísticas sintácticas; respuesta directa y factualidad necesitan revisión semántica. Outputs íntegros, sin editar.')


def forgetting(retention, baseline):
    relative = (retention['loss']/baseline['loss'] - 1) * 100
    return dict(base_loss=baseline['loss'], historical_base_loss=4.725705095024419,
                sft_loss=retention['loss'], delta_loss=retention['loss']-baseline['loss'], relative_increase_percent=relative,
                warning=relative > 10, possible_catastrophic_forgetting=relative > 15)


def final_report(checkpoint, base_instruction, base_retention, base_suite, instruction, retention, suite):
    guard = forgetting(retention, base_retention)
    base_by_id = {r['id']: r for r in base_suite['examples']}
    comparison = [dict(id=r['id'], prompt_question=r['question'], reference=r['reference_answer'],
                       base=base_by_id[r['id']], sft=r) for r in suite['examples']]
    # Abstención sobre TODOS los negativos Validation, no solo los ejemplos visibles.
    ds = DatasetSFT('validation')
    model, _ = load_model(BASE / 'modelos/checkpoint_sft_best.pt')
    negative_suite = generation_suite(model, ds, [r for r in ds.records if r['task_type'] == 'abstention'])
    baseline_all_negatives = json.loads((BASE / 'resultados/baseline_sft_5A.json').read_text(encoding='utf-8'))['all_validation_abstention']
    checks = dict(instruction_loss_decreased=instruction['loss'] < base_instruction['loss'],
                  EOS_reference_accuracy_improved=instruction['EOS_accuracy_at_reference_end'] > base_instruction['EOS_accuracy_at_reference_end'],
                  generated_EOS_improved=suite['metrics']['EOS_generated_rate'] > base_suite['metrics']['EOS_generated_rate'],
                  abstention_learned=negative_suite['metrics']['abstention_prefix_rate'] > baseline_all_negatives['metrics']['abstention_prefix_rate'],
                  most_answers_close=suite['metrics']['natural_close_rate'] > .5 or suite['metrics']['EOS_generated_rate'] > .5,
                  no_severe_forgetting=not guard['possible_catastrophic_forgetting'])
    warnings = list(json.loads((SFT / 'reporte_dataset_sft_5A.json').read_text(encoding='utf-8'))['warnings'])
    if guard['warning']:
        warnings.append('WARNING: Validation LM loss aumentó más de 10%.')
    if guard['possible_catastrophic_forgetting']:
        warnings.append('Posible catastrophic forgetting: incremento superior al 15%.')
    if not all(checks.values()):
        warnings.append('Uno o más criterios conductuales no se alcanzaron.')
    # Dataset reducido y revisión semántica pendientes impiden declarar validez automática.
    warnings.append('La muestra instruccional es pequeña y la evaluación de factualidad no está certificada: requiere revisión antes de 5B.')
    total_steps = Counter()
    for row in checkpoint['history']:
        total_steps.update(row['step_counts'])
    distribution = {k: dict(steps=v, percent=100*v/sum(total_steps.values())) for k, v in total_steps.items()}
    return dict(version='5A', status='official_training_completed', base_checkpoint='modelos/checkpoint_best.pt',
        base_checkpoint_sha256=checkpoint['base_checkpoint_hash'], best_sft_checkpoint='modelos/checkpoint_sft_best.pt',
        best_epoch=checkpoint['epoch'], training_config=checkpoint['training_config'], mixed_step_distribution=distribution,
        history=checkpoint['history'], best_validation_instruction_loss=instruction['loss'], best_validation_instruction_perplexity=instruction['perplexity'],
        EOS_metrics=dict(base=base_instruction, sft=instruction, generation_base=base_suite['metrics'], generation_sft=suite['metrics']),
        abstention_metrics=dict(base=baseline_all_negatives, sft=negative_suite), original_LM_retention_metrics=dict(base=base_retention, sft=retention),
        catastrophic_forgetting_assessment=guard, base_vs_sft_examples=comparison,
        acceptance_checks=checks, base_instruction_metrics=base_instruction, validation_instruction_metrics=instruction,
        weights_modified_base_checkpoint=False, test_used=False, warnings=warnings,
        verdict='INSTRUCTION TUNING MIXTO REQUIERE REVISIÓN',
        factuality_semantic_review_completed=False, protected_hashes=checkpoint['protected_hashes'],
        hashes=checkpoint['hashes'], scope='5A enseña comportamiento y formato. 5B debe aportar evidencia recuperada y criterio externo de suficiencia. RAG/chat no implementados.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    install_test_guard()
    torch.set_num_threads(2)
    before = frozen_hashes()
    model, cp = load_model(BASE / 'modelos/checkpoint_sft_best.pt')
    require(cp['hashes'] == before, 'Proveniencia del checkpoint modificada.')
    require(all(sha256(BASE / p) == h for p, h in cp['protected_hashes'].items()), 'Archivo protegido modificado.')
    last = torch.load(BASE / 'modelos/checkpoint_sft_last.pt', map_location='cpu', weights_only=True)
    require(last['hashes'] == before, 'Last de otro run.')
    cp['history'] = last['history']
    baseline = json.loads((BASE / 'resultados/baseline_sft_5A.json').read_text(encoding='utf-8'))
    fixed = json.loads((SFT / 'prompts_validation_5A.json').read_text(encoding='utf-8'))
    ds = DatasetSFT('validation')
    instruction = evaluate_tokens(model, ds)
    retention = evaluate_tokens(model, causal_dataset('validation'))
    suite = generation_suite(model, ds, fixed['records'], fixed['generation_config'])
    report = final_report(cp, baseline['instruction'], baseline['retention'], baseline['generation'], instruction, retention, suite)
    require(frozen_hashes() == before, 'Artefactos base/datos modificados.')
    write_json(BASE / 'resultados/reporte_entrenamiento_sft_5A.json', report)
    print(json.dumps(dict(best_epoch=report['best_epoch'], instruction_loss=instruction['loss'], retention_loss=retention['loss'], verdict=report['verdict']), indent=2))


if __name__ == '__main__':
    main()
