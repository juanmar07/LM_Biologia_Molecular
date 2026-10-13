"""Infraestructura aislada 5A.2: acceso protegido, targets de respuesta y mezcla."""
import ast
import hashlib
import json
import os
import random
import sys
import types
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'datos/sft_5A2'
RESULTS = BASE / 'resultados'
BASE_HASH = '3bcd3074e1020380b75f16ea0917bb0074f1961ed6937d66f93f2f52618c0cde'
ABSTENTIONS = ('No tengo información suficiente en el contexto proporcionado para responder con seguridad.',
               'El contexto proporcionado no contiene información suficiente para responder con seguridad.')
TASKS = ('extractive_grounded', 'paraphrastic_grounded', 'abstention')
CODE = ('preparar_sft_5A2.py', 'dataset_sft_5A2.py', 'entrenar_sft_5A2.py', 'evaluar_sft_5A2.py', 'probar_sft_5A2.py', 'curacion_sft_5A2.json')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def install_guard():
    meta = json.loads((BASE / 'datos/splits.json').read_text(encoding='utf-8'))
    norm = lambda p: os.path.normcase(str(Path(p).resolve()))
    banned = {norm(BASE / p) for p in ('datos/dataset/test.pt', 'datos/splits/test.txt', 'datos/corpus_limpio.txt', 'datos/reporte_corpus.json')}
    test = set(meta['splits']['test']['sources'])
    banned.update(norm(BASE / d['clean_file']) for d in meta['inventory'] if d['source'] in test)
    preservation = json.loads((DATA / 'integridad_previa_5A_5A1.json').read_text(encoding='utf-8'))
    readonly = {norm(BASE / p) for p in preservation['protected_hashes']}
    readonly.add(norm(DATA/'integridad_previa_5A_5A1.json'))
    if (DATA/'dataset_lock_5A2.json').exists():
        readonly.update(norm(DATA/p) for p in ('train_instrucciones_5A2.jsonl','validation_instrucciones_5A2.jsonl',
            'pares_contrastivos.jsonl','auditoria_evidencia.jsonl','suite_validation_5A2.json',
            'dataset_lock_5A2.json','reporte_dataset_sft_5A2.json'))
    corpus_prefix = norm(BASE / 'corpus') + os.sep
    previous_data_prefix = norm(BASE / 'datos/sft') + os.sep
    accesses = [dict(path=norm(BASE/'datos/splits.json'),operation='read'),
                dict(path=norm(DATA/'integridad_previa_5A_5A1.json'),operation='read')]

    def hook(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = norm(os.fsdecode(args[0]))
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            require(p not in banned and not p.startswith(corpus_prefix), 'Lectura/acceso Test/PDF prohibido: ' + p)
            require(not writing or (p not in readonly and not p.startswith(previous_data_prefix)), 'Escritura protegida 5A/base: ' + p)
            if p.startswith(norm(BASE) + os.sep):
                accesses.append(dict(path=p, operation='write' if writing else 'read'))
        elif event in ('os.rename', 'os.remove'):
            for value in args[:2 if event == 'os.rename' else 1]:
                if isinstance(value, (str, bytes, os.PathLike)):
                    p = norm(os.fsdecode(value))
                    require(p not in readonly and not p.startswith(previous_data_prefix) and not p.startswith(corpus_prefix) and p not in banned, 'Mutación de archivo protegido: ' + p)
    sys.addaudithook(hook)
    return accesses


ACCESS_LOG = install_guard()

# Importaciones después del guard; no ejecutan preparadores/evaluadores antiguos.
import torch
from torch.utils.data import Dataset
from tokenizador import TokenizadorEducativo
from dataset import DatasetSecuencias


def verify_preservation():
    original = json.loads((DATA / 'integridad_previa_5A_5A1.json').read_text(encoding='utf-8'))['protected_hashes']
    require(all(sha256(BASE / p) == h for p, h in original.items()), 'Experimento 5A/base modificado.')
    require(sha256(BASE / 'modelos/checkpoint_best.pt') == BASE_HASH, 'Checkpoint base original cambió.')
    return original


def documents():
    meta = json.loads((BASE / 'datos/splits.json').read_text(encoding='utf-8'))
    sets = {s: set(meta['splits'][s]['sources']) for s in ('train', 'validation', 'test')}
    require(not any(sets[a] & sets[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]), 'Splits solapados.')
    inventory = {d['source']: d for d in meta['inventory']}
    families = {s: f['id'] for f in meta['families'] for s in f['documents']}
    result = {}
    for split in ('train', 'validation'):
        result[split] = []
        for source in sorted(sets[split]):
            row = inventory[source]
            p = (BASE / row['clean_file']).resolve()
            require(p.parent == (BASE / 'datos/documentos_limpios').resolve(), 'Ruta documental inválida.')
            require(sha256(p) == row['sha256'], 'TXT limpio modificado.')
            result[split].append(dict(source=source, split=split, family=families[source], sha256=row['sha256'], text=p.read_text(encoding='utf-8')))
    return result


def prompt(r):
    return ('Contexto:\n' + r['context'] + '\n\n') + 'Usuario:\n' + r['question'] + '\n\nAsistente:\n'


def encode(r, tokenizer):
    p, a = tokenizer.encode(prompt(r)), tokenizer.encode(r['answer'])
    ids = [2] + p + a + [3]
    require(ids == tokenizer.encode(prompt(r) + r['answer'], agregar_bos=True, agregar_eos=True), 'Frontera BPE inconsistente.')
    require(len(ids) <= 128 and a and all(4 <= t < 6000 for t in p+a), 'Presupuesto/specials/IDs inválidos; nunca truncar.')
    x = torch.zeros(128, dtype=torch.long)
    y = torch.full((128,), -100, dtype=torch.long)
    x[:len(ids)-1] = torch.tensor(ids[:-1])
    first = len(p)
    y[first:first+len(a)+1] = torch.tensor(a+[3])
    require(y[first].item() == a[0] and y[first+len(a)].item() == 3, 'Primer/último target erróneos.')
    require((y == 3).sum().item() == 1 and (y[x == 0] == -100).all().item(), 'EOS/PAD incorrectos.')
    return x, y


class DatasetSFT5A2(Dataset):
    def __init__(self, split):
        require(split in ('train', 'validation'), 'Test prohibido.')
        self.split = split
        self.tokenizer = TokenizadorEducativo().cargar()
        self.records = [json.loads(line) for line in (DATA / f'{split}_instrucciones_5A2.jsonl').read_text(encoding='utf-8').splitlines()]
        require(self.records and self.tokenizer.vocab_size == 6000, 'Datos/vocabulario incorrectos.')
        require(all(r['split'] == r['source_split'] == split for r in self.records), 'Split mezclado.')
        self.encoded = [encode(r, self.tokenizer) for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.encoded[i]


def causal_dataset(split):
    require(split in ('train', 'validation'), 'Test prohibido.')
    ds = DatasetSecuencias(split, ruta=BASE / f'datos/dataset/{split}.pt')
    meta = json.loads((BASE / 'datos/splits.json').read_text(encoding='utf-8'))
    require({d['source'] for d in ds.documents} == set(meta['splits'][split]['sources']), 'Procedencia causal incorrecta.')
    require(ds.context == 128, 'Longitud causal incorrecta.')
    return ds


def load_model(which='base'):
    require(which in ('base', 'best', 'last'), 'Checkpoint 5A.2 no permitido.')
    name = 'checkpoint_best.pt' if which == 'base' else f'checkpoint_sft_5A2_{which}.pt'
    path = BASE / 'modelos' / name
    if which == 'base':
        require(sha256(path) == BASE_HASH, 'Base incorrecta.')
    cp = torch.load(path, weights_only=True, map_location='cpu')
    mc = cp['model_config']
    require(mc['vocab_size'] == 6000 and mc['context_length'] == 128 and mc['weight_tying'], 'Arquitectura incompatible.')
    if which == 'base':
        require(cp['epoch_completed'] == 19 and cp['model_code_sha256'] == sha256(BASE / 'modelo.py'), 'Base/código incompatible.')
    else:
        require(cp['version'] == '5A.2' and cp['base_checkpoint_sha256'] == BASE_HASH, 'Run distinto.')
    constants = {n: mc[k] for n, k in {'LONGITUD_CONTEXTO':'context_length', 'DIMENSION_EMBEDDING':'d_model',
        'NUM_CABEZAS':'n_heads', 'NUM_CAPAS':'n_layers', 'DIMENSION_FEED_FORWARD':'d_ff', 'DROPOUT':'dropout'}.items()}
    source = ast.parse((BASE / 'modelo.py').read_text(encoding='utf-8'))
    source.body = [n for n in source.body if not (isinstance(n, ast.ImportFrom) and n.module == 'config')]
    namespace = types.ModuleType('minigpt_sft_5A2')
    namespace.__dict__.update(constants)
    with torch.random.fork_rng(devices=[]):
        exec(compile(source, str(BASE/'modelo.py'), 'exec'), namespace.__dict__)
        model = namespace.MiniGPT(6000)
    require(torch.equal(cp['model_state_dict']['embedding_tokens.weight'], cp['model_state_dict']['salida_vocabulario.weight']), 'Aliases distintos.')
    model.load_state_dict(cp['model_state_dict'], strict=True)
    require(model.embedding_tokens.weight is model.salida_vocabulario.weight, 'Weight tying perdido.')
    require(sum(p.numel() for p in model.parameters()) == 1379456 and all(torch.isfinite(p).all().item() for p in model.parameters()), 'Parámetros incompatibles/no finitos.')
    return model.float(), cp


def hashes(include_code=True):
    paths = {'base_checkpoint':BASE/'modelos/checkpoint_best.pt', 'tokenizer':BASE/'datos/tokenizer/tokenizer.json',
             'splits':BASE/'datos/splits.json', 'causal_train':BASE/'datos/dataset/train.pt', 'causal_validation':BASE/'datos/dataset/validation.pt'}
    paths.update({n: DATA/f for n, f in {'train':'train_instrucciones_5A2.jsonl', 'validation':'validation_instrucciones_5A2.jsonl',
                 'pairs':'pares_contrastivos.jsonl', 'evidence_audit':'auditoria_evidencia.jsonl', 'suite':'suite_validation_5A2.json'}.items()})
    if include_code:
        paths.update({p:BASE/p for p in CODE})
    return {k:sha256(p) for k, p in paths.items()}


def verify_lock():
    lock = json.loads((DATA/'dataset_lock_5A2.json').read_text(encoding='utf-8'))
    require(lock['hashes'] == hashes(False), 'Dataset 5A.2/base congelados cambiaron.')
    verify_preservation()
    return hashes()


def schedule(ds, epoch=1, batch_size=16):
    """Una pasada instruccional; ajuste determinista de batches incompletos, sin repetir registros."""
    rng = random.Random(42+epoch)
    indices = {t:[i for i,r in enumerate(ds.records) if r['task_type']==t] for t in TASKS}
    # Múltiplo de 20 pasos; capacidades n_t <= pasos_t * batch_size.
    import math
    cycles = max(math.ceil(len(indices[t])/(quota*batch_size)) for t,quota in zip(TASKS,(15,2,1)))
    batches = []
    for t, quota in zip(TASKS, (15,2,1)):
        rng.shuffle(indices[t])
        count = cycles*quota
        require(len(indices[t]) >= count, 'Muy pocos registros para batches no vacíos de la mezcla.')
        # Balancea tamaños, en vez de sobrerrepresentar la tarea minoritaria.
        groups = [indices[t][j::count] for j in range(count)]
        require(max(map(len, groups)) <= batch_size, 'Batch supera tamaño configurado.')
        batches.extend((t,g) for g in groups)
    batches.extend(('causal_retention',None) for _ in range(cycles*2))
    rng.shuffle(batches)
    return batches


def log_accesses(mode):
    unique = {(r['path'],r['operation']) for r in ACCESS_LOG}
    write_json(RESULTS/f'accesos_sft_5A2_{mode}.json', dict(version='5A.2', test_documents_used=0, guard_active=True,
               logging_scope='workspace opens after early guard installation plus the two explicit bootstrap metadata reads; library files outside workspace omitted',
               accesses=[dict(path=p,operation=o) for p,o in sorted(unique)]))
