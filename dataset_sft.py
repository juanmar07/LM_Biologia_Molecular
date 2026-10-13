"""5A: datos y acceso explícitamente limitado a Train/Validation."""
import ast
import hashlib
import json
import random
import types
import sys
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizador import TokenizadorEducativo
from dataset import DatasetSecuencias

BASE = Path(__file__).resolve().parent
SFT = BASE / 'datos/sft'
ABSTENTION = 'No tengo información suficiente en el contexto proporcionado para responder con seguridad.'
TASKS = ('grounded_instruction', 'direct_instruction', 'abstention')


def install_test_guard():
    """Bloquea incluso lectura indirecta/hash de Test dentro de los procesos 5A."""
    meta = json.loads((BASE / 'datos/splits.json').read_text(encoding='utf-8'))
    forbidden = {os.path.normcase(str((BASE / p).resolve())) for p in
                 ('datos/dataset/test.pt', 'datos/splits/test.txt', 'datos/corpus_limpio.txt', 'datos/reporte_corpus.json')}
    test_sources = set(meta['splits']['test']['sources'])
    forbidden.update(os.path.normcase(str((BASE / d['clean_file']).resolve())) for d in meta['inventory'] if d['source'] in test_sources)
    corpus = os.path.normcase(str((BASE / 'corpus').resolve())) + os.sep
    log = set()

    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            require(path not in forbidden and not path.startswith(corpus), 'Acceso Test/PDF/corpus combinado prohibido en 5A: ' + path)
            if path.startswith(os.path.normcase(str(BASE)) + os.sep):
                log.add(path)
    sys.addaudithook(guard)
    return log


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_documents():
    # Se lee solo metadata de asignación. Nunca llamar verify_inputs/leer_documentos antiguos.
    meta = json.loads((BASE / 'datos/splits.json').read_text(encoding='utf-8'))
    sets = {s: set(meta['splits'][s]['sources']) for s in ('train', 'validation', 'test')}
    require(not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test']), 'Splits superpuestos.')
    inventory = {d['source']: d for d in meta['inventory']}
    families = {s: f['id'] for f in meta['families'] for s in f['documents']}
    docs = {}
    for split in ('train', 'validation'):
        docs[split] = []
        for source in sorted(sets[split]):
            row = inventory[source]
            path = (BASE / row['clean_file']).resolve()
            require(path.parent == (BASE / 'datos/documentos_limpios').resolve(), 'Ruta no autorizada.')
            require(sha256(path) == row['sha256'], 'TXT fuente modificado: ' + source)
            docs[split].append(dict(source=source, family=families[source], text=path.read_text(encoding='utf-8'), sha256=row['sha256']))
    return docs


def prompt(record):
    prefix = '' if record['task_type'] == 'direct_instruction' else 'Contexto:\n' + record['context'] + '\n\n'
    return prefix + 'Usuario:\n' + record['question'] + '\n\nAsistente:\n'


def encode_record(record, tokenizer):
    # El separador acaba en newline: segmentar respuesta aparte conserva el BPE.
    p = tokenizer.encode(prompt(record))
    a = tokenizer.encode(record['answer'])
    ids = [2] + p + a + [3]
    require(ids == tokenizer.encode(prompt(record) + record['answer'], agregar_bos=True, agregar_eos=True), 'Frontera BPE alterada.')
    require(len(ids) <= 128 and len(a) > 0, 'Ejemplo fuera del presupuesto; no se trunca.')
    require(all(4 <= i < 6000 for i in p + a), 'Special/ID inesperado en texto.')
    first = 1 + len(p)
    x = torch.zeros(128, dtype=torch.long)
    y = torch.full((128,), -100, dtype=torch.long)
    x[:len(ids)-1] = torch.tensor(ids[:-1], dtype=torch.long)
    y[first-1:len(ids)-1] = torch.tensor(ids[first:], dtype=torch.long)
    require(y[first-1].item() == a[0] and y[len(ids)-2].item() == 3, 'Desplazamiento de targets incorrecto.')
    require((y == 3).sum().item() == 1 and (y[x == 0] == -100).all().item(), 'EOS/PAD incorrectos.')
    return x, y


class DatasetSFT(Dataset):
    def __init__(self, split):
        require(split in ('train', 'validation'), 'Test prohibido en 5A.')
        self.split = split
        self.tokenizer = TokenizadorEducativo().cargar()
        self.records = [json.loads(line) for line in (SFT / f'{split}_instrucciones.jsonl').read_text(encoding='utf-8').splitlines()]
        require(self.records and self.tokenizer.vocab_size == 6000, 'Datos/vocabulario inválidos.')
        require(all(r['split'] == split and r['source_split'] == split for r in self.records), 'Split instruccional incorrecto.')
        self.encoded = [encode_record(r, self.tokenizer) for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.encoded[index]


def causal_dataset(split):
    require(split in ('train', 'validation'), 'Test prohibido en 5A.')
    ds = DatasetSecuencias(split=split, ruta=BASE / 'datos/dataset' / (split + '.pt'))
    meta = json.loads((BASE / 'datos/splits.json').read_text(encoding='utf-8'))
    require({d['source'] for d in ds.documents} == set(meta['splits'][split]['sources']), 'Procedencia causal incorrecta.')
    require(ds.context == 128, 'Contexto causal incompatible.')
    return ds


def load_model(path=None):
    path = Path(path or BASE / 'modelos/checkpoint_best.pt')
    require(path.resolve() in {(BASE / 'modelos' / n).resolve() for n in ('checkpoint_best.pt', 'checkpoint_sft_best.pt', 'checkpoint_sft_last.pt')}, 'Checkpoint no permitido.')
    cp = torch.load(path, map_location='cpu', weights_only=True)
    mc = cp['model_config']
    require(mc['vocab_size'] == 6000 and mc['context_length'] == 128 and mc['weight_tying'], 'Arquitectura incompatible.')
    if path.name == 'checkpoint_best.pt':
        require(cp['epoch_completed'] == 19 and cp['best_validation_loss'] == 4.725705095024419, 'Checkpoint base distinto de época 19.')
        require(cp['model_code_sha256'] == sha256(BASE / 'modelo.py'), 'Código base modificado.')
    constants = {name: mc[key] for name, key in {
        'LONGITUD_CONTEXTO': 'context_length', 'DIMENSION_EMBEDDING': 'd_model',
        'NUM_CABEZAS': 'n_heads', 'NUM_CAPAS': 'n_layers',
        'DIMENSION_FEED_FORWARD': 'd_ff', 'DROPOUT': 'dropout'}.items()}
    source = ast.parse((BASE / 'modelo.py').read_text(encoding='utf-8'))
    source.body = [n for n in source.body if not (isinstance(n, ast.ImportFrom) and n.module == 'config')]
    namespace = types.ModuleType('minigpt_sft_architecture')
    namespace.__dict__.update(constants)
    with torch.random.fork_rng(devices=[]):
        exec(compile(source, str(BASE / 'modelo.py'), 'exec'), namespace.__dict__)
        model = namespace.MiniGPT(mc['vocab_size'])
    state = cp['model_state_dict']
    require(torch.equal(state['embedding_tokens.weight'], state['salida_vocabulario.weight']), 'Aliases guardados distintos.')
    model.load_state_dict(state, strict=True)
    require(sum(p.numel() for p in model.parameters()) == 1379456, 'Parámetros inesperados.')
    require(model.embedding_tokens.weight is model.salida_vocabulario.weight, 'Weight tying perdido.')
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'Pesos no finitos.')
    return model.float(), cp


def mixed_schedule(ds, batch_size=16, seed=42, epoch=1):
    """Una pasada por instrucciones; batches homogéneos, reparto 45/10/15/30."""
    rng = random.Random(seed + epoch)
    batches = []
    for task in TASKS:
        indices = [i for i, r in enumerate(ds.records) if r['task_type'] == task]
        rng.shuffle(indices)
        batches.extend((task, indices[j:j+batch_size]) for j in range(0, len(indices), batch_size))
    causal_steps = round(len(batches) * 3 / 7)
    batches.extend(('causal_language_model_retention', None) for _ in range(causal_steps))
    rng.shuffle(batches)
    return batches


def frozen_hashes():
    paths = {'base_checkpoint': BASE / 'modelos/checkpoint_best.pt', 'tokenizer': BASE / 'datos/tokenizer/tokenizer.json',
             'train': SFT / 'train_instrucciones.jsonl', 'validation': SFT / 'validation_instrucciones.jsonl',
             'generation_prompts': SFT / 'prompts_validation_5A.json', 'splits': BASE / 'datos/splits.json',
             'causal_train': BASE / 'datos/dataset/train.pt', 'causal_validation': BASE / 'datos/dataset/validation.pt',
             'model_code': BASE / 'modelo.py'}
    result = {k: sha256(p) for k, p in paths.items()}
    report = json.loads((SFT / 'reporte_dataset_sft_5A.json').read_text(encoding='utf-8'))
    require(result['train'] == report['train_sha256'] and result['validation'] == report['validation_sha256'], 'Dataset congelado modificado.')
    require(result['tokenizer'] == report['tokenizer_sha256'] and result['splits'] == report['splits_sha256'], 'Procedencia modificada.')
    return result
