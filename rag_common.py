"""Infraestructura exclusivamente local de 5B. El guard precede a torch/tokenizador."""
import ast
import atexit
import hashlib
import json
import os
import sys
import types
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'datos/rag'
RESULTS = BASE / 'resultados'
MANIFEST = DATA / 'integridad_previa_5B.json'
CHECKPOINT = BASE / 'modelos/checkpoint_sft_5A2_best.pt'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)

SPLITS = json.loads((BASE/'datos/splits.json').read_text(encoding='utf-8'))
SOURCES = {s: set(SPLITS['splits'][s]['sources']) for s in ('train','validation','test')}
require(not any(SOURCES[a] & SOURCES[b] for a,b in [('train','validation'),('train','test'),('validation','test')]), 'Splits solapados')
def normalized_path(p):
    return os.path.normcase(str(Path(p).resolve()))

FORBIDDEN = {normalized_path(BASE/p) for p in ('datos/dataset/test.pt','datos/splits/test.txt','datos/corpus_limpio.txt','datos/reporte_corpus.json')}
FORBIDDEN.update(normalized_path(BASE/d['clean_file']) for d in SPLITS['inventory'] if d['source'] in SOURCES['test'])
# Names and directory entries only; never open/hash a Test document.
old_paths = set()
for pattern in ('*.py','*SFT*','*sft*','curacion*.json','modelos/*','resultados/*','datos/sft/**/*','datos/sft_5A1/**/*','datos/sft_5A2/**/*','datos/tokenizer/**/*'):
    old_paths.update(p for p in BASE.glob(pattern) if p.is_file() and 'rag' not in p.name.lower() and '5B' not in p.name)
old_paths.update(BASE/p for p in ('datos/splits.json','datos/dataset/train.pt','datos/dataset/validation.pt'))
old_paths.update(BASE/d['clean_file'] for d in SPLITS['inventory'] if d['source'] in SOURCES['train'] | SOURCES['validation'])
old_paths = {p for p in old_paths if normalized_path(p) not in FORBIDDEN and p.name not in ('retrieval.py','construir_indice_rag.py','evaluar_rag_5B.py')}
if MANIFEST.exists():
    old_paths = {BASE/p for p in json.loads(MANIFEST.read_text(encoding='utf-8'))['protected_hashes']}
READONLY = {normalized_path(p) for p in old_paths}
READONLY.add(normalized_path(MANIFEST))
LOG = [dict(path=normalized_path(BASE/'datos/splits.json'),operation='read',allowed=True)]
if MANIFEST.exists():
    LOG.append(dict(path=normalized_path(MANIFEST),operation='read',allowed=True))

def guard(event, args):
    if event == 'open' and isinstance(args[0], (str,bytes,os.PathLike)):
        p = normalized_path(os.fsdecode(args[0]))
        mode, flags = args[1:3]
        writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        forbidden = p in FORBIDDEN or p.startswith(normalized_path(BASE/'corpus')+os.sep) or p.endswith('.pdf')
        protected = writing and p in READONLY
        if p.startswith(normalized_path(BASE)+os.sep):
            LOG.append(dict(path=p,operation='write' if writing else 'read',allowed=not (forbidden or protected)))
        require(not forbidden, 'Test/PDF/corpus global prohibido: '+p)
        require(not protected, 'Escritura de etapa previa prohibida: '+p)
    elif event in ('os.rename','os.remove'):
        for value in args[:2 if event == 'os.rename' else 1]:
            if isinstance(value,(str,bytes,os.PathLike)):
                p = normalized_path(os.fsdecode(value))
                require(p not in READONLY and p not in FORBIDDEN and not p.endswith('.pdf'), 'Mutación protegida: '+p)
sys.addaudithook(guard)

def initialize_manifest():
    require(not MANIFEST.exists(), 'Manifest ya existe; no reemplazar línea base')
    hashes = {p.relative_to(BASE).as_posix():sha256(p) for p in sorted(old_paths)}
    # Bootstrap creates its own immutable baseline through a narrowly scoped raw write.
    READONLY.remove(normalized_path(MANIFEST))
    try:
        write_json(MANIFEST, dict(version='5B',protected_hashes=hashes,test_content_reads=0,test_documents_used=0))
    finally:
        READONLY.add(normalized_path(MANIFEST))
    return len(hashes)

def verify_preservation():
    hashes = json.loads(MANIFEST.read_text(encoding='utf-8'))['protected_hashes']
    changed = [p for p,h in hashes.items() if sha256(BASE/p) != h]
    require(not changed, 'Archivos previos cambiaron: '+str(changed))
    return dict(previous_artifacts_preserved=True,protected_files=len(hashes),test_documents_used=0,test_content_reads=0)

def save_access_log():
    unique = {(r['path'],r['operation'],r['allowed']) for r in LOG}
    write_json(RESULTS / ('accesos_rag_5B_'+Path(sys.argv[0]).stem+'_'+str(os.getpid())+'.json'),
        dict(guard_active=True,test_content_reads=sum(r['allowed'] and r['path'] in FORBIDDEN for r in LOG),
             test_documents_used=0,bootstrap_metadata_only=True,
             accesses=[dict(path=p,operation=o,allowed=a) for p,o,a in sorted(unique)]))
atexit.register(save_access_log)

def documents():
    inventory = {d['source']:d for d in SPLITS['inventory']}
    result = []
    for split in ('train','validation'):
        for source in sorted(SOURCES[split]):
            d = inventory[source]
            p = (BASE/d['clean_file']).resolve()
            require(p.parent == (BASE/'datos/documentos_limpios').resolve(), 'Ruta TXT no canónica')
            require(sha256(p) == d['sha256'], 'Documento limpio cambiado')
            result.append(dict(source_doc=source,split=split,clean_file=d['clean_file'],sha256=d['sha256'],text=p.read_text(encoding='utf-8')))
    return result

def load_runtime(checkpoint=None):
    import torch
    from tokenizador import TokenizadorEducativo
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    path = Path(checkpoint).resolve() if checkpoint else CHECKPOINT
    require(path.parent == (BASE/'modelos').resolve() and path.name in ('checkpoint_sft_5A2_best.pt','checkpoint_sft_5A2_last.pt'), 'Solo checkpoints 5A.2 locales protegidos')
    baseline = json.loads(MANIFEST.read_text(encoding='utf-8'))['protected_hashes']
    require(sha256(path) == baseline[path.relative_to(BASE).as_posix()], 'Checkpoint alterado')
    cp = torch.load(path, map_location='cpu', weights_only=True)
    mc = cp['model_config']
    require(cp['version'] == '5A.2' and mc['vocab_size']==6000 and mc['context_length']==128 and mc['weight_tying'], 'Modelo incompatible')
    tokenizer = TokenizadorEducativo().cargar()
    require(cp['tokenizer_sha256'] == sha256(BASE/'datos/tokenizer/tokenizer.json') and tokenizer.vocab_size==6000, 'Tokenizador incompatible')
    constants = {n:mc[k] for n,k in {'LONGITUD_CONTEXTO':'context_length','DIMENSION_EMBEDDING':'d_model','NUM_CABEZAS':'n_heads','NUM_CAPAS':'n_layers','DIMENSION_FEED_FORWARD':'d_ff','DROPOUT':'dropout'}.items()}
    source = ast.parse((BASE/'modelo.py').read_text(encoding='utf-8'))
    source.body = [n for n in source.body if not (isinstance(n,ast.ImportFrom) and n.module=='config')]
    namespace = types.ModuleType('minigpt_rag_5B')
    namespace.__dict__.update(constants)
    exec(compile(source,str(BASE/'modelo.py'),'exec'),namespace.__dict__)
    model = namespace.MiniGPT(6000)
    require(torch.equal(cp['model_state_dict']['embedding_tokens.weight'],cp['model_state_dict']['salida_vocabulario.weight']), 'Weight aliases diferentes')
    model.load_state_dict(cp['model_state_dict'], strict=True)
    require(model.embedding_tokens.weight is model.salida_vocabulario.weight and sum(p.numel() for p in model.parameters())==1379456, 'Arquitectura distinta')
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'Pesos no finitos')
    model.eval()
    model.requires_grad_(False)
    return model,tokenizer,dict(checkpoint=str(path),checkpoint_sha256=sha256(path),epoch=cp['epoch'],strict=True,parameters=1379456,seed=42)

if __name__ == '__main__':
    print('Protected files:', initialize_manifest() if not MANIFEST.exists() else verify_preservation())
