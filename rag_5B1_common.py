"""Aislamiento 5B.1: protección de 5B y acceso heredado que bloquea Test temprano."""
import ast
import atexit
import json
import os
import sys
import types
from pathlib import Path
import rag_common as previous
from rag_common import BASE, DATA, RESULTS, require, sha256, write_json, load_runtime

STAGE = BASE/'datos/rag_5B1'
LOCK = STAGE/'integridad_previa_5B1.json'
atexit.unregister(previous.save_access_log)

def initialize():
    require(not LOCK.exists(),'No reemplazar baseline 5B.1')
    previous.verify_preservation()
    protected=set(json.loads(previous.MANIFEST.read_text(encoding='utf-8'))['protected_hashes'])
    protected.update(p.relative_to(BASE).as_posix() for p in DATA.glob('*') if p.is_file())
    protected.update(p.relative_to(BASE).as_posix() for p in RESULTS.glob('*5B*') if p.is_file() and '5B1' not in p.name)
    protected.update(('rag_common.py','retrieval.py','construir_indice_rag.py','evaluar_rag_5B.py','probar_rag_5B.py','documentar_rag_5B.py','README_RAG_5B.md'))
    STAGE.mkdir(parents=True,exist_ok=True)
    (STAGE/'rag_5B_original.py.txt').write_bytes((BASE/'rag.py').read_bytes())
    baseline=dict(version='5B.1',protected_hashes={p:sha256(BASE/p) for p in sorted(protected)},
        authorized_changes=['rag.py'],original_rag_sha256=sha256(BASE/'rag.py'),test_content_reads=0)
    write_json(LOCK,baseline)

if LOCK.exists():
    BASELINE=json.loads(LOCK.read_text(encoding='utf-8'))
    previous.READONLY.update(previous.normalized_path(BASE/p) for p in BASELINE['protected_hashes'])
    previous.READONLY.update(previous.normalized_path(p) for p in (LOCK,STAGE/'rag_5B_original.py.txt'))

def preservation():
    b=json.loads(LOCK.read_text(encoding='utf-8'))
    changed=[p for p,h in b['protected_hashes'].items() if sha256(BASE/p)!=h]
    require(not changed,'Artefactos protegidos modificados: '+str(changed))
    require(sha256(STAGE/'rag_5B_original.py.txt')==b['original_rag_sha256'],'Copia de baseline RAG modificada')
    return dict(previous_artifacts_preserved=True,protected_files=len(b['protected_hashes']),
        authorized_rag_change=dict(before=b['original_rag_sha256'],after=sha256(BASE/'rag.py')),
        test_documents_used=0,test_content_reads=0)

def sft_functions():
    """Extraer funciones reales, sin ejecutar el módulo de entrenamiento/preparación."""
    source=ast.parse((BASE/'dataset_sft_5A2.py').read_text(encoding='utf-8'))
    definitions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('prompt','encode')]
    require(len(definitions)==2,'Funciones SFT no encontradas')
    namespace=dict(require=require)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(BASE/'dataset_sft_5A2.py'),'exec'),namespace)
    return namespace

def sft_prompt(context,question):
    return sft_functions()['prompt'](dict(context=context,question=question))

def legacy_functions(serialization='original'):
    source=ast.parse((STAGE/'rag_5B_original.py.txt').read_text(encoding='utf-8'))
    definitions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('prompt','pack_context','greedy_generate')]
    namespace=dict(require=require)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(STAGE/'rag_5B_original.py.txt'),'exec'),namespace)
    if serialization=='sft': namespace['prompt']=sft_prompt
    return namespace

def log_accesses():
    unique={(r['path'],r['operation'],r['allowed']) for r in previous.LOG}
    write_json(RESULTS/f'accesos_rag_5B1_{Path(sys.argv[0]).stem}_{os.getpid()}.json',
        dict(guard_active=True,test_content_reads=sum(r['allowed'] and r['path'] in previous.FORBIDDEN for r in previous.LOG),
             test_documents_used=0,accesses=[dict(path=p,operation=o,allowed=a) for p,o,a in sorted(unique)]))
atexit.register(log_accesses)

if __name__=='__main__':
    initialize() if not LOCK.exists() else preservation()
    print('Baseline 5B.1 registrada; Test cerrado.')
