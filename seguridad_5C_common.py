"""Guard e integridad 5C; ningún artefacto previo se modifica."""
import atexit
import json
import os
import sys
from pathlib import Path
import rag_5B1_common as previous_stage
from rag_5B1_common import BASE,DATA,RESULTS,require,sha256,write_json
from rag_common import READONLY,normalized_path,FORBIDDEN,LOG,CHECKPOINT

STAGE=BASE/'datos/evaluacion_5C'
MANIFEST=STAGE/'integridad_previa_5C.json'
atexit.unregister(previous_stage.log_accesses)

def initialize():
    require(not MANIFEST.exists(),'No reemplazar baseline 5C')
    previous_stage.preservation()
    protected=set(previous_stage.BASELINE['protected_hashes'])
    protected.update(p.relative_to(BASE).as_posix() for p in BASE.glob('*.py') if '5C' not in p.name and p.name not in ('respuesta_segura.py','chat_final.py'))
    protected.add('README_RAG_5B1.md')
    protected.update(p.relative_to(BASE).as_posix() for p in previous_stage.STAGE.glob('*') if p.is_file())
    protected.update(p.relative_to(BASE).as_posix() for p in RESULTS.glob('*5B1*') if p.is_file())
    write_json(MANIFEST,dict(protected_hashes={p:sha256(BASE/p) for p in sorted(protected)},
        checkpoint_sha256=sha256(CHECKPOINT),test_documents_used=0,test_content_reads=0))

if MANIFEST.exists():
    BASELINE=json.loads(MANIFEST.read_text(encoding='utf-8'))
    READONLY.update(normalized_path(BASE/p) for p in BASELINE['protected_hashes'])
    READONLY.add(normalized_path(MANIFEST))

def preservation():
    baseline=json.loads(MANIFEST.read_text(encoding='utf-8'))
    changed=[p for p,h in baseline['protected_hashes'].items() if sha256(BASE/p)!=h]
    require(not changed,'Etapas previas modificadas: '+str(changed))
    after=sha256(CHECKPOINT)
    require(after==baseline['checkpoint_sha256'],'Checkpoint modificado')
    return dict(previous_artifacts_preserved=True,protected_files=len(baseline['protected_hashes']),
        checkpoint_before=baseline['checkpoint_sha256'],checkpoint_after=after,
        test_documents_used=0,test_content_reads=0,index_rebuilt=False)

def log_accesses():
    unique={(r['path'],r['operation'],r['allowed']) for r in LOG}
    write_json(RESULTS/f'accesos_5C_{Path(sys.argv[0]).stem}_{os.getpid()}.json',
        dict(guard_active=True,test_documents_used=0,test_content_reads=sum(r['allowed'] and r['path'] in FORBIDDEN for r in LOG),
            accesses=[dict(path=p,operation=o,allowed=a) for p,o,a in sorted(unique)]))
atexit.register(log_accesses)

if __name__=='__main__':
    initialize() if not MANIFEST.exists() else preservation()
    print('Integridad previa 5C registrada; Test bloqueado.')
