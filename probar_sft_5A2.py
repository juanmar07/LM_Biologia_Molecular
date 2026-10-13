"""Probar 5A.2: suite fija, ejemplo por ID o contexto explícito sin recuperación."""
import argparse
import json
import torch
from dataset_sft_5A2 import BASE,DATA,RESULTS,DatasetSFT5A2,load_model,verify_lock,verify_preservation,require,write_json,sha256,log_accesses
from evaluar_sft_5A2 import behavior,generate,output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',choices=('base','best','last'),default='best')
    g=p.add_mutually_exclusive_group();g.add_argument('--suite',action='store_true');g.add_argument('--example-id');g.add_argument('--context')
    p.add_argument('--question');p.add_argument('--no-save',action='store_true');a=p.parse_args()
    require(bool(a.context)==bool(a.question),'--context requiere --question; la suite/ID usa su propia pregunta.')
    torch.set_num_threads(2);h=verify_lock();name='checkpoint_best.pt' if a.checkpoint=='base' else f'checkpoint_sft_5A2_{a.checkpoint}.pt'
    require((BASE/'modelos'/name).exists(),'Checkpoint 5A.2 todavía no existe. Ejecute el entrenamiento oficial o seleccione --checkpoint base.')
    model,cp=load_model(a.checkpoint)
    if a.checkpoint!='base':require(cp['hashes']==h,'Checkpoint de otro dataset/código.')
    ds=DatasetSFT5A2('validation')
    if a.context:
        r=dict(task_type='extractive_grounded',context=a.context,question=a.question)
        out=dict(mode='user_supplied_context',question=a.question,context=a.context,**generate(model,ds.tokenizer,r),
                 source_provenance='user supplied; not part of Validation/selection; scientific fidelity requires source review')
    elif a.example_id:
        r=next((r for r in ds.records if r['id']==a.example_id),None)
        require(r is not None,'ID no pertenece a Validation 5A.2.');out=dict(mode='frozen_validation_example',example=output(model,ds,r))
    else:out=dict(mode='frozen_validation_suite',**behavior(model,ds))
    result=dict(version='5A.2',checkpoint=name,checkpoint_sha256=sha256(BASE/'modelos'/name),hashes=h,
        training_executed=False,test_documents_used=0,**out)
    if not a.no_save:
        suffix='custom' if a.context else 'example' if a.example_id else 'suite'
        path=RESULTS/f'prueba_sft_5A2_{a.checkpoint}_{suffix}.json';write_json(path,result)
        print('Reporte:',str(path))
    if 'metrics' in out:print(json.dumps(out['metrics'],ensure_ascii=False,indent=2))
    elif 'example' in out:print(json.dumps(out['example'],ensure_ascii=False,indent=2))
    else:print(out['generated_response'])
    verify_preservation();require(verify_lock()==h,'Entradas cambiaron durante prueba.');log_accesses('prueba')


if __name__=='__main__':main()
