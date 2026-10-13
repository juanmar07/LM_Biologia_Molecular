"""5A.2: preparación desechable (--smoke-test) u oficial explícito (--train)."""
import argparse
import copy
import io
import json
import math
import random
import sys
import time
from collections import Counter

from dataset_sft_5A2 import (torch,BASE,DATA,RESULTS,TASKS,ABSTENTIONS,DatasetSFT5A2,causal_dataset,
    load_model,prompt,encode,schedule,verify_lock,verify_preservation,require,write_json,sha256,log_accesses)
import torch.nn.functional as F
from evaluar_sft_5A2 import evaluate_tokens,grounded,behavior,output,summarize,eligible,SELECTION_POLICY,report_best,save_rubric

CONFIG=dict(max_epochs=8,batch_size=16,learning_rate=5e-5,optimizer='AdamW',betas=[.9,.95],eps=1e-8,
    weight_decay=.01,gradient_clip_norm=1.,scheduler='none',seed=42,precision='FP32',num_workers=0,cpu_threads=2,
    early_stopping_patience=2,early_stopping_min_delta=.001,device='cpu',
    task_step_distribution=dict(extractive_grounded=75,paraphrastic_grounded=10,abstention=5,causal_retention=10),
    epoch_definition='one pass over all instruction Train records; exact task-step proportions; no record oversampling')
OVERFIT_POLICY=dict(max_steps=400,check_every=50,all_three_task_CE_ratio_max=.5,all_reference_mean_logP_gain_min=.5,
    extractive_exact_min=1.,paraphrastic_exact_min=1.,abstention_exact_min=1.,false_abstention_max=0.,EOS_min=1.)


def setup():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(42);random.seed(42)


def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)


def batch(ds,indices):
    rows=[ds[i] for i in indices];return torch.stack([x for x,y in rows]),torch.stack([y for x,y in rows])


def rng():return dict(torch=torch.get_rng_state(),python=random.getstate())
def restore(s):torch.set_rng_state(s['torch']);random.setstate(s['python'])


def step(m,opt,x,y):
    m.train();opt.zero_grad(set_to_none=True);logits,_=m(x);n=(y!=-100).sum().item();require(n>0,'Targets vacíos.')
    total=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum');loss=total/n
    require(torch.isfinite(loss).item(),'Loss no finita.');loss.backward()
    require(all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all().item() for p in m.parameters()),'Gradientes/entrenabilidad incorrectos.')
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
    return dict(ce_sum=total.item(),targets=n,gradient_norm=norm.item())


@torch.no_grad()
def diagnostics(m,ds,records):
    m.eval();rows=[];sums={}
    for r in records:
        x,y=encode(r,ds.tokenizer);logits,_=m(x.unsqueeze(0));ce=F.cross_entropy(logits[0],y,ignore_index=-100,reduction='none')
        n=(y!=-100).sum().item();total=ce.double().sum().item();rows.append(output(m,ds,r)|dict(reference_mean_log_probability=-total/n))
        b=sums.setdefault(r['task_type'],dict(ce_sum=0.,targets=0));b['ce_sum']+=total;b['targets']+=n
    return dict(examples=rows,metrics=summarize(rows),task_CE={t:b['ce_sum']/b['targets'] for t,b in sums.items()})


def select_overfit(ds):
    by_id={r['id']:r for r in ds.records};neg=[r for r in ds.records if not r['supported']][:2]
    require(len(neg)==2,'Faltan dos pares contrastivos Train.')
    chosen=[by_id[r['positive_id']] for r in neg];used={r['id'] for r in chosen};sources={r['source_doc'] for r in chosen}
    ext=[r for r in ds.records if r['task_type']=='extractive_grounded' and r['id'] not in used]
    # Diversidad de fuentes, no selección según comportamiento del modelo.
    ext.sort(key=lambda r:(r['source_doc'] in sources,r['source_doc'],r['id']))
    for r in ext:
        if len(chosen)==10:break
        if r['source_doc'] in sources:continue
        chosen.append(r);used.add(r['id']);sources.add(r['source_doc'])
    for r in ext:
        if len(chosen)==10:break
        if r['id'] not in used:chosen.append(r);used.add(r['id'])
    para=[r for r in ds.records if r['task_type']=='paraphrastic_grounded'];seen=set();selected_para=[]
    for r in para:
        if r['source_doc'] in seen:continue
        seen.add(r['source_doc']);selected_para.append(r)
        if len(selected_para)==4:break
    records=chosen+selected_para+neg
    require(len(records)==16 and len({r['id'] for r in records})==16,'Overfit requiere 10 extractivos/4 paráfrasis/2 negativos distintos.')
    return records


def preflight(train,val,h):
    start=time.perf_counter();initial_rng=rng();d=json.loads((DATA/'reporte_dataset_sft_5A2.json').read_text(encoding='utf-8'))
    require(len(d['examples_by_document']['train'])==14 and len(d['examples_by_document']['validation'])==4,'Fuentes incompletas.')
    require(d['balance']['max_document_percent']<=18 and d['balance']['top_two_document_percent']<=35,'Concentración documental excesiva.')
    require(not ({r['question'] for r in train.records}&{r['question'] for r in val.records}),'Preguntas compartidas entre splits.')
    suite=json.loads((DATA/'suite_validation_5A2.json').read_text(encoding='utf-8'));require(Counter(r['task_type'] for r in suite['records'])==dict(zip(TASKS,(40,8,8))),'Suite incorrecta.')
    require(all(r in val.records for r in suite['records']),'Suite fuera de Validation.')
    for ds in (train,val):
        for r,(x,y) in zip(ds.records,ds.encoded):
            first=len(ds.tokenizer.encode(prompt(r)));a=ds.tokenizer.encode(r['answer'])
            require(x.shape==y.shape==(128,) and x.dtype==y.dtype==torch.long,'Shapes/dtypes.')
            require(torch.equal(y[first:first+len(a)+1],torch.tensor(a+[3])),'Primer/último target incorrectos.')
            require((y[:first]==-100).all().item() and (y[first+len(a)+1:]==-100).all().item(),'Prompt/PAD supervisados.')
            require((y==3).sum().item()==1 and (y[x==0]==-100).all().item(),'EOS/PAD incorrectos.')
            require(((y==-100)|((y>=4)&(y<6000))|(y==3)).all().item(),'Target fuera de vocabulario.')
    planned=schedule(train);counts=Counter(t for t,ids in planned);seen=[i for t,ids in planned if ids is not None for i in ids]
    proportions={t:n*100/len(planned) for t,n in counts.items()}
    require(sorted(seen)==list(range(len(train))) and planned==schedule(train) and proportions==CONFIG['task_step_distribution'],'Agenda/mix no exacta.')
    split_meta=json.loads((BASE/'datos/splits.json').read_text(encoding='utf-8'));test_sources=set(split_meta['splits']['test']['sources'])
    prohibited=[(BASE/'datos/dataset/test.pt','r'),(BASE/'datos/splits/test.txt','r'),(BASE/'datos/corpus_limpio.txt','r')]
    prohibited.extend((BASE/d['clean_file'],'r') for d in split_meta['inventory'] if d['source'] in test_sources)
    prohibited.extend((BASE/'corpus'/source,'r') for source in test_sources)
    protected=[(BASE/'modelos/checkpoint_best.pt','w'),(BASE/'modelos/checkpoint_sft_5A1_best.pt','w'),(DATA/'train_instrucciones_5A2.jsonl','w')]
    denied=0
    for path,mode in prohibited+protected:
        try:sys.audit('open',str(path),mode,0)
        except RuntimeError:denied+=1
    require(denied==len(prohibited+protected),'Guard insuficiente.')
    # Métricas léxicas no pueden acreditar una salida semánticamente errónea.
    from evaluar_sft_5A2 import lexical_diagnostics
    fixture=lexical_diagnostics('El ADN no contiene genes.','El ADN contiene genes.','El ADN contiene genes.')
    require(not fixture['reference_exact'] and not fixture['extractive_span_match'],'El evaluador ignora una negación contradictoria.')
    m,base_cp=load_model('base');opt=optimizer(m);records=select_overfit(train);idx={r['id']:i for i,r in enumerate(train.records)}
    x,y=batch(train,[idx[r['id']] for r in records]);m.eval()
    with torch.no_grad():
        full,_=m(x);short,_=m(x[:,:40]);require(torch.allclose(full[:,:40],short,atol=2e-5,rtol=1e-5),'Fuga causal/padding.')
        loss=F.cross_entropy(full.flatten(0,1),y.flatten(),ignore_index=-100);altered=full.clone();altered[y==-100]=100
        require(torch.equal(loss,F.cross_entropy(altered.flatten(0,1),y.flatten(),ignore_index=-100)),'Tokens ignorados afectan loss.')
    initial=diagnostics(m,train,records);progress=[];timings=[];passed=False
    print('Overfit DESECHABLE 5A.2: 10 extractivos +4 paráfrasis +2 abstenciones, hasta 400 pasos.',flush=True)
    for it in range(1,401):
        begin=time.perf_counter();step(m,opt,x,y);timings.append(time.perf_counter()-begin)
        if it%50==0:
            final=diagnostics(m,train,records);met=final['metrics'];gains=[a['reference_mean_log_probability']-b['reference_mean_log_probability'] for a,b in zip(final['examples'],initial['examples'])]
            passed=(all(final['task_CE'][t]<initial['task_CE'][t]*.5 for t in TASKS) and min(gains)>.5
                and met['extractive_exact_rate']==1 and met['paraphrastic_reference_exact_rate']==1
                and met['abstention_exact_rate']==1 and met['false_abstention_rate']==0
                and all(r['EOS_generated'] for r in final['examples']))
            progress.append(dict(steps=it,task_CE=final['task_CE'],metrics=met,passed=passed))
            print(f"Overfit {it}: extractivo={met['extractive_exact_rate']:.2f}, paráfrasis={met['paraphrastic_reference_exact_rate']:.2f}, abstención={met['abstention_exact_rate']:.2f}, false abstention={met['false_abstention_rate']:.2f}",flush=True)
            if passed:break
    state=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng())
    mem=io.BytesIO();torch.save(state,mem);mem.seek(0);saved=torch.load(mem,weights_only=True,map_location='cpu')
    actual=step(m,opt,x,y);expected=copy.deepcopy(m.state_dict());rebuilt,_=load_model('base');rebuilt.load_state_dict(saved['model'],strict=True)
    new_opt=optimizer(rebuilt);new_opt.load_state_dict(saved['optimizer']);restore(saved['rng']);repeat=step(rebuilt,new_opt,x,y)
    require(actual==repeat and all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in expected.items()),'Resume no exacto.')
    cx,cy=batch(causal_dataset('train'),[0,100,1000]);causal_smoke=step(rebuilt,new_opt,cx,cy)
    del m,opt,rebuilt,new_opt,state,saved,mem,expected;restore(initial_rng)
    # Evaluación read-only para comprobar todo el evaluador y reproducir la base.
    base,_=load_model('base');tokens=evaluate_tokens(base,val);ret=evaluate_tokens(base,causal_dataset('validation'));g=behavior(base,val)
    require(abs(ret['causal_retention']['loss']-base_cp['best_validation_loss'])<1e-5,'CE causal de época 19 no reproducida.')
    write_json(RESULTS/'verificacion_evaluador_sft_5A2.json',dict(version='5A.2',hashes=h,base_epoch=19,tokens=tokens,retention=ret,behavior=g,test_documents_used=0,training_executed=False))
    require(verify_lock()==h,'Entradas cambiaron durante preflight.');old=verify_preservation()
    report=dict(version='5A.2',passed=passed,hashes=h,training_config=CONFIG,selection_policy=SELECTION_POLICY,overfit_policy=OVERFIT_POLICY,
        dataset=dict(train_examples=len(train),validation_examples=len(val),by_task=d['counts_by_task_type']),
        masking_EOS_PAD_shapes_vocab_passed=True,finite_gradients=True,all_parameters_trainable=True,resume_exact=True,
        ignored_target_loss_invariance=True,causal_attention_padding_invariance=True,guard_synthetic_denials=denied,
        guard_test_used_no_open_or_Test_content=True,lexical_negation_regression_checked=True,
        forbidden_Test_sources_from_metadata=len(test_sources),forbidden_Test_paths_synthetically_checked=len(prohibited),
        planned_steps_per_epoch=len(planned),step_counts=dict(counts),step_distribution_percent=proportions,
        batch_sizes={t:dict(min=min(len(ids) for task,ids in planned if task==t),max=max(len(ids) for task,ids in planned if task==t)) for t in TASKS},
        overfit=dict(records=16,task_counts=dict(Counter(r['task_type'] for r in records)),contrastive_pairs=2,source_count=len({r['source_doc'] for r in records}),
            steps=it,initial=initial,final=final,progress=progress,passed=passed,temporary_weights_discarded=True,checkpoint_files_created=0,used_validation_for_updates=False),
        causal_smoke=causal_smoke,base_causal_loss=ret['causal_retention']['loss'],protected_files_verified=len(old),
        mean_train_step_seconds=mean(timings),duration_seconds=time.perf_counter()-start,
        test_documents_used=0,official_training_executed=False,previous_experiments_preserved=True,
        verdict='5A.2 PREPARADO — LISTO PARA ENTRENAMIENTO OFICIAL' if passed else '5A.2 REQUIERE REVISIÓN ANTES DE ENTRENAR')
    write_json(RESULTS/'preflight_sft_5A2.json',report);log_accesses('preflight');print(report['verdict'],flush=True)
    require(passed,'El overfit no memoriza fielmente los tres tipos: no iniciar entrenamiento oficial.')


def mean(xs):return sum(xs)/len(xs)


def atomic_checkpoint(path,cp):
    require(path.name in ('checkpoint_sft_5A2_best.pt','checkpoint_sft_5A2_last.pt'),'Salida no autorizada.')
    temp=path.with_suffix('.pt.tmp');torch.save(cp,temp);temp.replace(path)


def official(train,val,h,resume):
    p=json.loads((RESULTS/'preflight_sft_5A2.json').read_text(encoding='utf-8'))
    require(p['passed'] and p['hashes']==h and p['training_config']==CONFIG and p['selection_policy']==SELECTION_POLICY and p['overfit_policy']==OVERFIT_POLICY,'Preflight inválido.')
    last=BASE/'modelos/checkpoint_sft_5A2_last.pt';best=BASE/'modelos/checkpoint_sft_5A2_best.pt'
    require(resume or not any(x.exists() for x in (last,best,DATA/'official_training_lock.json',RESULTS/'historial_sft_5A2.json')),'Run existente: --resume; no sobrescribir.')
    causal_train=causal_dataset('train');causal_val=causal_dataset('validation')
    if resume:
        m,cp=load_model('last');require(cp['hashes']==h and cp['training_config']==CONFIG and cp['selection_policy']==SELECTION_POLICY,'Resume incompatible.')
        opt=optimizer(m);opt.load_state_dict(cp['optimizer_state_dict']);restore(cp['rng_state']);mc=cp['model_config']
        history=cp['history'];start=cp['epoch']+1;global_step=cp['global_step'];best_score=cp['best_score'];best_loss=cp['best_loss'];best_eligible=cp['best_eligible'];early_best=cp['early_best'];counter=cp['early_stopping_counter']
        baseline=json.loads((RESULTS/'baseline_sft_5A2.json').read_text(encoding='utf-8'));require(baseline['hashes']==h,'Baseline incompatible.')
    else:
        m,cp=load_model('base');mc=cp['model_config'];baseline=dict(version='5A.2',hashes=h,tokens=evaluate_tokens(m,val),retention=evaluate_tokens(m,causal_val),behavior=behavior(m,val))
        require(abs(baseline['retention']['causal_retention']['loss']-cp['best_validation_loss'])<1e-5,'Baseline no reproducida.')
        write_json(RESULTS/'baseline_sft_5A2.json',baseline);opt=optimizer(m);history=[];start=1;global_step=0;best_score=-1.;best_loss=math.inf;best_eligible=False;early_best=math.inf;counter=0
        write_json(DATA/'official_training_lock.json',dict(version='5A.2',hashes=h,training_config=CONFIG,selection_policy=SELECTION_POLICY))
    if counter<2:
        for epoch in range(start,9):
            begin=time.perf_counter();planned=schedule(train,epoch);counts=Counter();sums={};ids=list(range(len(causal_train)));random.Random(10042+epoch).shuffle(ids);cursor=0
            for task,indices in planned:
                if indices is None:indices=ids[cursor:cursor+16];cursor+=16;x,y=batch(causal_train,indices)
                else:x,y=batch(train,indices)
                s=step(m,opt,x,y);b=sums.setdefault(task,dict(ce_sum=0.,targets=0));b['ce_sum']+=s['ce_sum'];b['targets']+=s['targets'];counts[task]+=1;global_step+=1
            tokens=evaluate_tokens(m,val);ret=evaluate_tokens(m,causal_val);g=behavior(m,val);loss=grounded(tokens)['loss'];score=g['metrics']['fidelity_reference_exact_rate']
            ok,constraints=eligible(tokens,ret,g,baseline)
            improve=(ok and not best_eligible) or (ok==best_eligible and (score>best_score or score==best_score and loss<best_loss))
            if improve:best_score=score;best_loss=loss;best_eligible=ok
            if loss<early_best-.001:early_best=loss;counter=0
            else:counter+=1
            row=dict(epoch=epoch,global_step=global_step,train_by_task={t:b|dict(loss=b['ce_sum']/b['targets']) for t,b in sums.items()},
                step_counts=dict(counts),step_distribution_percent={t:100*n/len(planned) for t,n in counts.items()},validation_tokens=tokens,grounded_validation=grounded(tokens),
                extractive_validation_loss=tokens['extractive_grounded']['loss'],paraphrastic_validation_loss=tokens['paraphrastic_grounded']['loss'],
                abstention_validation_loss=tokens['abstention']['loss'],EOS_accuracy_reference_end=tokens['EOS']['EOS_accuracy_reference_end'],
                causal_validation_loss=ret['causal_retention']['loss'],causal_validation_perplexity=ret['causal_retention']['perplexity'],
                behavior_metrics=g['metrics'],false_abstention_rate=g['metrics']['false_abstention_rate'],fidelity_reference_exact_rate=score,
                selection_constraints=constraints,eligible=ok,selected_best=improve,early_stopping_counter=counter,duration_seconds=time.perf_counter()-begin)
            history.append(row);cp=dict(version='5A.2',epoch=epoch,global_step=global_step,model_config=mc,model_state_dict=m.state_dict(),optimizer_state_dict=opt.state_dict(),
                base_checkpoint_sha256=h['base_checkpoint'],tokenizer_sha256=h['tokenizer'],hashes=h,training_config=CONFIG,selection_policy=SELECTION_POLICY,
                selection_status='ELIGIBLE' if ok else 'PROVISIONAL_REVIEW_REQUIRED',history=copy.deepcopy(history),rng_state=rng(),best_score=best_score,best_loss=best_loss,
                best_eligible=best_eligible,early_best=early_best,early_stopping_counter=counter,pytorch_version=str(torch.__version__),test_documents_used=0)
            require(verify_lock()==h,'Entradas cambiaron.');atomic_checkpoint(last,cp)
            if improve:atomic_checkpoint(best,cp)
            write_json(RESULTS/'historial_sft_5A2.json',history);print(f"Época {epoch}/8: fidelidad exacta={score:.3f}, false abstention={g['metrics']['false_abstention_rate']:.3f}, grounded CE={loss:.4f}, elegible={ok}",flush=True)
            if counter>=2:break
    require(best.exists(),'No hay época completada.');m,cp=load_model('best');cp['history']=history
    tokens=evaluate_tokens(m,val);ret=evaluate_tokens(m,causal_val);g=behavior(m,val);report=report_best(cp,baseline,tokens,ret,g)
    write_json(RESULTS/'reporte_entrenamiento_sft_5A2.json',report);save_rubric(g,sha256(best));verify_preservation();log_accesses('oficial');print(report['verdict'])


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--smoke-test',action='store_true');g.add_argument('--train',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    require(not a.resume or a.train,'--resume requiere --train.');setup();h=verify_lock();train=DatasetSFT5A2('train');val=DatasetSFT5A2('validation')
    if a.smoke_test:preflight(train,val,h)
    else:official(train,val,h,a.resume)


if __name__=='__main__':main()
