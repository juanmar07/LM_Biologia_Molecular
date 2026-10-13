"""5A.1: preflight desechable y entrenamiento oficial solo con --train explícito."""
import argparse
import copy
import io
import json
import math
import random
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F
from dataset_sft_5A1 import (BASE, DATA, RESULTS, TASKS, CODE, ABSTENTIONS, DatasetSFT5A1,
    causal_dataset, load_model, prompt, encode, schedule, verify_lock, verify_preservation,
    require, write_json, sha256, log_accesses)
from evaluar_sft_5A1 import (evaluate_tokens, generate, behavior, eligible, SELECTION_POLICY,
                            report_best, save_rubric)

CONFIG=dict(max_epochs=8,batch_size=16,learning_rate=5e-5,optimizer='AdamW',betas=[.9,.95],eps=1e-8,
            weight_decay=.01,gradient_clip_norm=1.,scheduler='none',seed=42,precision='FP32',
            num_workers=0,cpu_threads=2,early_stopping_patience=2,early_stopping_min_delta=.001,
            device='cpu',task_step_distribution=dict(grounded_answerable=55,direct_instruction=5,abstention=25,causal_retention=15),
            epoch_definition='one complete pass over instruction Train; small batches balance exact task steps, no record oversampling')


def setup():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(42);random.seed(42)


def optimizer(model):
    return torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)


def batch(ds,indices):
    rows=[ds[i] for i in indices]
    return torch.stack([x for x,y in rows]),torch.stack([y for x,y in rows])


def step(model,opt,x,y):
    model.train();opt.zero_grad(set_to_none=True);logits,_=model(x);n=(y!=-100).sum().item()
    require(n>0,'Targets vacíos.');summed=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum');loss=summed/n
    require(torch.isfinite(loss).item(),'Loss no finita.');loss.backward()
    require(all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),'Gradientes/parámetros incorrectos.')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    return dict(ce_sum=summed.item(),targets=n,gradient_norm=norm.item())


def rng():
    return dict(torch=torch.get_rng_state(),python=random.getstate())


def restore(state):
    torch.set_rng_state(state['torch']);random.setstate(state['python'])


@torch.no_grad()
def log_probability(model,tok,r,answer):
    row=dict(r,answer=answer);x,y=encode(row,tok);logits,_=model(x.unsqueeze(0));targets=y!=-100
    losses=F.cross_entropy(logits[0],y,ignore_index=-100,reduction='none')
    return -losses[targets].mean().item()


def contrastive_diagnostics(model,ds,records):
    model.eval();rows=[]
    for r in records:
        wrong=ABSTENTIONS[0] if r['supported'] else r['positive_reference']
        correct_score=log_probability(model,ds.tokenizer,r,r['answer'])
        other_score=log_probability(model,ds.tokenizer,r,wrong)
        g=generate(model,ds.tokenizer,r)
        rows.append(dict(id=r['id'],pair_id=r['pair_id'],supported=r['supported'],question=r['question'],context=r['context'],reference=r['answer'],
                         correct_mean_log_probability=correct_score,opposite_mean_log_probability=other_score,
                         correct_vs_opposite_margin=correct_score-other_score,**g))
    positive=[r for r in rows if r['supported']];negative=[r for r in rows if not r['supported']]
    return dict(examples=rows,positive_loss=-sum(r['correct_mean_log_probability'] for r in positive)/len(positive),
                negative_loss=-sum(r['correct_mean_log_probability'] for r in negative)/len(negative),
                positive_greedy_exact=sum(r['generated_response'].strip()==r['reference'] for r in positive)/len(positive),
                negative_greedy_prefix=sum(r['abstention_prefix'] for r in negative)/len(negative),
                negative_greedy_exact=sum(r['abstention_exact'] for r in negative)/len(negative),
                contrastive_probability_decision_accuracy=sum(r['correct_vs_opposite_margin']>0 for r in rows)/len(rows))


def preflight(train,validation,h):
    start=time.perf_counter();initial_rng=rng();dataset_report=json.loads((DATA/'reporte_dataset_sft_5A1.json').read_text(encoding='utf-8'))
    require(dataset_report['contrastive_pair_count']['train']>=100,'Se requieren >=100 pares Train auditados.')
    require(dataset_report['validation_examples']>=120 and dataset_report['abstention_count']['validation']>=30,'Validation insuficiente.')
    require(dataset_report['balance']['max_document_percent']<=18 and dataset_report['balance']['top_two_document_percent']<=35,'Balance documental fuera de límites.')
    require(all(n>=2 for n in dataset_report['examples_by_category']['train'].values()),'Categoría Train singleton.')
    require(len(dataset_report['examples_by_document']['train'])==14,'Deben contribuir los 14 documentos Train.')
    require(not ({r['question'] for r in train.records}&{r['question'] for r in validation.records}),'Pregunta exacta compartida entre Train/Validation.')
    suite=json.loads((DATA/'suite_validation_5A1.json').read_text(encoding='utf-8'))
    require(len(suite['records'])==48 and sum(r['supported'] for r in suite['records'])==24 and len(suite['pair_ids'])==12,'Suite fija incorrecta.')
    val_ids={r['id'] for r in validation.records};require(all(r['id'] in val_ids for r in suite['records']),'Suite ajena a Validation.')
    # Eventos sintéticos: prueban el guard sin llamar open ni leer un byte de Test.
    denied=0
    for path,mode in ((BASE/'datos/dataset/test.pt','r'),(BASE/'datos/corpus_limpio.txt','r'),
                      (BASE/'modelos/checkpoint_best.pt','w'),(BASE/'README_SFT_5A.md','w'),
                      (DATA/'train_instrucciones_v2.jsonl','w')):
        try:sys.audit('open',str(path),mode,0)
        except RuntimeError:denied+=1
    require(denied==5,'Guard no impide accesos protegidos.')
    for ds in (train,validation):
        for r,(x,y) in zip(ds.records,ds.encoded):
            require(x.shape==y.shape==(128,) and x.dtype==y.dtype==torch.long,'Shapes/dtypes incorrectos.')
            first=len(ds.tokenizer.encode(prompt(r)));a=ds.tokenizer.encode(r['answer'])
            require(torch.equal(y[first:first+len(a)+1],torch.tensor(a+[3])),'Targets no equivalen a respuesta+EOS.')
            require((y[:first]==-100).all().item() and (y[first+len(a)+1:]==-100).all().item(),'Prompt/PAD contribuyen a loss.')
            require((y==3).sum().item()==1 and (y[x==0]==-100).all().item(),'EOS/PAD incorrectos.')
            require(((y==-100)|((y>=0)&(y<6000))).all().item(),'Target fuera de vocabulario.')
    planned=schedule(train);counts=Counter(t for t,i in planned);seen=[i for t,ids in planned if ids is not None for i in ids]
    require(sorted(seen)==list(range(len(train))) and planned==schedule(train),'Agenda no determinista/no cubre una época.')
    proportions={t:100*n/len(planned) for t,n in counts.items()};require(proportions==CONFIG['task_step_distribution'],'Mezcla 55/5/25/15 incorrecta.')
    pairs=[json.loads(l) for l in (DATA/'pares_contrastivos.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(l)['split']=='train']
    # Selección determinista por documentos, independiente de las predicciones.
    chosen=[];used_sources=set();by_id={r['id']:r for r in train.records}
    for p in pairs:
        if p['positive_source_doc'] in used_sources:continue
        # Ambas respuestas alternativas deben caber; nunca truncar un diagnóstico.
        try:
            encode(dict(by_id[p['positive_id']],answer=ABSTENTIONS[0]),train.tokenizer)
            encode(dict(by_id[p['negative_id']],answer=p['positive_answer']),train.tokenizer)
        except RuntimeError:continue
        chosen.append(p);used_sources.add(p['positive_source_doc'])
        if len(chosen)==8:break
    require(len(chosen)==8,'Faltan ocho pares de documentos distintos para overfit.')
    records=[by_id[p[k]] for p in chosen for k in ('positive_id','negative_id')]
    indices=[i for r in records for i,t in enumerate(train.records) if t['id']==r['id']]
    model,_=load_model('base');opt=optimizer(model);x,y=batch(train,indices)
    model.eval()
    with torch.no_grad():
        full,_=model(x);short,_=model(x[:,:40]);require(torch.allclose(full[:,:40],short,atol=2e-5,rtol=1e-5),'Padding/futuro altera logits anteriores.')
        before=F.cross_entropy(full.flatten(0,1),y.flatten(),ignore_index=-100);altered=full.clone();altered[y==-100]=torch.randn_like(altered[y==-100])*100
        require(torch.equal(before,F.cross_entropy(altered.flatten(0,1),y.flatten(),ignore_index=-100)),'Targets ignorados afectan loss.')
    initial=contrastive_diagnostics(model,train,records);final=None;passed=False;timings=[];checkpoints=[]
    print('Overfit DESECHABLE: 8 pares Train, 16 registros; base original; hasta 400 pasos.',flush=True)
    for iteration in range(1,401):
        begin=time.perf_counter();step(model,opt,x,y);timings.append(time.perf_counter()-begin)
        if iteration%50==0:
            final=contrastive_diagnostics(model,train,records)
            gains=[a['correct_mean_log_probability']-b['correct_mean_log_probability'] for a,b in zip(final['examples'],initial['examples'])]
            passed=(final['positive_loss']<initial['positive_loss']*.5 and final['negative_loss']<initial['negative_loss']*.5
                    and min(gains)>.5 and final['contrastive_probability_decision_accuracy']>=.875
                    and final['negative_greedy_prefix']>=.875)
            checkpoints.append(dict(steps=iteration,positive_loss=final['positive_loss'],negative_loss=final['negative_loss'],
                                    decision_accuracy=final['contrastive_probability_decision_accuracy'],negative_prefix=final['negative_greedy_prefix'],passed=passed))
            print(f"Overfit {iteration}: CE+={final['positive_loss']:.3f}; CE-={final['negative_loss']:.3f}; prefijo-={final['negative_greedy_prefix']:.3f}; decisión={final['contrastive_probability_decision_accuracy']:.3f}",flush=True)
            if passed:break
    # Serialización temporal solo en RAM y prueba real del siguiente paso al restaurar.
    state=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng())
    memory=io.BytesIO();torch.save(state,memory);memory.seek(0);saved=torch.load(memory,weights_only=True,map_location='cpu')
    original_step=step(model,opt,x,y);expected=copy.deepcopy(model.state_dict())
    rebuilt,_=load_model('base');rebuilt.load_state_dict(saved['model'],strict=True);new_opt=optimizer(rebuilt);new_opt.load_state_dict(saved['optimizer']);restore(saved['rng'])
    repeated_step=step(rebuilt,new_opt,x,y)
    resume_exact=original_step==repeated_step and all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in expected.items())
    require(resume_exact,'Reanudación no exacta.')
    causal=causal_dataset('train');cx,cy=batch(causal,[0,len(causal)//2,len(causal)-1]);causal_result=step(rebuilt,new_opt,cx,cy)
    del model,opt,rebuilt,new_opt,state,saved,memory,expected;restore(initial_rng)
    require(verify_lock()==h,'Datos/código cambiaron durante preflight.');verify_preservation()
    report=dict(version='5A.1',passed=passed,hashes=h,training_config=CONFIG,selection_policy=SELECTION_POLICY,
                dataset_quality=dict(train_examples=len(train),validation_examples=len(validation),train_below_target_justification=dataset_report['warnings'],
                    document_balance_passed=True,contrastive_pairs=dataset_report['contrastive_pair_count'],all_categories_at_least_two=True),
                shape_masking_EOS_PAD_vocab_passed=True,finite_gradients=True,all_parameters_trainable=True,
                guard_synthetic_denials=denied,guard_test_performed_without_open_or_Test_content=True,
                validation_suite_frozen_and_separate=True,all_14_Train_documents_present=True,
                resume_exact=resume_exact,planned_steps_per_epoch=len(planned),step_counts=dict(counts),step_distribution_percent=proportions,
                batch_sizes_by_task={t:dict(min=min(len(ids) for task,ids in planned if task==t),max=max(len(ids) for task,ids in planned if task==t)) for t in TASKS},
                overfit=dict(pair_count=8,records=16,steps=iteration,initial=initial,final=final,checkpoints=checkpoints,passed=passed,
                             criterion='CE positive/negative drop >50%; every reference mean logP gain >0.5; correct-vs-opposite decisions >=87.5%; negative greedy prefix >=87.5%',
                             temporary_weights_discarded=True,temporary_checkpoint_files=0,used_validation_for_overfit=False),
                causal_smoke=causal_result,mean_train_step_seconds=sum(timings)/len(timings),duration_seconds=time.perf_counter()-start,
                official_training_executed=False,test_documents_used=0,previous_experiment_preserved=True,
                verdict='5A.1 PREPARADO — LISTO PARA ENTRENAMIENTO OFICIAL POR EL USUARIO' if passed else '5A.1 DATASET/PREFLIGHT REQUIERE REVISIÓN')
    write_json(RESULTS/'preflight_sft_5A1.json',report)
    log_accesses('preflight');print(report['verdict'],flush=True)
    require(passed,'Overfit no aprende ambas decisiones: entrenamiento oficial bloqueado.')


def atomic_checkpoint(path,cp):
    require(path.name in ('checkpoint_sft_5A1_best.pt','checkpoint_sft_5A1_last.pt'),'Salida no autorizada.')
    temp=path.with_suffix('.pt.tmp');torch.save(cp,temp);temp.replace(path)


def official(train,validation,h,resume=False):
    check=json.loads((RESULTS/'preflight_sft_5A1.json').read_text(encoding='utf-8'))
    require(check['passed'] and check['hashes']==h and check['training_config']==CONFIG,'Preflight inválido/datos o código cambiaron.')
    require(check['selection_policy']==SELECTION_POLICY,'Política cambió desde preflight.')
    last_path=BASE/'modelos/checkpoint_sft_5A1_last.pt';best_path=BASE/'modelos/checkpoint_sft_5A1_best.pt'
    require(resume or not (last_path.exists() or best_path.exists() or (DATA/'official_training_lock.json').exists()),'Run existente: usar --resume; no sobrescribir.')
    causal_train=causal_dataset('train');causal_val=causal_dataset('validation')
    baseline_path=RESULTS/'baseline_sft_5A1.json'
    if resume:
        model,cp=load_model('last');require(cp['hashes']==h and cp['training_config']==CONFIG,'Resume incompatible.')
        opt=optimizer(model);opt.load_state_dict(cp['optimizer_state_dict']);restore(cp['rng_state'])
        history=cp['history'];start=cp['epoch']+1;global_step=cp['global_step'];best_loss=cp['best_loss'];best_eligible=cp['best_eligible'];early_best=cp['early_best'];counter=cp['early_stopping_counter']
        baseline=json.loads(baseline_path.read_text(encoding='utf-8'));require(baseline['hashes']==h,'Baseline incompatible.')
        mc=cp['model_config']
    else:
        model,base_cp=load_model('base');mc=base_cp['model_config']
        baseline=dict(version='5A.1',hashes=h,tokens=evaluate_tokens(model,validation),retention=evaluate_tokens(model,causal_val),behavior=behavior(model,validation))
        require(abs(baseline['retention']['causal_retention']['loss']-4.725705095024419)<1e-5,'Baseline LM no reproduce época 19.')
        write_json(baseline_path,baseline);opt=optimizer(model)
        history=[];start=1;global_step=0;best_loss=math.inf;best_eligible=False;early_best=math.inf;counter=0
        write_json(DATA/'official_training_lock.json',dict(version='5A.1',hashes=h,training_config=CONFIG,selection_policy=SELECTION_POLICY))
    if counter<2:
        for epoch in range(start,9):
            begin=time.perf_counter();planned=schedule(train,epoch);counts=Counter();sums={}
            causal_indices=list(range(len(causal_train)));random.Random(10042+epoch).shuffle(causal_indices);cursor=0
            for task,indices in planned:
                if indices is None:indices=causal_indices[cursor:cursor+16];cursor+=16;x,y=batch(causal_train,indices)
                else:x,y=batch(train,indices)
                s=step(model,opt,x,y);b=sums.setdefault(task,dict(ce_sum=0.,targets=0));b['ce_sum']+=s['ce_sum'];b['targets']+=s['targets'];counts[task]+=1;global_step+=1
            tokens=evaluate_tokens(model,validation);retention=evaluate_tokens(model,causal_val);generated=behavior(model,validation)
            is_eligible,constraints=eligible(tokens,retention,generated,baseline);loss=tokens['grounded_answerable']['loss']
            improve=(is_eligible and not best_eligible) or (is_eligible==best_eligible and loss<best_loss)
            if improve:best_loss=loss;best_eligible=is_eligible
            if loss<early_best-.001:early_best=loss;counter=0
            else:counter+=1
            row=dict(epoch=epoch,global_step=global_step,step_counts=dict(counts),
                     step_distribution_percent={t:100*n/len(planned) for t,n in counts.items()},
                     train_by_task={t:b|dict(loss=b['ce_sum']/b['targets']) for t,b in sums.items()},validation_tokens=tokens,
                     grounded_validation_loss=loss,grounded_validation_perplexity=tokens['grounded_answerable']['perplexity'],
                     abstention_validation_loss=tokens['abstention']['loss'],abstention_exact_rate=generated['metrics']['abstention_exact_rate'],
                     abstention_prefix_rate=generated['metrics']['abstention_prefix_rate'],EOS_accuracy_reference_end=tokens['EOS']['EOS_accuracy_reference_end'],
                     causal_validation_loss=retention['causal_retention']['loss'],causal_validation_perplexity=retention['causal_retention']['perplexity'],
                     behavior_metrics=generated['metrics'],selection_constraints=constraints,eligible=is_eligible,selected_best=improve,
                     early_stopping_counter=counter,duration_seconds=time.perf_counter()-begin)
            history.append(row)
            cp=dict(version='5A.1',epoch=epoch,global_step=global_step,model_config=mc,model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),
                    base_checkpoint_sha256=h['base_checkpoint'],tokenizer_sha256=h['tokenizer'],hashes=h,training_config=CONFIG,
                    selection_policy=SELECTION_POLICY,selection_status='ELIGIBLE' if is_eligible else 'PROVISIONAL_REVIEW_REQUIRED',
                    history=copy.deepcopy(history),rng_state=rng(),best_loss=best_loss,best_eligible=best_eligible,early_best=early_best,
                    early_stopping_counter=counter,pytorch_version=str(torch.__version__),test_documents_used=0)
            require(verify_lock()==h,'Entradas cambiaron.');atomic_checkpoint(last_path,cp)
            if improve:atomic_checkpoint(best_path,cp)
            write_json(RESULTS/'historial_sft_5A1.json',history)
            print(f"Época {epoch}/8: grounded CE={loss:.5f}; abstención={generated['metrics']['abstention_prefix_rate']:.3f}; elegible={is_eligible}; best={improve}",flush=True)
            if counter>=2:break
    require(best_path.exists(),'No se completó una época; no hay best para reportar.')
    best_model,best_cp=load_model('best');best_cp['history']=history
    tokens=evaluate_tokens(best_model,validation);retention=evaluate_tokens(best_model,causal_val);generated=behavior(best_model,validation)
    report=report_best(best_cp,baseline,tokens,retention,generated);write_json(RESULTS/'reporte_entrenamiento_sft_5A1.json',report)
    save_rubric(generated,sha256(best_path));verify_preservation();log_accesses('oficial');print(report['verdict'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument('--smoke-test',action='store_true');m.add_argument('--train',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    require(not args.resume or args.train,'--resume requiere --train.');setup();h=verify_lock()
    train=DatasetSFT5A1('train');validation=DatasetSFT5A1('validation')
    if args.smoke_test:preflight(train,validation,h)
    else:
        print('Entrenamiento OFICIAL 5A.1 solicitado explícitamente: hasta ocho épocas; inicio exclusivo desde base época 19.',flush=True)
        official(train,validation,h,args.resume)


if __name__=='__main__':main()
