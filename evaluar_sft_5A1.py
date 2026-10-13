"""Evaluación 5A.1 por tareas, evidencia contrastiva y rúbrica trazable."""
import argparse
import json
import math
import re
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_sft_5A1 import (BASE, DATA, RESULTS, ABSTENTIONS, DatasetSFT5A1, causal_dataset,
    load_model, prompt, require, write_json, verify_lock, verify_preservation, sha256, log_accesses)

LABELS=('SUPPORTED_COMPLETE','SUPPORTED_PARTIAL','UNSUPPORTED_OR_WRONG','ABSTAIN_CORRECT','ABSTAIN_INCORRECT')


def normalized(s):
    return ' '.join(s.split())


def is_abstention(s):
    s=s.strip()
    return s.startswith('No tengo información suficiente') or s.startswith('El contexto proporcionado no contiene información suficiente')


@torch.no_grad()
def evaluate_tokens(model,ds,batch_size=16):
    model.eval();sums={};eos1=eos5=eosn=0
    offset=0
    for x,y in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
        logits,_=model(x)
        losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='none').reshape_as(y)
        require(torch.isfinite(losses).all().item(),'Loss no finita.')
        for j in range(len(x)):
            task=ds.records[offset+j]['task_type'] if isinstance(ds,DatasetSFT5A1) else 'causal_retention'
            b=sums.setdefault(task,dict(ce_sum=0.,targets=0))
            b['ce_sum']+=losses[j].double().sum().item();b['targets']+=(y[j]!=-100).sum().item()
        offset+=len(x)
        if isinstance(ds,DatasetSFT5A1):
            e=logits[y==3];eosn+=len(e);eos1+=(e.argmax(-1)==3).sum().item();eos5+=(e.topk(5,-1).indices==3).any(-1).sum().item()
    result={t:b|dict(loss=b['ce_sum']/b['targets'],perplexity=math.exp(min(700,b['ce_sum']/b['targets']))) for t,b in sums.items()}
    if eosn:result['EOS']=dict(EOS_accuracy_reference_end=eos1/eosn,EOS_top5_reference_end=eos5/eosn,targets=eosn)
    return result


@torch.no_grad()
def generate(model,tok,r,max_new_tokens=48):
    model.eval();prefix=[2]+tok.encode(prompt(r));ids=[];eos=False
    require(len(prefix)<128,'Prompt excede contexto.')
    budget=min(max_new_tokens,129-len(prefix))
    for _ in range(budget):
        logits,_=model(torch.tensor([prefix+ids],dtype=torch.long))
        scores=logits[0,-1].clone();scores[:3]=-torch.inf;t=scores.argmax().item()
        if t==3:eos=True;break
        ids.append(t)
    text=tok.decode(ids);s=text.strip();w=re.findall(r'\w+',s.lower());tri=[tuple(w[i:i+3]) for i in range(max(0,len(w)-2))]
    return dict(generated_response=text,generated_token_ids=ids,EOS_generated=eos,answer_tokens=len(ids),
                natural_closure=bool(re.search(r'[.!?][”"\)]?$',s)),abstention_exact=s in ABSTENTIONS,
                abstention_prefix=is_abstention(s),repetition=any(n>=3 for n in Counter(tri).values()),
                generation_budget=budget,stop_reason='EOS' if eos else 'token_budget')


def behavior(model,ds):
    suite=json.loads((DATA/'suite_validation_5A1.json').read_text(encoding='utf-8'))
    rows=[];cache={}
    for r in suite['records']:
        g=generate(model,ds.tokenizer,r,suite['generation_config']['max_new_tokens']);cache[r['id']]=g
        exact=normalized(g['generated_response'])==normalized(r['answer'])
        label='SUPPORTED_COMPLETE' if r['supported'] and exact else 'ABSTAIN_CORRECT' if not r['supported'] and g['abstention_exact'] else 'ABSTAIN_INCORRECT' if not r['supported'] and not g['abstention_prefix'] else None
        rows.append(dict(id=r['id'],pair_id=r['pair_id'],task_type=r['task_type'],source_doc=r['source_doc'],source_split=r['source_split'],
                         context=r['context'],evidence_text=r['evidence_text'],question=r['question'],reference=r['answer'],
                         **g,manual_label=label,manual_natural_closure=g['natural_closure'] if exact or g['abstention_exact'] else None,
                         label_origin='exact_reference_or_abstention_check' if label else 'requires_manual_source_review'))
    negative_rows=[]
    for r in ds.records:
        if r['task_type']=='abstention':
            g=cache.get(r['id']) or generate(model,ds.tokenizer,r,suite['generation_config']['max_new_tokens'])
            negative_rows.append(dict(id=r['id'],question=r['question'],context=r['context'],reference=r['answer'],**g))
    positives=[r for r in rows if r['task_type']!='abstention'];negatives=[r for r in rows if r['task_type']=='abstention']
    by_id={r['id']:r for r in rows};decisions=[]
    for pair_id in suite['pair_ids']:
        pair=[r for r in rows if r['pair_id']==pair_id]
        p=next(r for r in pair if r['task_type']!='abstention');n=next(r for r in pair if r['task_type']=='abstention')
        decisions.append(dict(pair_id=pair_id,positive_id=p['id'],negative_id=n['id'],
                              correct=p['manual_label']=='SUPPORTED_COMPLETE' and n['manual_label']=='ABSTAIN_CORRECT'))
    rate=lambda items,k:sum(r[k] for r in items)/len(items)
    metrics=dict(answerable_count=len(positives),unanswerable_count=len(negatives),contrastive_pairs=len(decisions),
                 supported_answer_rate=sum(r['manual_label']=='SUPPORTED_COMPLETE' for r in positives)/len(positives),
                 supported_answer_rate_method='conservative_exact_reference_lower_bound_pending_manual_review',
                 abstention_exact_rate=rate(negative_rows,'abstention_exact'),abstention_prefix_rate=rate(negative_rows,'abstention_prefix'),
                 suite_abstention_prefix_rate=rate(negatives,'abstention_prefix'),natural_closure_rate=rate(rows,'natural_closure'),
                 EOS_generation_rate=rate(rows,'EOS_generated'),contrastive_decision_accuracy=sum(d['correct'] for d in decisions)/len(decisions),
                 false_abstention_rate=rate(positives,'abstention_prefix'),manual_review_pending=sum(r['manual_label'] is None for r in rows),
                 closure_review_pending=sum(r['manual_natural_closure'] is None for r in rows),
                 natural_closure_method='punctuation_proxy_pending_review_except_exact_complete_references')
    return dict(metrics=metrics,suite_examples=rows,all_validation_negatives=negative_rows,contrastive_decisions=decisions,
                generation_config=suite['generation_config'],suite_sha256=sha256(DATA/'suite_validation_5A1.json'))


SELECTION_POLICY=dict(primary='grounded_validation_loss',abstention_prefix_min=.8,EOS_generation_min=.8,
    maximum_baseline_rate_drop=.05,causal_degradation_max_exclusive_percent=10.,
    fallback='If no epoch eligible, save lowest grounded CE as PROVISIONAL_REVIEW_REQUIRED; never declare success.',
    tie_break='earliest epoch',early_stopping='grounded_validation_loss min_delta=.001 patience=2; independent of fallback status')


def eligible(tokens,retention,generated,baseline):
    b=baseline['behavior']['metrics'];m=generated['metrics'];delta=100*(retention['causal_retention']['loss']/baseline['retention']['causal_retention']['loss']-1)
    checks=dict(abstention=m['abstention_prefix_rate']>=max(.8,b['abstention_prefix_rate']-.05),
                EOS_generation=m['EOS_generation_rate']>=max(.8,b['EOS_generation_rate']-.05),
                EOS_reference=tokens['EOS']['EOS_accuracy_reference_end']>=max(0,baseline['tokens']['EOS']['EOS_accuracy_reference_end']-.05),
                retention=delta<10)
    return all(checks.values()),dict(checks=checks,causal_degradation_percent=delta)


def report_best(cp,baseline,tokens,retention,generated):
    m=generated['metrics'];delta=100*(retention['causal_retention']['loss']/baseline['retention']['causal_retention']['loss']-1)
    checks=dict(supported_answer_rate=m['supported_answer_rate']>=.8,abstention_prefix_rate=m['abstention_prefix_rate']>=.8,
                natural_closure_rate=m['natural_closure_rate']>=.9,EOS_generation_rate=m['EOS_generation_rate']>=.8,
                contrastive_decision_accuracy=m['contrastive_decision_accuracy']>=.8,causal_degradation=delta<10,
                source_review_complete=m['manual_review_pending']==0 and m['closure_review_pending']==0,checkpoint_eligible=cp['selection_status']=='ELIGIBLE')
    warnings=[]
    if m['manual_review_pending']:warnings.append('Supported answer rate es un límite inferior por coincidencia exacta; revisar paráfrasis contra evidencia antes de aprobar.')
    if delta>=10:warnings.append('WARNING: degradación causal >=10%.')
    if delta>15:warnings.append('Posible catastrophic forgetting >15%.')
    if cp['selection_status']!='ELIGIBLE':warnings.append('Ningún checkpoint satisface los filtros: best es provisional y requiere revisión.')
    return dict(version='5A.1',base_checkpoint='modelos/checkpoint_best.pt',base_checkpoint_sha256=cp['base_checkpoint_sha256'],
                best_epoch=cp['epoch'],best_checkpoint='modelos/checkpoint_sft_5A1_best.pt',selection_status=cp['selection_status'],
                selection_policy=SELECTION_POLICY,training_config=cp['training_config'],history=cp['history'],hashes=cp['hashes'],
                grounded_validation_loss=tokens['grounded_answerable']['loss'],grounded_validation_perplexity=tokens['grounded_answerable']['perplexity'],
                abstention_validation_loss=tokens['abstention']['loss'],EOS_accuracy_reference_end=tokens['EOS']['EOS_accuracy_reference_end'],
                causal_validation_loss=retention['causal_retention']['loss'],causal_validation_perplexity=retention['causal_retention']['perplexity'],
                behavior=generated,baseline=baseline,causal_degradation_percent=delta,acceptance_checks=checks,warnings=warnings,
                test_documents_used=0,previous_5A_preserved=True,
                verdict='5A.1 VÁLIDO — LISTO PARA 5B' if all(checks.values()) else '5A.1 REQUIERE REVISIÓN',
                limitations='Exact match no sustituye revisión de respuestas parafraseadas. No se utiliza conocimiento externo ni Test. No se implementa RAG.')


def save_rubric(generated,checkpoint_hash):
    write_json(RESULTS/'rubrica_generacion_sft_5A1.json',dict(version='5A.1',checkpoint_sha256=checkpoint_hash,
               suite_sha256=generated['suite_sha256'],allowed_labels=LABELS,examples=generated['suite_examples'],
               instructions='Completar manual_label y manual_natural_closure (booleano) contra contexto/evidencia. La puntuación final por sí sola no acredita cierre natural. SUPPORTED_PARTIAL no cuenta como éxito completo; no añadir conocimiento externo.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rubric',type=str);args=parser.parse_args()
    torch.set_num_threads(2);h=verify_lock();model,cp=load_model('best');require(cp['hashes']==h,'Checkpoint de otros datos/código.')
    ds=DatasetSFT5A1('validation');tokens=evaluate_tokens(model,ds);retention=evaluate_tokens(model,causal_dataset('validation'));generated=behavior(model,ds)
    if args.rubric:
        rubric=json.loads((BASE/args.rubric).read_text(encoding='utf-8'))
        require(rubric['checkpoint_sha256']==sha256(BASE/'modelos/checkpoint_sft_5A1_best.pt') and rubric['suite_sha256']==generated['suite_sha256'],'Rúbrica de otro checkpoint/suite.')
        labels={r['id']:r for r in rubric['examples']}
        for r in generated['suite_examples']:
            review=labels[r['id']];require(review['generated_response']==r['generated_response'],'Output revisado alterado.')
            require(review['manual_label'] in LABELS,'Rúbrica incompleta/etiqueta inválida.')
            require(isinstance(review['manual_natural_closure'],bool),'Revisar cierre natural: booleano requerido.')
            r['manual_label']=review['manual_label'];r['label_origin']='manual_source_rubric'
            r['manual_natural_closure']=review['manual_natural_closure'];r['natural_closure']=review['manual_natural_closure']
        positives=[r for r in generated['suite_examples'] if r['task_type']!='abstention']
        generated['metrics']['supported_answer_rate']=sum(r['manual_label']=='SUPPORTED_COMPLETE' for r in positives)/len(positives)
        generated['metrics']['supported_answer_rate_method']='manual_source_rubric';generated['metrics']['manual_review_pending']=0
        generated['metrics']['closure_review_pending']=0;generated['metrics']['natural_closure_method']='manual_source_rubric'
        generated['metrics']['natural_closure_rate']=sum(r['natural_closure'] for r in generated['suite_examples'])/len(generated['suite_examples'])
        by_id={r['id']:r for r in generated['suite_examples']}
        for p in generated['contrastive_decisions']:p['correct']=by_id[p['positive_id']]['manual_label']=='SUPPORTED_COMPLETE' and by_id[p['negative_id']]['manual_label']=='ABSTAIN_CORRECT'
        generated['metrics']['contrastive_decision_accuracy']=sum(p['correct'] for p in generated['contrastive_decisions'])/len(generated['contrastive_decisions'])
    baseline=json.loads((RESULTS/'baseline_sft_5A1.json').read_text(encoding='utf-8'));require(baseline['hashes']==h,'Baseline incompatible.')
    last=torch.load(BASE/'modelos/checkpoint_sft_5A1_last.pt',weights_only=True,map_location='cpu')
    require(last['version']=='5A.1' and last['hashes']==h and last['base_checkpoint_sha256']==cp['base_checkpoint_sha256'],'Historial last incompatible.')
    cp['history']=last['history']
    result=report_best(cp,baseline,tokens,retention,generated);write_json(RESULTS/'reporte_entrenamiento_sft_5A1.json',result)
    if not args.rubric:save_rubric(generated,sha256(BASE/'modelos/checkpoint_sft_5A1_best.pt'))
    verify_preservation();require(verify_lock()==h,'Artefactos cambiaron.');log_accesses('evaluacion');print(result['verdict'])


if __name__=='__main__':main()
