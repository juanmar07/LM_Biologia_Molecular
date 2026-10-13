"""Fidelidad 5A.2: generación real, diagnóstico léxico y rúbrica de fuente."""
import argparse
import json
import math
import re
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_sft_5A2 import BASE,DATA,RESULTS,ABSTENTIONS,DatasetSFT5A2,causal_dataset,load_model,prompt,require,write_json,verify_lock,verify_preservation,sha256,log_accesses
LABELS=('SUPPORTED_COMPLETE','SUPPORTED_PARTIAL','UNSUPPORTED_OR_WRONG','ABSTAIN_CORRECT','ABSTAIN_INCORRECT')

def normalized(s):
    return ' '.join(s.split())


def is_abstention(s):
    s=s.strip()
    return (s.startswith('No tengo información suficiente') or s.startswith('El contexto proporcionado no contiene información suficiente')
            or bool(re.match(r'^(?:No (?:puedo responder|dispongo de (?:información|datos)|se puede responder)|No hay (?:información|datos|evidencia) suficiente|El contexto (?:es insuficiente|no permite responder|no proporciona información))',s,re.I)))


@torch.no_grad()
def evaluate_tokens(model,ds,batch_size=16):
    model.eval();sums={};eos1=eos5=eosn=0
    offset=0
    for x,y in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
        logits,_=model(x)
        losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='none').reshape_as(y)
        require(torch.isfinite(losses).all().item(),'Loss no finita.')
        for j in range(len(x)):
            task=ds.records[offset+j]['task_type'] if isinstance(ds,DatasetSFT5A2) else 'causal_retention'
            b=sums.setdefault(task,dict(ce_sum=0.,targets=0))
            b['ce_sum']+=losses[j].double().sum().item();b['targets']+=(y[j]!=-100).sum().item()
        offset+=len(x)
        if isinstance(ds,DatasetSFT5A2):
            e=logits[y==3];eosn+=len(e);eos1+=(e.argmax(-1)==3).sum().item();eos5+=(e.topk(5,-1).indices==3).any(-1).sum().item()
    result={t:b|dict(loss=b['ce_sum']/b['targets'],perplexity=math.exp(min(700,b['ce_sum']/b['targets']))) for t,b in sums.items()}
    if eosn:result['EOS']=dict(EOS_accuracy_reference_end=eos1/eosn,EOS_top5_reference_end=eos5/eosn,targets=eosn)
    return result


def grounded(tokens):
    rows=[tokens[t] for t in ('extractive_grounded','paraphrastic_grounded')]
    n=sum(r['targets'] for r in rows);loss=sum(r['ce_sum'] for r in rows)/n
    return dict(loss=loss,perplexity=math.exp(loss),targets=n)


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


def lexical_diagnostics(text,reference,context):
    words=lambda s:re.findall(r'\w+',s.lower())
    g=Counter(words(text));ref=Counter(words(reference));ctx=Counter(words(context))
    return dict(reference_exact=normalized(text)==normalized(reference),
        extractive_span_match=bool(normalized(text)) and normalized(text) in normalized(context),
        context_token_precision=sum((g&ctx).values())/max(1,sum(g.values())),
        reference_token_recall=sum((g&ref).values())/max(1,sum(ref.values())))


def summarize(rows,all_outputs=None):
    pos=[r for r in rows if r['supported']];neg=[r for r in rows if not r['supported']]
    rate=lambda rs,k:sum(r[k] for r in rs)/len(rs) if rs else 0.
    ext=[r for r in pos if r['task_type']=='extractive_grounded'];para=[r for r in pos if r['task_type']=='paraphrastic_grounded']
    allpos=[r for r in (all_outputs or rows) if r['supported']]
    return dict(answerable_count=len(pos),unanswerable_count=len(neg),
        fidelity_reference_exact_rate=rate(pos,'reference_exact'),extractive_exact_rate=rate(ext,'reference_exact'),
        paraphrastic_reference_exact_rate=rate(para,'reference_exact'),extractive_span_match_rate=rate(ext,'extractive_span_match'),
        supported_answer_rate=sum(r['manual_label']=='SUPPORTED_COMPLETE' for r in pos)/len(pos),
        supported_answer_rate_method='reference_exact_lower_bound_pending_manual_source_review',
        context_token_precision=rate(pos,'context_token_precision'),reference_token_recall=rate(pos,'reference_token_recall'),
        false_abstention_rate=rate(allpos,'abstention_prefix'),false_abstention_denominator=len(allpos),suite_false_abstention_rate=rate(pos,'abstention_prefix'),
        abstention_exact_rate=rate(neg,'abstention_exact'),abstention_prefix_rate=rate(neg,'abstention_prefix'),
        EOS_generation_rate=rate(pos,'EOS_generated'),overall_EOS_generation_rate=rate(rows,'EOS_generated'),
        natural_closure_rate=rate(pos,'natural_closure'),manual_review_pending=sum(r['manual_label'] is None for r in rows),
        closure_review_pending=sum(r['manual_natural_closure'] is None for r in rows),
        lexical_metric_limit='Overlap and span match are diagnostics, not semantic proof of fidelity or completeness.')


def output(model,ds,r):
    g=generate(model,ds.tokenizer,r);diag=lexical_diagnostics(g['generated_response'],r['answer'],r['context'])
    exact=diag['reference_exact'];label='SUPPORTED_COMPLETE' if r['supported'] and exact else 'ABSTAIN_CORRECT' if not r['supported'] and g['abstention_exact'] else 'ABSTAIN_INCORRECT' if not r['supported'] and not g['abstention_prefix'] else None
    return dict(id=r['id'],pair_id=r['pair_id'],task_type=r['task_type'],supported=r['supported'],source_doc=r['source_doc'],
        context=r['context'],evidence_text=r['evidence_text'],question=r['question'],reference=r['answer'],**g,**diag,
        manual_label=label,manual_natural_closure=g['natural_closure'] if exact or g['abstention_exact'] else None,
        label_origin='exact_reference_check' if label else 'pending_source_review')


def behavior(model,ds):
    suite=json.loads((DATA/'suite_validation_5A2.json').read_text(encoding='utf-8'));cache={}
    for r in suite['records']:cache[r['id']]=output(model,ds,r)
    rows=[cache[r['id']] for r in suite['records']]
    all_rows=[cache.get(r['id']) or output(model,ds,r) for r in ds.records]
    return dict(metrics=summarize(rows,all_rows),suite_examples=rows,all_validation_outputs=all_rows,
        suite_sha256=sha256(DATA/'suite_validation_5A2.json'),generation_config=suite['generation_config'])


SELECTION_POLICY=dict(primary='maximize fixed-suite fidelity_reference_exact_rate on answerable examples',
    tie_break='minimize combined token-weighted grounded Validation CE; then earliest epoch',
    false_abstention_max=.10,EOS_generation_min=.80,causal_degradation_max_exclusive_percent=10.,
    fallback='same fidelity-first ordering with PROVISIONAL_REVIEW_REQUIRED if no epoch eligible',
    early_stopping='combined grounded CE: min_delta=.001 patience=2; no composite behavioral score')


def eligible(tokens,retention,generated,baseline):
    m=generated['metrics'];delta=100*(retention['causal_retention']['loss']/baseline['retention']['causal_retention']['loss']-1)
    checks=dict(false_abstention=m['false_abstention_rate']<=.1,EOS=m['EOS_generation_rate']>=.8,retention=delta<10)
    return all(checks.values()),dict(checks=checks,causal_degradation_percent=delta)


def report_best(cp,baseline,tokens,retention,generated):
    m=generated['metrics'];delta=100*(retention['causal_retention']['loss']/baseline['retention']['causal_retention']['loss']-1)
    checks=dict(supported_answer_rate=m['supported_answer_rate']>=.8,false_abstention=m['false_abstention_rate']<=.1,
        natural_closure=m['natural_closure_rate']>=.9,EOS=m['EOS_generation_rate']>=.8,retention=delta<10,
        review_complete=m['manual_review_pending']==0 and m['closure_review_pending']==0,
        eligible=cp['selection_status']=='ELIGIBLE')
    return dict(version='5A.2',best_epoch=cp['epoch'],base_checkpoint_sha256=cp['base_checkpoint_sha256'],hashes=cp['hashes'],
        training_config=cp['training_config'],selection_policy=SELECTION_POLICY,selection_status=cp['selection_status'],
        history=cp['history'],grounded_validation=grounded(tokens),validation_by_task=tokens,causal_validation=retention,
        behavior=generated,baseline=baseline,causal_degradation_percent=delta,acceptance_checks=checks,
        test_documents_used=0,previous_experiments_preserved=True,
        verdict='5A.2 FIDELIDAD VALIDADA' if all(checks.values()) else '5A.2 REQUIERE REVISIÓN',
        limitations='Exact match is conservative. Review paraphrases against source, not external knowledge. Lexical precision does not prove grounding. Abstention is reported as secondary behavior; no RAG implemented.')


def save_rubric(generated,checkpoint_hash):
    write_json(RESULTS/'rubrica_generacion_sft_5A2.json',dict(version='5A.2',checkpoint_sha256=checkpoint_hash,
        suite_sha256=generated['suite_sha256'],allowed_labels=LABELS,examples=generated['suite_examples'],
        instructions='Complete manual_label and boolean manual_natural_closure using the evidence. Partial support is not full success. Lexical overlap alone does not establish support.'))


def apply_rubric(generated,rubric,checkpoint_hash):
    require(rubric['checkpoint_sha256']==checkpoint_hash and rubric['suite_sha256']==generated['suite_sha256'],'Rúbrica incompatible.')
    labels={r['id']:r for r in rubric['examples']}
    require(set(labels)=={r['id'] for r in generated['suite_examples']},'IDs de rúbrica incorrectos/duplicados.')
    for r in generated['suite_examples']:
        review=labels[r['id']];require(review['generated_response']==r['generated_response'],'Output revisado alterado.')
        require(review['manual_label'] in LABELS and isinstance(review['manual_natural_closure'],bool),'Rúbrica incompleta.')
        require((r['supported'] and review['manual_label'] in LABELS[:3]) or (not r['supported'] and review['manual_label'] in LABELS[3:]),'Etiqueta incompatible con respondibilidad.')
        r['manual_label']=review['manual_label'];r['manual_natural_closure']=review['manual_natural_closure'];r['natural_closure']=review['manual_natural_closure'];r['label_origin']='manual_source_rubric'
    generated['metrics']=summarize(generated['suite_examples'],generated['all_validation_outputs'])
    generated['metrics']['supported_answer_rate_method']='manual_source_rubric'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rubric');a=p.parse_args();torch.set_num_threads(2)
    h=verify_lock();m,cp=load_model('best');require(cp['hashes']==h,'Checkpoint incompatible.')
    ds=DatasetSFT5A2('validation');tokens=evaluate_tokens(m,ds);ret=evaluate_tokens(m,causal_dataset('validation'));g=behavior(m,ds)
    if a.rubric:apply_rubric(g,json.loads((BASE/a.rubric).read_text(encoding='utf-8')),sha256(BASE/'modelos/checkpoint_sft_5A2_best.pt'))
    else:save_rubric(g,sha256(BASE/'modelos/checkpoint_sft_5A2_best.pt'))
    b=json.loads((RESULTS/'baseline_sft_5A2.json').read_text(encoding='utf-8'));require(b['hashes']==h,'Baseline incompatible.')
    last=torch.load(BASE/'modelos/checkpoint_sft_5A2_last.pt',weights_only=True,map_location='cpu');require(last['hashes']==h and last['version']=='5A.2','Last incompatible.');cp['history']=last['history']
    result=report_best(cp,b,tokens,ret,g);write_json(RESULTS/'reporte_entrenamiento_sft_5A2.json',result);verify_preservation();log_accesses('evaluacion');print(result['verdict'])


if __name__=='__main__':main()

