"""Ablación A/B/C/D y oráculo directo, sólo sobre Validation 5B congelada."""
import collections
import hashlib
import json
from rag_5B1_common import BASE,DATA,STAGE,RESULTS,require,sha256,write_json,preservation,sft_prompt
from rag import RAG
from retrieval import retrieve
from rerank_rag_5B1 import WEIGHTS
from contexto_rag_5B1 import pack_neighbors,make_packed
from evaluar_rag_5B import evidence_match,response_metrics,classify

CODE=('rag.py','rag_5B1_common.py','rerank_rag_5B1.py','contexto_rag_5B1.py','evaluar_rag_5B1.py')

def signature(row):
    obj=dict(question=row['question'],reference=row['reference'],context=row['packing']['context'],answer=row['generation']['answer'])
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def freeze_experiment(system,suite):
    path=STAGE/'experimento_congelado_5B1.json'
    code_hashes={p:sha256(BASE/p) for p in CODE}
    if path.exists():
        lock=json.loads(path.read_text(encoding='utf-8'))
        require(lock['code_hashes']==code_hashes,'Código de experimento cambió; conservar corrida y revisar protocolo')
        require(lock['suite_sha256']==sha256(DATA/'suite_validation_5B.json'),'Suite cambió')
        return lock
    # Mechanical fit selection, before observing a single oracle result.
    eligible=[r for r in suite if make_packed(r['evidence_text'],r['question'],[],system.tokenizer,32)]
    groups={s:[r for r in eligible if r['expected_source_doc']==s] for s in sorted({r['expected_source_doc'] for r in eligible})}
    oracle=[]
    while len(oracle)<24 and any(groups.values()):
        for source in groups:
            if groups[source] and len(oracle)<24: oracle.append(groups[source].pop(0)['id'])
    require(len(oracle)>=20,'Menos de 20 oráculos íntegros que caben')
    lock=dict(seed=42,suite_sha256=sha256(DATA/'suite_validation_5B.json'),index_sha256=sha256(DATA/'indice_bm25.pkl'),
        checkpoint_sha256=system.model_metadata['checkpoint_sha256'],code_hashes=code_hashes,weights=WEIGHTS,
        bm25_candidates=10,question_rewrite=False,oracle_ids=oracle,
        oracle_selection='24 deterministic document round-robin among frozen suite rows whose FULL literal evidence fits with reserve32; no answer-based selection',
        oracle_eligible_count=len(eligible),minimum_response_tokens=24,preferred_response_tokens=32,
        evidence_match_rule='Same frozen 5B expected-source and exact/80% contiguous evidence overlap; D evaluates budgeted neighbor-context units, also stores raw chunk metrics',
        scientific_support_policy='Source-based review required for nonexact responses; no F1 threshold used as scientific truth',frozen_before_metrics=True)
    write_json(path,lock)
    return lock

def retrieval_summary(question,units):
    found=[evidence_match(question,c)['match'] for c in units]
    first=next((i+1 for i,value in enumerate(found) if value),None)
    return {**{f'evidence_recall_at_{k}':float(any(found[:k])) for k in (1,3,5)},
            **{f'source_document_recall_at_{k}':float(any(c['source_doc']==question['expected_source_doc'] for c in units[:k])) for k in (1,3,5)},
            'MRR_at_5':1/first if first and first<=5 else 0.,'evidence_rank':first}

def attach_result(q,result):
    packed=result['packing']
    delivered=evidence_match(q,dict(source_doc=packed['selected_chunks'][0]['source_doc'],text=packed['context']))['match']
    metrics=response_metrics(result['generation'],q['reference'],packed['context'])
    return dict(q,**{k:v for k,v in result.items() if k!='question'},metrics=metrics,
        supplied_context_evidence_match=delivered,classification=classify(delivered,metrics,result['generation']),
        supported_answer=None,scientific_review_pending=True)

def apply_scientific_reviews(report,oracle):
    path=RESULTS/'revision_cientifica_rag_5B1.json'
    if not path.exists(): return report,oracle
    reviews=json.loads(path.read_text(encoding='utf-8'))
    require(reviews['experiment_sha256']==sha256(STAGE/'experimento_congelado_5B1.json'),'Revisión obsoleta')
    for variant,section in report['variants'].items():
        for row in section['rows']:
            key=variant+':'+row['id']
            if key in reviews['reviews']:
                r=reviews['reviews'][key]
                require(r['signature']==signature(row),'Respuesta/contexto revisados cambiaron')
                row['scientific_review']=r
                row['scientific_review_pending']=False
                row['supported_answer']=r['supported_answer']
                row['classification']=r['classification']
        section['classification_counts']=dict(collections.Counter(r['classification'] for r in section['rows']))
    for row in oracle['rows']:
        r=reviews['reviews'].get('ORACLE:'+row['id'])
        if r:
            require(r['signature']==signature(row),'Oráculo revisado cambió')
            row['scientific_review']=r; row['scientific_review_pending']=False
            row['supported_answer']=r['supported_answer']
    decided=[r for r in oracle['rows'] if r['supported_answer'] is not None]
    oracle['metrics']['supported_answer_rate']=sum(r['supported_answer'] for r in decided)/len(decided) if decided else None
    oracle['metrics']['supported_answer_reviewed_count']=len(decided)
    oracle['review_sha256']=sha256(path)
    report['review_sha256']=sha256(path)
    return report,oracle

def main():
    preservation()
    suite_path=DATA/'suite_validation_5B.json'
    prior=json.loads((DATA/'suite_validation_5B.sha256.json').read_text(encoding='utf-8'))
    require(sha256(suite_path)==prior['sha256'],'Suite 5B alterada')
    suite=json.loads(suite_path.read_text(encoding='utf-8'))['questions']
    system=RAG()
    lock=freeze_experiment(system,suite)
    baseline=json.loads((RESULTS/'evaluacion_rag_5B.json').read_text(encoding='utf-8'))
    old={r['id']:r for r in baseline['rows']}
    variant_sections={}
    retrieval_sections={}
    for variant in ('A','B','C','D'):
        rows=[]; retrieval_rows=[]
        for q in suite:
            result=system.answer(q['question'],top_k=5 if variant in ('A','B') else 10,variant=variant)
            if variant=='A':
                require(result['generation']['generated_token_ids']==old[q['id']]['generation']['generated_token_ids'], 'A no reproduce baseline original')
            row=attach_result(q,result)
            base_units=result['retrieved'][:5]
            units=[]
            if variant=='D':
                for seed in base_units:
                    try:
                        p=pack_neighbors(q['question'],[seed],system.tokenizer)
                        units.append(dict(seed,text=p['context'],neighbor_chunks=[c['chunk_id'] for c in p['selected_chunks']],
                            prompt_bpe_tokens=p['prompt_bpe_tokens'],response_reserve=p['minimum_response_reserve']))
                    except ValueError:
                        units.append(dict(seed,text='',mechanical_budget_failure=True))
            else: units=base_units
            retrieval_values=retrieval_summary(q,units)
            retrieval_rows.append(dict(id=q['id'],question=q['question'],metrics=retrieval_values,
                units=units,base_chunk_metrics=retrieval_summary(q,base_units),bm25_candidates=result['bm25_retrieved']))
            rows.append(row)
        gm={k:sum(r['metrics'][k] for r in rows)/len(rows) for k in rows[0]['metrics']}
        rm={k:sum(r['metrics'][k] for r in retrieval_rows)/len(rows) for k in retrieval_rows[0]['metrics'] if k!='evidence_rank'}
        gm['supplied_context_evidence_rate']=sum(r['supplied_context_evidence_match'] for r in rows)/len(rows)
        variant_sections[variant]=dict(n=len(rows),metrics=gm,classification_counts=dict(collections.Counter(r['classification'] for r in rows)),rows=rows)
        retrieval_sections[variant]=dict(n=len(rows),metrics=rm,unit_definition='budgeted same-source neighbor context per seed' if variant=='D' else 'original indexed chunk',rows=retrieval_rows)
        print(variant,json.dumps(dict(retrieval=rm,generation=gm,errors=variant_sections[variant]['classification_counts'])),flush=True)
    oracle_rows=[]
    from rag import greedy_generate
    by_id={r['id']:r for r in suite}
    for identity in lock['oracle_ids']:
        q=by_id[identity]
        packed=make_packed(q['evidence_text'],q['question'],[],system.tokenizer,32)
        require(packed and packed['context']==q['evidence_text'],'Oráculo no tiene evidencia literal íntegra')
        gen=greedy_generate(system.model,system.tokenizer,packed)
        metrics=response_metrics(gen,q['reference'],packed['context'])
        oracle_rows.append(dict(q,packing={k:v for k,v in packed.items() if k!='prompt_ids'},generation=gen,metrics=metrics,
                               supplied_context_evidence_match=True,supported_answer=None,scientific_review_pending=True))
    oracle=dict(n=len(oracle_rows),retrieval_used=False,full_literal_evidence=True,model=system.model_metadata,
        metrics={k:sum(r['metrics'][k] for r in oracle_rows)/len(oracle_rows) for k in oracle_rows[0]['metrics']},rows=oracle_rows,
        oracle_selection=lock['oracle_selection'],experiment_sha256=sha256(STAGE/'experimento_congelado_5B1.json'))
    report=dict(version='5B.1',suite_sha256=lock['suite_sha256'],experiment_sha256=sha256(STAGE/'experimento_congelado_5B1.json'),
        model=system.model_metadata,variants=variant_sections,integrity=preservation(),A_reproduces_original_generation=True,
        annotation='A original; B changes serialization only; C additionally BM25top10+rerank; D additionally neighbor packing with response reserve24–32; D context-recall differs from raw chunk recall, both saved',no_gate_5C=True)
    report,oracle=apply_scientific_reviews(report,oracle)
    write_json(RESULTS/'evaluacion_retrieval_5B1.json',dict(version='5B.1',suite_sha256=lock['suite_sha256'],variants=retrieval_sections,experiment_sha256=report['experiment_sha256'],test_documents_used=0))
    write_json(RESULTS/'evaluacion_rag_5B1.json',report)
    write_json(RESULTS/'evaluacion_oraculo_generacion_5B1.json',oracle)
    print('ORACLE',json.dumps(oracle['metrics']),flush=True)

if __name__=='__main__': main()
