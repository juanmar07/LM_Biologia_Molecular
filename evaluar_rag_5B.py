"""Evaluación reproducible de la suite congelada; sin optimizar parámetros por resultados."""
import collections
import difflib
import json
import re
import hashlib
from rag_common import BASE, DATA, RESULTS, require, sha256, verify_preservation, write_json
from retrieval import retrieve, signals, normalize, tokens
from rag import RAG

def evidence_match(question,chunk):
    if chunk['source_doc']!=question['expected_source_doc']:
        return dict(match=False,exact=False,contiguous_coverage=0.)
    evidence=normalize(question['evidence_text'])
    text=normalize(chunk['text'])
    exact=evidence in text
    a,b=tokens(evidence,False),tokens(text,False)
    longest=difflib.SequenceMatcher(None,a,b,autojunk=False).find_longest_match(0,len(a),0,len(b)).size
    coverage=longest/len(a) if a else 0.
    return dict(match=exact or (coverage>=.8 and longest>=min(5,len(a))),exact=exact,contiguous_coverage=coverage)

def overlap(answer,reference):
    a,b=collections.Counter(tokens(answer,False)),collections.Counter(tokens(reference,False))
    matched=sum((a&b).values())
    precision=matched/sum(a.values()) if a else 0.
    recall=matched/sum(b.values()) if b else 0.
    return dict(normalized_token_precision=precision,normalized_token_recall=recall,
                normalized_token_F1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                exact_match_descriptive=normalize(answer)==normalize(reference))

def response_metrics(generation,reference,context):
    answer=generation['answer']
    words=tokens(answer,False)
    shingles=[tuple(words[i:i+4]) for i in range(len(words)-3)]
    repetition=len(shingles)!=len(set(shingles))
    norm=normalize(answer)
    abstention=any(s in norm for s in ('no tengo informacion suficiente','no contiene informacion suficiente','no hay informacion suficiente','no puedo responder'))
    values=overlap(answer,reference)
    values.update(EOS_generation_rate=generation['EOS'],natural_closure_rate=generation['EOS'] and bool(re.search(r'[.!?]\s*$',answer)),
                  repetition_rate=repetition,false_abstention_rate=abstention,
                  context_token_precision=overlap(answer,context)['normalized_token_precision'])
    return values

def classify(delivered,metrics,generation):
    if not delivered: return 'RETRIEVAL_FAILURE'
    if metrics['exact_match_descriptive']: return 'RETRIEVAL_AND_GENERATION_SUCCESS'
    if not generation['answer'].strip() or metrics['false_abstention_rate'] or metrics['repetition_rate'] or generation['stop_reason']=='context_budget':
        return 'RETRIEVAL_SUCCESS_GENERATION_FAILURE'
    return 'AMBIGUOUS_REVIEW_REQUIRED'

def review_signature(row):
    content={k:row[k] for k in ('id','question','reference','expected_source_doc')}
    content.update(context=row['packing']['context'],answer=row['generation']['answer'],
                   chunk_ids=[c['chunk_id'] for c in row['packing']['selected_chunks']])
    return hashlib.sha256(json.dumps(content,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()

def apply_reviews(report):
    path=RESULTS/'revision_fuentes_rag_5B.json'
    if not path.exists(): return report
    review=json.loads(path.read_text(encoding='utf-8'))
    require(review['suite_sha256']==report['suite_sha256'] and review['index_sha256']==sha256(DATA/'indice_bm25.pkl') and
            review['checkpoint_sha256']==report['model']['checkpoint_sha256'],'Revisión corresponde a otros artefactos')
    for row in report['rows']:
        if row['id'] in review['reviews']:
            item=review['reviews'][row['id']]
            require(item['signature']==review_signature(row),'Respuesta/contexto revisados cambiaron')
            row.setdefault('automatic_classification',row['classification'])
            row['classification']=item['classification']
            row['source_based_review']=item
    report['automatic_classification_counts']=dict(collections.Counter(r.get('automatic_classification',r['classification']) for r in report['rows']))
    report['classification_counts']=dict(collections.Counter(r['classification'] for r in report['rows']))
    report['source_review_sha256']=sha256(path)
    report['source_review_count']=len(review['reviews'])
    representative=[]
    for category in report['classification_counts']:
        representative.extend([r for r in report['rows'] if r['classification']==category and category!='RETRIEVAL_AND_GENERATION_SUCCESS'][:4])
    used={r['id'] for r in representative}
    representative.extend(r for r in report['rows'] if r['id'] not in used and r['classification']!='RETRIEVAL_AND_GENERATION_SUCCESS')
    report['representative_errors']=representative[:12]
    return report

def main():
    integrity=verify_preservation()
    suite_path=DATA/'suite_validation_5B.json'
    lock=json.loads((DATA/'suite_validation_5B.sha256.json').read_text(encoding='utf-8'))
    require(sha256(suite_path)==lock['sha256'], 'Suite congelada cambió')
    suite=json.loads(suite_path.read_text(encoding='utf-8'))['questions']
    retrieval_rows=[]
    counts=collections.Counter()
    for q in suite:
        found=retrieve(q['question'],5)
        matches=[evidence_match(q,c) for c in found]
        rank=next((i+1 for i,m in enumerate(matches) if m['match']),None)
        for k in (1,3,5):
            counts[f'source_document_recall_at_{k}']+=any(c['source_doc']==q['expected_source_doc'] for c in found[:k])
            counts[f'evidence_recall_at_{k}']+=any(m['match'] for m in matches[:k])
        counts['mean_reciprocal_rank']+=1/rank if rank else 0
        counts['mean_top1_BM25_score']+=found[0]['score'] if found else 0
        retrieval_rows.append(dict(q,retrieved=found,evidence_matches=matches,evidence_rank=rank,signals_5C=signals(q['question'],found)))
    retrieval_report=dict(version='5B',n=len(suite),suite_sha256=lock['sha256'],metrics={k:v/len(suite) for k,v in sorted(counts.items())},
        evidence_definition=lock['evidence_rule'],targets=lock['retrieval_targets'],rows=retrieval_rows,test_documents_used=0)
    write_json(RESULTS/'evaluacion_retrieval_5B.json',retrieval_report)
    system=RAG()
    rows=[]
    for i,q in enumerate(suite):
        try:
            result=system.answer(q['question'])
            delivered=any(evidence_match(q,c)['match'] for c in result['packing']['selected_chunks'])
            metrics=response_metrics(result['generation'],q['reference'],result['packing']['context'])
            row=dict(q,**{k:v for k,v in result.items() if k!='question'},metrics=metrics,
                     search_evidence_in_top5=retrieval_rows[i]['evidence_rank'] is not None,
                     supplied_context_evidence_match=delivered,
                     classification=classify(delivered,metrics,result['generation']),
                     classification_method='frozen conservative automatic rubric; nonexact fluent responses require source-based review')
        except ValueError as exc:
            row=dict(q,retrieved=retrieval_rows[i]['retrieved'],signals_5C=retrieval_rows[i]['signals_5C'],mechanical_error=str(exc),
                generation=dict(answer='',raw_generated_text='',generated_token_ids=[],EOS=False,stop_reason='mechanical_error',total_bpe_tokens=0),
                metrics=response_metrics(dict(answer='',EOS=False),q['reference'],''),classification='RETRIEVAL_FAILURE',
                supplied_context_evidence_match=False,search_evidence_in_top5=retrieval_rows[i]['evidence_rank'] is not None)
        rows.append(row)
        print(f'{i+1}/{len(suite)} {q["id"]} {row["classification"]}',flush=True)
    metric_keys=rows[0]['metrics'].keys()
    metrics={k:sum(r['metrics'][k] for r in rows)/len(rows) for k in metric_keys}
    classification_counts=dict(collections.Counter(r['classification'] for r in rows))
    errors=[r for r in rows if r['classification']!='RETRIEVAL_AND_GENERATION_SUCCESS']
    # Stratified examples cover all observed error categories, then fill to >=10 if available.
    representative=[]
    for category in classification_counts:
        if category!='RETRIEVAL_AND_GENERATION_SUCCESS':
            representative.extend([r for r in errors if r['classification']==category][:4])
    used={r['id'] for r in representative}
    representative.extend(r for r in errors if r['id'] not in used)
    representative=representative[:max(10,len(classification_counts)*4)]
    report=dict(version='5B',n=len(rows),suite_sha256=lock['sha256'],model=system.model_metadata,metrics=metrics,
                supplied_context_evidence_rate=sum(r['supplied_context_evidence_match'] for r in rows)/len(rows),
                classification_counts=classification_counts,rows=rows,representative_errors=representative,
                interpretation='Token F1/context precision/closure are descriptive proxies, not scientific validation; ambiguous cases require review. 5A.2 provisional.',
                prompt_format='Contexto/Pregunta/Respuesta as requested; trained SFT used Contexto/Usuario/Asistente',
                no_gate_5C=True,test_documents_used=0,integrity=verify_preservation())
    report=apply_reviews(report)
    write_json(RESULTS/'evaluacion_rag_5B.json',report)
    print(json.dumps(dict(retrieval=retrieval_report['metrics'],generation=metrics,classifications=classification_counts),indent=2),flush=True)

if __name__=='__main__': main()
