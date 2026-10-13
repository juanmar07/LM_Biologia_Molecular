"""Construir y congelar 5A.2 desde TXT originales Train/Validation, sin Test."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from statistics import mean

from dataset_sft_5A2 import (BASE,DATA,RESULTS,TASKS,ABSTENTIONS,TokenizadorEducativo,documents,
    encode,prompt,require,write_json,sha256,hashes,verify_preservation,log_accesses)
# Solo reutiliza filtros y anotaciones literales del código preservado; no lee
# el dataset 5A.1 ni llama a su constructor, evaluación o entrenamiento.
from preparar_sft_5A1 import (extract,CURATED,norm,key,terms,anchors,grams,compatible_negative,
                            DEPENDENT,NOISE,REVIEW_NOISE,broken_hyphen)

VERSION='5A.2-v1.0'
INCOMPLETE=re.compile(r'\b(?:siguientes?|anteriores?|figuras?|tablas?|cuadros?|remanentes?|resultantes?)\b|\b(?:es decir|por ejemplo|como|y|o|a|de|[a-z])\.$',re.I)


def fields(r,tok):
    r.update(context_token_count=len(tok.encode(r['context'])),question_token_count=len(tok.encode(r['question'])),
             answer_token_count=len(tok.encode(r['answer'])),total_token_count=2+len(tok.encode(prompt(r)+r['answer'])))
    encode(r,tok)


def row(item,split,task,question=None,answer=None):
    r=dict(item,split=split,task_type=task,supported=task!='abstention',pair_id=None)
    r['question']=question or item['question'];r['answer']=answer or item['answer']
    r['evidence_sources']=[dict(source_doc=r['source_doc'],source_split=split,source_sha256=r['source_sha256'],
        start=r['evidence_start'],end=r['evidence_end'],text=r['evidence_text'])]
    return r


def audit(r,by_source,tok):
    source=by_source[r['source_doc']]
    require(source['split']==r['split']==r['source_split'],'Split incorrecto.')
    require(source['text'][r['evidence_start']:r['evidence_end']]==r['evidence_text'],'Span literal incorrecto.')
    literal=norm(r['evidence_text']);ctx=norm(r['context'])
    require(not INCOMPLETE.search(ctx) and not INCOMPLETE.search(r['answer']),'Referencia ausente o cierre fragmentario.')
    require(ctx==literal or literal.startswith(ctx[:-1]) and r['transformation']=='faithful_first_complete_clause','Contexto sin procedencia.')
    require(not NOISE.search(literal) and not REVIEW_NOISE.search(literal) and not broken_hyphen(literal),'Extracción rota.')
    require(r.get('curated') or not DEPENDENT.search(literal),'Antecedente ausente.')
    require(r['answer']==r['answer'].strip() and r['answer'][0].isupper() and r['answer'].endswith('.'),'Inicio/cierre de respuesta incorrectos.')
    if r['task_type']=='extractive_grounded':
        require(norm(r['answer']) in ctx,'Respuesta extractiva ausente del contexto.')
    elif r['task_type']=='paraphrastic_grounded':
        require(r['curation_reviewed'] and r['curation_id'] is not None,'Paráfrasis sin curación.')
        require(key(r['answer'])!=key(ctx),'Paráfrasis es copia exacta.')
        require(set(re.findall(r'\d+',r['answer']))<=set(re.findall(r'\d+',ctx)),'Cantidad añadida en paráfrasis.')
    else:
        require(r['answer'] in ABSTENTIONS and not anchors(r['requested_subject'])&anchors(ctx),'Negativo no insuficiente/canónico.')
    fields(r,tok)
    return dict(id=r['id'],task_type=r['task_type'],split=r['split'],source_doc=r['source_doc'],
        span_exact=True,context_source_verified=True,extractive_answer_in_context=r['task_type']=='extractive_grounded',
        paraphrase_curated=r['task_type']=='paraphrastic_grounded',negative_insufficiency_checked=r['task_type']=='abstention',
        answer_start_complete=True,EOS_and_mask_valid=True,total_tokens=r['total_token_count'])


def build(freeze):
    require(not (DATA/'dataset_lock_5A2.json').exists(),'Dataset congelado; no reconstruir.')
    verify_preservation();tok=TokenizadorEducativo().cargar();docs=documents();rejected=Counter();dedup=Counter()
    curated=json.loads((BASE/'curacion_sft_5A2.json').read_text(encoding='utf-8'))
    outputs={};pairs=[];audits=[];questions=set();claims=set();candidates_by_doc={}
    for split in ('train','validation'):
        by_source={d['source']:d for d in docs[split]}
        pools={d['source']:extract(d,tok,rejected) for d in docs[split]}
        for source,pool in pools.items():
            clean=[i for i in pool if not INCOMPLETE.search(norm(i['context'])) and not INCOMPLETE.search(i['answer'])]
            rejected['5A2_incomplete_context_or_answer']+=len(pool)-len(clean);pools[source]=clean
        for csplit,source,expression,q,category,topic,subject in CURATED:
            if csplit!=split:continue
            d=by_source[source];m=re.search(expression,d['text']);require(m is not None,'Fuente curada ausente.')
            pools[source].insert(0,dict(source_doc=source,source_split=split,source_family=d['family'],source_sha256=d['sha256'],
                evidence_text=m[0],evidence_start=m.start(),evidence_end=m.end(),context=m[0],answer=norm(m[0]),
                question=q,instruction_category=category,topic=topic,subject=subject,verb='curated',curated=True,
                transformation='whitespace_only',claim_key=key(m[0])))
        all_items=[i for pool in pools.values() for i in pool];candidates_by_doc[split]={s:len(p) for s,p in pools.items()}
        rows=[];counts=Counter();used_claims=set()
        # Cada paráfrasis sustituye a su afirmación extractiva; no se duplica un
        # hecho para aumentar el número de registros ni se inventan sinónimos.
        for ci,c in enumerate(curated):
            if c['split']!=split:continue
            matches=[i for i in all_items if c['source_hint'] in i['source_doc'] and key(i['answer']).startswith(key(c['evidence_prefix']))]
            matches=list({(i['source_doc'],i['evidence_start'],i['evidence_end']):i for i in matches}.values())
            require(len(matches)==1,'Curación debe identificar una afirmación única: '+c['evidence_prefix'])
            item=matches[0];r=row(item,split,'paraphrastic_grounded',c['question'],c['answer']);r.update(curation_id=ci,curation_reviewed=True,curation_note=c['review'])
            # Preferir la oración original completa; si el presupuesto falla,
            # conservar la cláusula autónoma ya auditada, sin cortar respuesta.
            r['context']=norm(item['evidence_text'])
            try:fields(r,tok)
            except RuntimeError:r['context']=item['context'];fields(r,tok)
            require(key(r['question']) not in questions and item['claim_key'] not in claims,'Curación duplicada entre splits.')
            rows.append(r);counts[item['source_doc']]+=1;questions.add(key(r['question']));claims.add(item['claim_key']);used_claims.add(item['claim_key'])
        limit=240 if split=='train' else 64;cap=47 if split=='train' else 30
        for s,pool in pools.items():random.Random(42+int(hashlib.sha256(s.encode()).hexdigest()[:8],16)).shuffle(pool)
        near=[];n_extract=0
        while any(pools.values()) and n_extract<limit:
            for source in sorted(pools):
                if counts[source]>=cap:
                    rejected['document_quota']+=len(pools[source]);pools[source].clear();continue
                if not pools[source] or n_extract>=limit:continue
                item=pools[source].pop();q=key(item['question']);g=grams(item['question'])
                if q in questions or item['claim_key'] in claims:
                    dedup['exact']+=1;continue
                if any(len(g&p)/max(1,len(g|p))>=.8 for p in near):dedup['near']+=1;continue
                r=row(item,split,'extractive_grounded');r['id']='draft'
                try:audit(r,by_source,tok)
                except RuntimeError as exc:rejected[str(exc)]+=1;continue
                rows.append(r);n_extract+=1;counts[source]+=1;questions.add(q);claims.add(item['claim_key']);used_claims.add(item['claim_key']);near.append(g)
        for index,r in enumerate(rows):r['id']=f'{split}-5A2-{index:05d}'
        reuse=Counter();n_negative=16 if split=='train' else 8;neg_count=0
        # Las preguntas genéricas sobre un tema completo admiten respuestas
        # parciales de subtemas; no etiquetarlas artificialmente como negativas.
        broad=re.compile(r'epigenétic|epigenómica|expresión de gen|procesos epigen|modificaciones epigen',re.I)
        positives=sorted([r for r in rows if r['task_type']=='extractive_grounded' and not broad.search(r['subject'])],key=lambda r:(counts[r['source_doc']],r['id']))
        for pos in positives:
            if neg_count>=n_negative:break
            candidates=[i for i in all_items if compatible_negative(pos,i) and reuse[i['claim_key']]<2]
            candidates.sort(key=lambda i:(counts[i['source_doc']],i['source_doc']!=pos['source_doc'],i['claim_key']))
            for other in candidates:
                neg=row(other,split,'abstention',pos['question'],ABSTENTIONS[neg_count%2]);pid=f'{split}-5A2-pair-{neg_count:03d}'
                neg.update(id=f'{split}-5A2-{len(rows):05d}',pair_id=pid,requested_subject=pos['subject'],
                    positive_reference=pos['answer'],positive_id=pos['id'],positive_source_doc=pos['source_doc'],
                    negative_reason='Tema relacionado, sin entidad solicitada/flexiones ni información necesaria para resolver la pregunta.')
                try:audit(neg,by_source,tok)
                except RuntimeError:continue
                pos['pair_id']=pid;rows.append(neg);counts[other['source_doc']]+=1;reuse[other['claim_key']]+=1;neg_count+=1
                pairs.append(dict(pair_id=pid,split=split,positive_id=pos['id'],negative_id=neg['id'],question=pos['question'],
                    positive_context=pos['context'],negative_context=neg['context'],positive_answer=pos['answer'],negative_answer=neg['answer'],audit_passed=True))
                break
        require(neg_count==n_negative,'Faltan negativos de evidencia auditables.')
        require(all(counts[s]>0 for s in by_source),'Documento sin ejemplos.')
        for r in rows:audits.append(audit(r,by_source,tok))
        outputs[split]=rows
    # Selección fijada antes de conocer cualquier salida del modelo.
    suite_rows=[]
    for task,n in zip(TASKS,(40,8,8)):
        available=[r for r in outputs['validation'] if r['task_type']==task]
        require(len(available)>=n,'Validation insuficiente para suite fija.')
        random.Random(4242+TASKS.index(task)).shuffle(available);suite_rows.extend(available[:n])
    suite=dict(version='5A.2',seed=4242,selection='stratified deterministic 40 extractive +8 paraphrastic +8 abstention before training',
        generation_config=dict(greedy=True,max_new_tokens=48,seed=42),records=suite_rows)
    for split,rows in outputs.items():
        (DATA/f'{split}_instrucciones_5A2.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (DATA/'pares_contrastivos.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs),encoding='utf-8')
    (DATA/'auditoria_evidencia.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audits),encoding='utf-8')
    write_json(DATA/'suite_validation_5A2.json',suite)
    counts={s:Counter(r['source_doc'] for r in rows) for s,rows in outputs.items()};n=len(outputs['train'])
    percent=sorted((100*c/n for c in counts['train'].values()),reverse=True)
    report=dict(version='5A.2',constructor_version=VERSION,seed=42,status='frozen' if freeze else 'draft',
        train_examples=n,validation_examples=len(outputs['validation']),
        counts_by_task_type={s:dict(Counter(r['task_type'] for r in rows)) for s,rows in outputs.items()},
        examples_by_document={s:{d:dict(count=c,percent=100*c/len(outputs[s])) for d,c in counts[s].items()} for s in outputs},
        examples_by_topic={s:dict(Counter(r['topic'] for r in rows)) for s,rows in outputs.items()},
        examples_by_category={s:dict(Counter(r['instruction_category'] for r in rows)) for s,rows in outputs.items()},
        token_statistics={s:dict(mean_context_tokens=mean(r['context_token_count'] for r in rows),mean_question_tokens=mean(r['question_token_count'] for r in rows),
            mean_answer_tokens=mean(r['answer_token_count'] for r in rows),max_total_tokens=max(r['total_token_count'] for r in rows)) for s,rows in outputs.items()},
        contrastive_pair_count={s:sum(p['split']==s for p in pairs) for s in outputs},deduplication=dict(dedup),rejected_examples=dict(rejected),
        candidate_counts_by_document=candidates_by_doc,balance=dict(max_document_percent=percent[0],top_two_document_percent=sum(percent[:2])),
        target_step_distribution=dict(extractive_grounded=75,paraphrastic_grounded=10,abstention=5,causal_retention=10),
        evidence_audits_passed=len(audits),paraphrase_method='42 explicit curated rewrites; original proposition excluded from extractive pool; no external facts',
        limitations=['Dataset size follows audited statements/document quotas; no arbitrary expansion to 800.',
            'Lexical negative filters do not prove universal absence of scientific inference.',
            'Paraphrases curated by the preparation agent; professor review remains possible, not claimed completed.',
            'Topic/category counts are inherited textual heuristics, not semantic quality scores.'],
        test_documents_used=0,previous_experiments_preserved=True,hashes=hashes(False))
    write_json(DATA/'reporte_dataset_sft_5A2.json',report)
    md=['# Auditoría de evidencia 5A.2','Fuentes originales Train/Validation. Cada paráfrasis se muestra íntegra para revisión.']
    for split,rows in outputs.items():
        paraphrases=[r for r in rows if r['task_type']=='paraphrastic_grounded'];others=[r for r in rows if r['task_type']!='paraphrastic_grounded']
        random.Random(42).shuffle(others);chosen=paraphrases+others[:max(0,(50 if split=='train' else 30)-len(paraphrases))]
        md.append(f'## {split}: {len(chosen)} registros')
        for r in chosen:md.append(f"### {r['id']} — {r['task_type']}\n\nFuente: {r['source_doc']} · Par: {r['pair_id']}\n\nContexto: {norm(r['context'])}\n\nPregunta: {r['question']}\n\nRespuesta: {r['answer']}\n\nEvidencia literal: {norm(r['evidence_text'])}\n\nSpan: {r['evidence_start']}–{r['evidence_end']} · SHA-256: {r['source_sha256']}")
    (RESULTS/'auditoria_dataset_sft_5A2.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    if freeze:write_json(DATA/'dataset_lock_5A2.json',dict(version='5A.2',constructor_version=VERSION,seed=42,hashes=hashes(False)))
    verify_preservation();log_accesses('preparacion');print(json.dumps({k:report[k] for k in ('train_examples','validation_examples','counts_by_task_type','balance')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--draft',action='store_true');g.add_argument('--freeze',action='store_true');a=p.parse_args();build(a.freeze)
