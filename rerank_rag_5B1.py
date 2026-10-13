"""Reranking lexical determinista sin ML, corpus externo ni reglas de abstención."""
import math
import re
from collections import Counter
from retrieval import tokens,normalize
from rag_5B1_common import require

WEIGHTS=dict(bm25=1.,coverage=4.,scientific_exact=2.,proximity=1.25,principal=2.,intent=2.5,
             informative=1.,mention_only=-3.,heading=-2.,fragment=-1.,repetition=-2.)
GENERIC=frozenset('afirma contexto indicado indicada indicadas indicados resume dato datos describe descripcion explica explicacion interpreta interpretacion informacion relaciona relacion relacionar segun define definicion funcion finalidad efecto dependencia como cuando donde sobre comparar compara diferencia diferencias entre actua realiza significa tamaño simple forma expresa permite indicado contexto proporcionado importante caracteristica caracteristicas caracteriza clasificación clasificacion'.split())
PREDICATE=re.compile(r'\b(?:es|son|se|puede|pueden|permite|permiten|contiene|contienen|tiene|tienen|consiste|consta|designa|representa|sirve|sirven|produce|producen|genera|generan|requiere|requieren|corresponde|provoca|implica|refiere|cataliza|reconoce|reconocen|forman|posee|poseen|expresa|expresan|define|definen)\b')

def query_analysis(question):
    norm=normalize(question)
    significant=list(dict.fromkeys(t for t in tokens(question) if t not in GENERIC))
    scientific=list(dict.fromkeys(normalize(t) for t in re.findall(r'\b[A-ZÁÉÍÓÚ][A-ZÁÉÍÓÚ0-9]{1,}(?:[-‐‑][A-Za-z0-9]+)?\b',question)))
    if re.search(r'que (?:es|son)|\bdefin|\bdefine',norm): intent='definition'
    elif re.search(r'funcion|para que|sirve|permite',norm): intent='function'
    elif re.search(r'diferencia|compara|\bentre\b',norm): intent='comparison'
    elif re.search(r'caracteriz|caracteristica',norm): intent='characteristic'
    else: intent='description'
    principal=scientific[0] if scientific else (significant[0] if significant else '')
    return dict(intent=intent,significant_terms=significant,scientific_terms=scientific,principal_term=principal,
                candidate_query=question)  # Original BM25 query preserved for a controlled ablation.

def intent_signal(analysis,text):
    term=re.escape(analysis['principal_term'])
    if not term: return 0.
    # Patterns are grammatical templates, never scientific answer templates.
    # Direct grammatical subject/predicate. A distant «es» in an unrelated clause,
    # or an exam question ending «es:», does not define the queried term.
    definition=bool(re.search(r'\b'+term+r'\b(?:\s*\([^)]{0,60}\))?(?:\s+(?:no|tambien|generalmente|habitualmente|principalmente))*\s+(?:es|son|consiste|se define|designa|representa|contiene)\b(?!\s*:)',text))
    function=bool(re.search(r'\b(?:permite|permiten|sirve|sirven|se utiliza|se utilizan|funcion|se emplea|se emplean|para producir|cataliza)\b',text))
    comparison=bool(re.search(r'diferencia|respectivamente|mientras|en lugar de|en cambio|a diferencia|pero',text))
    characteristic=bool(re.search(r'se caracteriza|caracteristica|presenta|posee|tiene|consta',text))
    return float(dict(definition=definition,function=function,comparison=comparison,characteristic=characteristic,
                      description=bool(PREDICATE.search(text)))[analysis['intent']])

def rerank(question,found):
    if not found: return []
    analysis=query_analysis(question)
    query=set(analysis['significant_terms'])
    scientific=set(analysis['scientific_terms'])
    max_score=max(c['score'] for c in found) or 1.
    result=[]
    for c in found:
        text=normalize(c['text'])
        wordlist=tokens(c['text'],False)
        present=set(tokens(c['text']))
        matches=query & present
        positions=[i for i,w in enumerate(wordlist) if w in query]
        if len(matches)>1:
            # Smallest window covering the matched query terms (missing terms lower coverage).
            windows=[]
            for p in positions:
                covered=set()
                for q in positions:
                    if q<p: continue
                    covered.add(wordlist[q])
                    if matches <= covered:
                        windows.append(q-p+1); break
            proximity=len(matches)/min(windows) if windows else 0.
        else: proximity=1. if matches else 0.
        principal=bool(analysis['principal_term'] and re.search(r'\b'+re.escape(analysis['principal_term'])+r'\b',text))
        scientific_exact=sum(bool(re.search(r'\b'+re.escape(t)+r'\b',text)) for t in scientific)/len(scientific) if scientific else 0.
        predicate=bool(PREDICATE.search(text))
        informative=predicate and len(wordlist)>=6
        grams=[tuple(wordlist[i:i+4]) for i in range(len(wordlist)-3)]
        repetition=(len(grams)-len(set(grams)))/max(1,len(grams))
        heading=bool(re.match(r'^\d+(?:\.\d+)+\s',text)) or (not predicate and len(wordlist)<=14)
        signals=dict(bm25=c['score']/max_score,coverage=len(matches)/len(query) if query else 0.,
            scientific_exact=scientific_exact,proximity=proximity,principal=float(principal),
            intent=intent_signal(analysis,text),informative=float(informative),
            mention_only=float(bool(matches) and not informative),heading=float(heading),
            fragment=float(c['sentence_fragment'] or not re.search(r'[.!?]["»”\)]*$',text)),repetition=repetition)
        score=sum(WEIGHTS[k]*v for k,v in signals.items())
        result.append(dict(c,bm25_rank=c['rank'],bm25_score=c['score'],rerank_score=score,
                           rerank_signals=signals,query_analysis=analysis,
                           rerank_summary=f"intent={analysis['intent']}; principal={analysis['principal_term']}; coverage={signals['coverage']:.2f}; intent_match={signals['intent']}; informative={informative}; fragment={bool(signals['fragment'])}; repetition={repetition:.2f}"))
    result.sort(key=lambda c:(-c['rerank_score'],-c['score'],c['chunk_id']))
    return [dict(c,rerank_rank=i,rank=i) for i,c in enumerate(result,1)]
