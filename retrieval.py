"""BM25 local; no generación ni decisiones de suficiencia (5C)."""
import collections
import math
import pickle
import re
import unicodedata
from functools import lru_cache
from rag_common import DATA, SOURCES, require, sha256
import json

STOPWORDS = frozenset('a al algo ante bajo con contra cual cuales como de del desde donde el ella ellas ellos en entre era es esa ese eso esta este estos estas existe ha hay la las lo los me muy o para pero por que qué quien se sin sobre son su sus tiene tienen un una unos unas y e u ser es decir'.split())

def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = ''.join(c for c in unicodedata.normalize('NFD',text) if not unicodedata.combining(c))
    return ' '.join(text.split())

def tokens(text, stopwords=True):
    words = re.findall(r'[a-z0-9]+(?:[-‐‑–][a-z0-9]+)*', normalize(text))
    terms = []
    for w in words:
        w = re.sub('[‐‑–]','-',w)
        if not stopwords or w not in STOPWORDS:
            terms.append(w)
            if stopwords and '-' in w:
                terms.extend(x for x in w.split('-') if x not in STOPWORDS)
    return terms

def build_bm25(chunks):
    postings = collections.defaultdict(list)
    lengths = []
    for i,c in enumerate(chunks):
        counts = collections.Counter(tokens(c['text']))
        lengths.append(sum(counts.values()))
        for term,tf in sorted(counts.items()):
            postings[term].append((i,tf))
    n = len(chunks)
    require(n>0 and sum(lengths)>0, 'Índice vacío')
    df = {t:len(p) for t,p in sorted(postings.items())}
    return dict(schema='bm25-5B-v1',k1=1.5,b=.75,n=n,avgdl=sum(lengths)/n,lengths=lengths,
                df=df,idf={t:math.log1p((n-f+.5)/(f+.5)) for t,f in df.items()},
                postings=dict(sorted(postings.items())),chunks=chunks)

@lru_cache(maxsize=1)
def load_index():
    metadata = json.loads((DATA/'metadata_indice.json').read_text(encoding='utf-8'))
    require(sha256(DATA/'indice_bm25.pkl') == metadata['index_sha256'], 'Hash de índice incorrecto')
    require(sha256(DATA/'chunks.jsonl') == metadata['chunks_sha256'], 'Hash de chunks incorrecto')
    # Only deserialize our local hash-verified index, never arbitrary paths.
    index = pickle.loads((DATA/'indice_bm25.pkl').read_bytes())
    require(index['schema']=='bm25-5B-v1' and index['n']==len(index['chunks']), 'Índice inválido')
    require(all(c['split'] in ('train','validation') and c['source_doc'] in SOURCES[c['split']] for c in index['chunks']), 'Test/split inválido en índice')
    return index

def retrieve(query, top_k=5):
    if not isinstance(top_k,int) or isinstance(top_k,bool) or not 1<=top_k<=100:
        raise ValueError('top_k debe ser entero entre 1 y 100')
    if not isinstance(query,str):
        raise TypeError('query debe ser texto')
    terms = sorted(set(tokens(query)))
    if not terms:
        return []
    index = load_index()
    scores = collections.defaultdict(float)
    for t in terms:
        for i,tf in index['postings'].get(t,[]):
            denominator = tf+index['k1']*(1-index['b']+index['b']*index['lengths'][i]/index['avgdl'])
            scores[i] += index['idf'][t]*tf*(index['k1']+1)/denominator
    ranked = sorted(scores,key=lambda i:(-scores[i],index['chunks'][i]['chunk_id']))[:top_k]
    return [dict(index['chunks'][i],rank=rank,score=scores[i]) for rank,i in enumerate(ranked,1)]

def signals(query, found):
    top = found[0] if found else None
    query_terms = set(tokens(query))
    s1 = top['score'] if top else 0.
    s2 = found[1]['score'] if len(found)>1 else 0.
    return dict(top1_score=s1,top2_score=s2,score_margin_top1_top2=s1-s2,
                query_term_coverage=len(query_terms & set(tokens(top['text'])))/len(query_terms) if top and query_terms else 0.,
                retrieved_chunk_length=top['bpe_tokens'] if top else 0,
                source_doc=top['source_doc'] if top else None,retrieval_rank=top['rank'] if top else None)
