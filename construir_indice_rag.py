"""Congela Validation antes de recuperar y construye solo los 18 TXT permitidos."""
import collections
import json
import pickle
import re
from rag_common import BASE, DATA, RESULTS, documents, require, sha256, write_json, verify_preservation
from retrieval import build_bm25
from tokenizador import TokenizadorEducativo

def freeze_suite(docs):
    suite_path = DATA/'suite_validation_5B.json'
    lock_path = DATA/'suite_validation_5B.sha256.json'
    if suite_path.exists():
        require(sha256(suite_path)==json.loads(lock_path.read_text(encoding='utf-8'))['sha256'], 'Suite congelada alterada')
        return
    source_path = BASE/'datos/sft_5A2/validation_instrucciones_5A2.jsonl'
    rows = [json.loads(s) for s in source_path.read_text(encoding='utf-8').splitlines()]
    rows = sorted([r for r in rows if r['supported'] and r['task_type'] in ('extractive_grounded','paraphrastic_grounded')],key=lambda r:r['id'])
    require(len(rows)>=40, 'Validation respondible insuficiente')
    corpus = {d['source_doc']:d for d in docs if d['split']=='validation'}
    suite = []
    for r in rows:
        require(r['source_split']=='validation' and r['source_doc'] in corpus, 'Fuente Validation inválida')
        require(corpus[r['source_doc']]['text'][r['evidence_start']:r['evidence_end']]==r['evidence_text'], 'Span de evidencia inválido')
        suite.append(dict(id=r['id'],question=r['question'],reference=r['answer'],expected_source_doc=r['source_doc'],
                          evidence_text=r['evidence_text'],evidence_start=r['evidence_start'],evidence_end=r['evidence_end'],task_type=r['task_type']))
    write_json(suite_path,dict(seed=42,selection='ALL grounded answerable Validation 5A.2, sorted by id; no result-based filtering',
                             source_sha256=sha256(source_path),questions=suite))
    write_json(lock_path,dict(sha256=sha256(suite_path),frozen_before_retrieval=True,n=len(suite),
        evidence_rule='same expected source AND exact normalized evidence substring OR longest contiguous normalized token overlap >=80% of evidence tokens and at least 5 tokens (or all when shorter)',
        evidence_overlap_min=.8,scientific_fidelity_note='Evidence overlap is a retrieval proxy, not proof of scientific completeness.',
        generation_classification_rule='supplied evidence missing -> RETRIEVAL_FAILURE; exact reference -> SUCCESS; empty/abstention/repeated/budget-stop -> GENERATION_FAILURE; otherwise AMBIGUOUS_REVIEW_REQUIRED',
        repetition_rule='any repeated 4-word sequence; natural closure proxy = EOS and terminal sentence punctuation',
        retrieval_targets=dict(source_at_3=.90,evidence_at_3=.80,evidence_at_5=.90)))

NOISE = re.compile(r'https?://|www\.|@|doi\s*:|isbn|©|todos los derechos|all rights reserved',re.I)
FINITE_VERB = re.compile(r'\b(?:es|son|era|eran|se|tiene|tienen|puede|pueden|permite|permiten|consiste|consisten|incluye|incluyen|contiene|contienen|sirve|sirven|requiere|requieren|produce|producen|genera|generan|consta|refiere|representa|corresponde|provoca|sufren|necesita|implica|designa|realiza|utiliza|forman|existe|existen|ha|han|hay|está|están|estaba|actúa|posee|poseen|debe|deben|detecta|cataliza|reconocen|reconoce|corta|cortan|codifica|codifican)\b',re.I)

def isolated_noise(text, n, fragment=False):
    # These are structural filters, not query/score/answer-dependent filters.
    value=text.strip()
    if NOISE.search(value):
        return True
    if not FINITE_VERB.search(value) and (re.match(r'^\d+(?:\.\d+){1,}\s',value) or
            re.search(r'clinical applications of|aplicaciones clínicas de la herramienta',value,re.I)):
        return True
    if re.search(r'\bV\.\s+[a-záéíóú]|(?:restricción-modificación|restrictasas)[,\.]',value,re.I):
        return True
    if not fragment and value.endswith('?'):
        return True  # A standalone exercise question is not an evidence statement.
    if not fragment and n<=32 and not FINITE_VERB.search(value):
        return True  # Isolated title/index/caption, no declarative scientific proposition.
    return False

def regions(text, rejects):
    # Exact offsets; filter only unmistakable standalone lines, retain scientific prose.
    active = None
    bibliographic = False
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        words = re.findall(r'\w+',stripped)
        heading = bool(stripped) and (stripped.casefold() in ('bibliografía','bibliografia','referencias bibliográficas','references','índice','indice') or
            (len(words)<=12 and stripped.isupper()) or bool(re.fullmatch(r'[\d\W_]+',stripped)) or
            bool(re.search(r'\.{4,}\s*\d+\s*$',stripped)))
        if stripped.casefold() in ('bibliografía','bibliografia','referencias bibliográficas','references'):
            bibliographic = True
        # Bibliographic heading itself is excluded. Citation-looking entries are excluded,
        # but never suppress the rest of a book merely because an earlier chapter had references.
        citation = bibliographic and bool(re.search(r'\b(?:19|20)\d{2}\b',stripped)) and bool(re.search(r'\bet al\b|\b[A-Z]\.|\bpp?\.|\d+\s*:\s*\d+',stripped))
        bad = heading or bool(NOISE.search(stripped)) or citation or (stripped and len(words)<2)
        if bad:
            rejects['noise_or_isolated_heading_lines'] += 1
            if active is not None:
                yield active,offset
                active = None
        elif stripped and active is None:
            active = offset
        offset += len(line)
    if active is not None:
        yield active,len(text)

def sentence_spans(text,start,end):
    # Sentence punctuation followed by whitespace; decimals remain intact. Paragraph breaks
    # also delimit regions. Abbreviations may be split: explicitly documented limitation.
    cursor = start
    for m in re.finditer(r'(?<=[.!?])\s+|\n\s*\n',text[start:end]):
        boundary = start+m.start()
        if cursor<boundary:
            yield cursor,boundary
        cursor = start+m.end()
    if cursor<end:
        yield cursor,end

def chunk_document(doc,tokenizer,stats):
    text=doc['text']
    spans=[]
    for a,b in regions(text,stats):
        spans.extend(sentence_spans(text,a,b))
    chunks=[]
    pending=None
    def emit(a,b,fragment=False,overlap=0):
        while a<b and text[a].isspace(): a+=1
        while b>a and text[b-1].isspace(): b-=1
        value=text[a:b]
        ids=tokenizer.encode(value)
        if not value or len(re.findall(r'\w+',value))<3:
            stats['short_fragments_rejected']+=1
            return
        if isolated_noise(value,len(ids),fragment):
            stats['isolated_titles_indices_questions_rejected']+=1
            return
        require(0<len(ids)<=48 and all(t>=4 for t in ids),'Chunk fuera de límite o special token literal')
        require(tokenizer.decode(ids)==value,'BPE no preservó el texto')
        i=len(chunks)
        prefix=__import__('hashlib').sha256(doc['source_doc'].encode()).hexdigest()[:12]
        chunks.append(dict(chunk_id=f'{doc["split"]}-{prefix}-{i:06d}',source_doc=doc['source_doc'],split=doc['split'],chunk_index=i,
                           text=value,bpe_tokens=len(ids),char_start_if_available=a,char_end_if_available=b,
                           sentence_fragment=fragment,overlap_bpe_tokens=overlap))
    for a,b in spans:
        while a<b and text[a].isspace(): a+=1
        while b>a and text[b-1].isspace(): b-=1
        if a==b: continue
        n=len(tokenizer.encode(text[a:b]))
        if isolated_noise(text[a:b],n,False) and n<=48:
            stats['isolated_titles_indices_questions_rejected']+=1
            if pending is not None: emit(*pending); pending=None
            continue
        if n<=48:
            # Merge short consecutive sentences only if the exact contiguous span fits target.
            if pending is not None and a-pending[1]<=4 and len(tokenizer.encode(text[pending[0]:b]))<=40:
                pending=(pending[0],b)
            else:
                if pending is not None: emit(*pending)
                pending=(a,b)
            continue
        if pending is not None:
            emit(*pending); pending=None
        words=list(re.finditer(r'\S+',text[a:b]))
        left=0
        previous_end=0
        while left<len(words):
            right=left
            while right<len(words):
                length=len(tokenizer.encode(text[a+words[left].start():a+words[right].end()]))
                if length>48: break
                right+=1
                if length>=40: break
            if right==left:
                stats['oversized_whole_words_rejected']+=1
                left+=1; previous_end=left
                continue
            overlap=len(tokenizer.encode(text[a+words[left].start():a+words[previous_end-1].end()])) if previous_end>left else 0
            emit(a+words[left].start(),a+words[right-1].end(),True,overlap)
            if right==len(words): break
            next_left=right
            while next_left>left+1 and len(tokenizer.encode(text[a+words[next_left-1].start():a+words[right-1].end()]))<=8:
                next_left-=1
            previous_end=right
            left=next_left
    if pending is not None: emit(*pending)
    return chunks

def main():
    verify_preservation()
    docs=documents()
    freeze_suite(docs)  # This happens before indexing or any retrieval results.
    tokenizer=TokenizadorEducativo().cargar()
    stats=collections.Counter()
    chunks=[]
    summary=[]
    for d in docs:
        part=chunk_document(d,tokenizer,stats)
        require(part,'Documento sin chunks: '+d['source_doc'])
        chunks.extend(part)
        summary.append({k:v for k,v in d.items() if k!='text'} | dict(chunks=len(part),characters=len(d['text'])))
        print(d['split'],len(part),d['source_doc'],flush=True)
    DATA.mkdir(parents=True,exist_ok=True)
    (DATA/'chunks.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False,sort_keys=True)+'\n' for c in chunks),encoding='utf-8')
    index=build_bm25(chunks)
    (DATA/'indice_bm25.pkl').write_bytes(pickle.dumps(index,protocol=4))
    meta=dict(version='5B',documents=summary,total_documents=len(docs),total_chunks=len(chunks),
        splits=dict(collections.Counter(c['split'] for c in chunks)),k1=1.5,b=.75,avgdl=index['avgdl'],vocabulary=len(index['df']),
        chunking=dict(preferred_bpe=40,max_bpe=48,overlap_target_bpe=8,overlap_policy='whole-word overlap within split long sentences; intact sentences take priority over overlap',
                      sentence_aware=True,source_text_unchanged=True,rejected=dict(stats),fragments=sum(c['sentence_fragment'] for c in chunks)),
        normalization='NFKC/casefold/accent removal; small Spanish stopword list; hyphen compounds retained plus component terms; no stemming',
        tie_break='descending score, ascending stable chunk_id',seed=42,test_documents_used=0,
        chunks_sha256=sha256(DATA/'chunks.jsonl'),index_sha256=sha256(DATA/'indice_bm25.pkl'),
        tokenizer_sha256=sha256(BASE/'datos/tokenizer/tokenizer.json'),splits_sha256=sha256(BASE/'datos/splits.json'),
        suite_sha256=sha256(DATA/'suite_validation_5B.json'),integrity=verify_preservation())
    write_json(DATA/'metadata_indice.json',meta)
    write_json(RESULTS/'reporte_indice_rag_5B.json',meta)
    print('Chunks:',len(chunks),'Test:',0,flush=True)

if __name__=='__main__': main()
