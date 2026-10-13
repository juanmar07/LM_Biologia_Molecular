"""Contexto vecino exacto por offsets, con pregunta íntegra y reserva de 24–32 tokens."""
import re
from functools import lru_cache
from rag_5B1_common import require,sft_prompt
from rag_common import documents
from retrieval import load_index

def make_packed(context,question,chunks,tokenizer,min_response=32):
    ids=[2]+tokenizer.encode(sft_prompt(context,question))
    if len(ids)+min_response>128: return None
    require(all(t>=4 for t in ids[1:]),'Tokens especiales literales en pregunta/contexto')
    capacity=min(48,128-len(ids))
    return dict(context=context,selected_chunks=chunks,prompt=sft_prompt(context,question),prompt_ids=ids,
        context_bpe_tokens=len(tokenizer.encode(context)),prompt_bpe_tokens=len(ids),requested_max_new_tokens=48,
        effective_max_new_tokens=capacity,generation_budget_reduced=capacity<48,
        minimum_response_reserve=min_response,question_truncated=False,context_truncated=False,reductions=[])

@lru_cache(maxsize=1)
def source_lookup():
    return {d['source_doc']:d for d in documents()}

@lru_cache(maxsize=1)
def neighbor_lookup():
    return {(c['source_doc'],c['chunk_index']):c for c in load_index()['chunks']}

def closure(text):
    return bool(re.search(r'[.!?]["»”\)]*\s*$',text))

def completeness(text,source,start,end):
    # Structural proxy only: a boundary, terminal punctuation and non-dangling last word.
    before=source[:start].rstrip()
    left=(not before or before[-1] in '.!?' or '\n\n' in source[max(0,start-3):start])
    right=closure(text) and not re.search(r'\b(?:de|del|en|para|por|y|o|que|con|una|un|el|la|los|las)\s*$',text,re.I)
    return dict(start_boundary=left,end_boundary=right,structurally_complete=left and right)

def pack_neighbors(question,ranked,tokenizer,context_top_n=1):
    require(context_top_n in (1,2),'context_top_n inválido')
    for candidate in ranked:
        neighbors=neighbor_lookup()
        prev=neighbors.get((candidate['source_doc'],candidate['chunk_index']-1))
        following=neighbors.get((candidate['source_doc'],candidate['chunk_index']+1))
        source=source_lookup()[candidate['source_doc']]['text']
        options=[[candidate]]
        if candidate['sentence_fragment'] or not closure(candidate['text']):
            options += [list(filter(None,x)) for x in ([prev,candidate],[candidate,following],[prev,candidate,following])]
        variants=[]
        for group in options:
            group=sorted(group,key=lambda c:c['char_start_if_available'])
            contiguous=all(not source[a['char_end_if_available']:b['char_start_if_available']].strip()
                for a,b in zip(group,group[1:]) if a['char_end_if_available']<b['char_start_if_available'])
            if not contiguous: continue  # Never reinsert text rejected between neighboring chunks.
            start=min(c['char_start_if_available'] for c in group)
            end=max(c['char_end_if_available'] for c in group)
            value=source[start:end]
            require(all(source[c['char_start_if_available']:c['char_end_if_available']]==c['text'] for c in group),'Offsets no coinciden')
            quality=completeness(value,source,start,end)
            for reserve in (32,24):
                packed=make_packed(value,question,group,tokenizer,reserve)
                if packed:
                    packed.update(neighbor_expansion=len(group)>1,anchor_chunk_id=candidate['chunk_id'],
                        anchor_rerank_rank=candidate['rank'],context_source_span=dict(source_doc=candidate['source_doc'],start=start,end=end),
                        completeness=quality,neighbors_attempted=len(options)>1,
                        overlap_merge='exact source span; overlapping words not duplicated')
                    variants.append(packed)
                    break
        if variants:
            # Complete semantics boundary takes priority, then reserve32, then minimal source span.
            variants.sort(key=lambda p:(-int(p['completeness']['structurally_complete']),
                -int(p['completeness']['end_boundary']),-p['minimum_response_reserve'],p['context_bpe_tokens']))
            chosen=variants[0]
            if context_top_n==2:
                extra=next((c for c in ranked if c['chunk_id'] not in {n['chunk_id'] for n in chosen['selected_chunks']}),None)
                if extra:
                    merged=make_packed(chosen['context']+'\n\n'+extra['text'],question,chosen['selected_chunks']+[extra],tokenizer,32)
                    if merged:
                        chosen.update({k:v for k,v in merged.items() if k not in ('reductions',)})
                        chosen['additional_diagnostic_context']=True
            return chosen
    raise ValueError('Ningún contexto completo cabe con pregunta íntegra y al menos 24 tokens para responder; error de presupuesto, no gate 5C')
