"""CLI RAG 5B.1: serialización SFT, reranking local y contexto vecino, sin gate 5C."""
import argparse
import json
from rag_5B1_common import BASE, load_runtime, require, sft_prompt, legacy_functions
from retrieval import retrieve, signals, tokens
from rerank_rag_5B1 import rerank
from contexto_rag_5B1 import pack_neighbors

def prompt(context, question):
    return sft_prompt(context,question)

def pack_context(question, found, tokenizer, context_top_n=1, max_new_tokens=48):
    if context_top_n not in (1,2):
        raise ValueError('context_top_n debe ser 1 o 2')
    selected = found[:context_top_n]
    reductions = []
    while selected:
        context = '\n\n'.join(c['text'] for c in selected)
        ids = [2]+tokenizer.encode(prompt(context,question))
        if len(tokenizer.encode(context))<=52 and len(ids)<128:
            require(all(t>=4 for t in ids[1:]),'Pregunta/contexto contiene tokens especiales reservados')
            capacity = min(max_new_tokens,128-len(ids))
            return dict(context=context,selected_chunks=selected,prompt=prompt(context,question),prompt_ids=ids,
                        context_bpe_tokens=len(tokenizer.encode(context)),prompt_bpe_tokens=len(ids),
                        requested_max_new_tokens=max_new_tokens,effective_max_new_tokens=capacity,
                        generation_budget_reduced=capacity<max_new_tokens,reductions=reductions,
                        question_truncated=False,context_truncated=False)
        reductions.append(dict(removed_chunk=selected[-1]['chunk_id'],reason='context/prompt budget'))
        selected.pop()
    raise ValueError('No hay chunk completo utilizable dentro de 128 tokens; error mecánico, sin gate de suficiencia')

def greedy_generate(model, tokenizer, packed):
    import torch
    ids=list(packed['prompt_ids'])
    generated=[]
    eos=False
    with torch.no_grad():
        for _ in range(packed['effective_max_new_tokens']):
            require(len(ids)<128, 'Presupuesto agotado antes de generar')
            logits,_=model(torch.tensor([ids],dtype=torch.long))
            token=int(logits[0,-1].argmax().item())
            generated.append(token); ids.append(token)
            if token==3:
                eos=True; break
    require(len(ids)<=128,'Secuencia excedió 128 tokens')
    payload=generated[:-1] if eos else generated
    return dict(answer=tokenizer.decode(payload,omitir_especiales=False),raw_generated_text=tokenizer.decode(generated,omitir_especiales=False),
                generated_token_ids=generated,EOS=eos,stop_reason='EOS' if eos else ('context_budget' if packed['generation_budget_reduced'] else 'max_new_tokens'),
                generated_bpe_tokens=len(generated),total_bpe_tokens=len(ids))

class RAG:
    def __init__(self,checkpoint=None):
        self.model,self.tokenizer,self.model_metadata=load_runtime(checkpoint)

    def answer(self,question,top_k=10,context_top_n=1,variant='D'):
        require(variant in ('A','B','C','D'),'Variante inválida')
        found=retrieve(question,top_k)
        if not found:
            raise ValueError('Consulta vacía o sin términos presentes en el índice; no se invoca el modelo')
        ranked=rerank(question,found) if variant in ('C','D') else found
        if variant=='D':
            packed=pack_neighbors(question,ranked,self.tokenizer,context_top_n)
        else:
            legacy=legacy_functions('original' if variant=='A' else 'sft')
            packed=legacy['pack_context'](question,ranked,self.tokenizer,context_top_n)
        generated=greedy_generate(self.model,self.tokenizer,packed)
        wordlist=tokens(generated['answer'],False)
        grams=[tuple(wordlist[i:i+4]) for i in range(len(wordlist)-3)]
        generated['repetition']=len(grams)!=len(set(grams))
        return dict(question=question,retrieved=ranked,bm25_retrieved=found,variant=variant,
                    signals_5C=signals(question,found),
                    packing={k:v for k,v in packed.items() if k!='prompt_ids'},generation=generated,
                    model=self.model_metadata,prompt_format='5B legacy Pregunta/Respuesta' if variant=='A' else 'exact SFT Contexto/Usuario/Asistente + BOS')

def show(result,debug=False,raw=False):
    if raw:
        print(json.dumps(result,ensure_ascii=False,indent=2)); return
    print('Pregunta:',result['question'])
    print('Respuesta:',result['generation']['answer'])
    for c in result['packing']['selected_chunks']:
        print('Fuente utilizada:',c['source_doc'])
        print('Chunk recuperado:',c['chunk_id'])
        print(c['text'])
        score=c.get('bm25_score',c.get('score'))
        print('BM25 score:',round(score,6) if score is not None else 'contexto vecino')
        if 'rerank_score' in c: print('Rerank score:',round(c['rerank_score'],6))
    print('EOS:',result['generation']['EOS'],'/ stop:',result['generation']['stop_reason'])
    print('Repetición:',result['generation']['repetition'])
    if result['packing']['generation_budget_reduced']:
        print('Presupuesto de generación:',result['packing']['effective_max_new_tokens'],'tokens (límite total 128).')
    if debug:
        print('TOP-K + metadata completa:')
        for c in result['retrieved']:
            used=c['chunk_id'] in {u['chunk_id'] for u in result['packing']['selected_chunks']}
            print(json.dumps(dict(c,finally_used=used),ensure_ascii=False,indent=2))
        print('Packing:',json.dumps(result['packing'],ensure_ascii=False,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--pregunta')
    modes.add_argument('--interactive',action='store_true')
    parser.add_argument('--top-k',type=int,choices=(10,20),default=10)
    parser.add_argument('--variant',choices=('A','B','C','D'),default='D')
    parser.add_argument('--context-top-n',type=int,choices=(1,2),default=1)
    parser.add_argument('--show-retrieval',action='store_true')
    parser.add_argument('--raw',action='store_true')
    parser.add_argument('--checkpoint',type=lambda p:str((BASE/p).resolve()))
    args=parser.parse_args()
    system=RAG(args.checkpoint)
    def run(question):
        try:
            show(system.answer(question,args.top_k,args.context_top_n,args.variant),args.show_retrieval,args.raw)
            return True
        except (ValueError,RuntimeError) as exc:
            print('Error:',str(exc)); return False
    if args.pregunta is not None:
        if not run(args.pregunta): parser.exit(2)
    else:
        while True:
            try: question=input('Pregunta (salir/exit): ')
            except (EOFError,KeyboardInterrupt): break
            if question.strip().casefold() in ('salir','exit'): break
            run(question)

if __name__=='__main__': main()
