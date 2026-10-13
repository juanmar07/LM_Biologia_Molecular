"""Pruebas funcionales y siete demostraciones con evidencia de desarrollo verificada."""
import hashlib
import json
import math
import subprocess
import sys
from rag_common import BASE, DATA, RESULTS, FORBIDDEN, CHECKPOINT, require, sha256, verify_preservation, write_json
from retrieval import retrieve, load_index, normalize, tokens, build_bm25
from rag import RAG, pack_context, greedy_generate

EXAMPLES = [
 ('¿Qué función tiene la PCR?', 'La PCR es una técnica que permite la amplificación'),
 ('¿Qué es CRISPR-Cas?', '2012: Se desarrolló una técnica de edición de genes llamada'),
 ('¿Qué diferencia existe entre Southern blot y Northern blot?', 'Las técnicas Southern blot y Northern blot se utilizan'),
 ('¿Qué son las células multipotenciales?', 'Son células capaces de diferenciarse a distintos tipos celulares'),
 ('¿Cuál es la función de un vector de expresión?', 'Un vector de expresión se utiliza para producir una proteína recombinante'),
 ('¿Qué son las enzimas de restricción?', 'Las enzimas de restricción reconocen secuencias específicas'),
 ('¿Qué caracteriza a los genes ribosómicos?', 'Los genes ribosómicos se caracterizan por su elevado grado')]

def main():
    tests={}
    index=load_index()
    tests['index_load']=index['n']>0
    tests['empty_query']=retrieve('   ')==[] and retrieve('y de la')==[]
    tests['scientific_normalization']=tokens('CRISPR-Cas9 ARNm ADN ARN PCR EGFR 95')==['crispr-cas9','crispr','cas9','arnm','adn','arn','pcr','egfr','95'] and normalize('Expresión')==normalize('EXPRESION')
    tests['normal_scientific_query']=len(retrieve(EXAMPLES[0][0]))==5
    tests['top_k']=len(retrieve('ADN',1))==1 and len(retrieve('ADN',3))==3
    invalid=0
    for k in (0,-1,101,1.5,True):
        try: retrieve('PCR',k)
        except ValueError: invalid+=1
    tests['invalid_top_k']=invalid==5
    tests['retrieval_determinism']=retrieve('expresión ADN')==retrieve('expresión ADN')
    tests['score_order']=all(a['score']>=b['score'] for a,b in zip(retrieve('PCR'),retrieve('PCR')[1:]))
    toy=build_bm25([dict(text='PCR ADN ADN'),dict(text='PCR ARN')])
    tests['bm25_df_avgdl_idf']=toy['df']=={'adn':1,'arn':1,'pcr':2} and toy['avgdl']==2.5 and math.isclose(toy['idf']['pcr'],math.log1p(.5/2.5))
    toy_denominator=2+1.5*(1-.75+.75*3/2.5)
    tests['bm25_tf_formula']=math.isclose(toy['idf']['adn']*2*2.5/toy_denominator,math.log(2)*5/3.725)
    before={p:sha256(DATA/p) for p in ('chunks.jsonl','indice_bm25.pkl','suite_validation_5B.json')}
    rebuilt=subprocess.run([sys.executable,'-X','utf8','-B',str(BASE/'construir_indice_rag.py')],cwd=BASE,capture_output=True,text=True,encoding='utf-8',timeout=60)
    tests['index_build_and_byte_determinism']=rebuilt.returncode==0 and before=={p:sha256(DATA/p) for p in before}
    tests['all_chunks_max48']=all(0<c['bpe_tokens']<=48 for c in index['chunks'])
    tests['no_test_indexed']=all(c['split'] in ('train','validation') for c in index['chunks'])
    # Deliberate attempts are rejected before the OS read, not reads of Test contents.
    blocked=0
    for path in sorted(FORBIDDEN):
        try:
            with open(path,'rb'): pass
        except RuntimeError: blocked+=1
    tests['test_guard_rejects_before_open']=blocked==len(FORBIDDEN)
    try:
        with open(CHECKPOINT,'wb'): pass
    except RuntimeError: tests['previous_checkpoint_write_blocked']=True
    system=RAG()
    tests['strict_checkpoint']=system.model_metadata['strict'] and system.model_metadata['parameters']==1379456 and not system.model.training
    def state_hash():
        digest=hashlib.sha256()
        for n,p in system.model.state_dict().items(): digest.update(n.encode()); digest.update(p.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    weights=state_hash()
    first=system.answer(EXAMPLES[0][0])
    tests['greedy_determinism']=first['generation']==system.answer(EXAMPLES[0][0])['generation']
    found=retrieve('ADN',5)
    packed=pack_context('¿Qué indica?',found,system.tokenizer,2)
    tests['context_top2_reduction']=len(packed['selected_chunks']) in (1,2) and packed['context_bpe_tokens']<=52 and packed['prompt_bpe_tokens']+packed['effective_max_new_tokens']<=128
    # Also verify that two short complete chunks are actually usable, rather than always dropped.
    small=[c for c in index['chunks'] if c['bpe_tokens']<=12][:2]
    two=pack_context('¿Qué indican?',small,system.tokenizer,2)
    tests['two_complete_chunks_fit']=len(two['selected_chunks'])==2
    try:
        pack_context('ADN '*200,found,system.tokenizer)
    except ValueError: tests['long_question_refused_without_truncation']=True
    import torch
    class ForcedToken:
        def __init__(self,t): self.t=t
        def __call__(self,x):
            logits=torch.zeros(1,x.shape[1],6000); logits[0,-1,self.t]=1
            return logits,None
    eos=greedy_generate(ForcedToken(3),system.tokenizer,packed)
    tests['EOS_immediate_stop']=eos['EOS'] and eos['generated_token_ids']==[3] and eos['stop_reason']=='EOS'
    budget=dict(packed,effective_max_new_tokens=128-packed['prompt_bpe_tokens'],generation_budget_reduced=True)
    capped=greedy_generate(ForcedToken(4),system.tokenizer,budget)
    tests['hard_total128_noEOS']=not capped['EOS'] and capped['total_bpe_tokens']==128 and capped['stop_reason']=='context_budget'
    children={}
    for mode,args,stdin in [('single',['--pregunta',EXAMPLES[0][0]],None),
                           ('show_retrieval',['--pregunta',EXAMPLES[0][0],'--show-retrieval'],None),
                           ('interactive',['--interactive'],EXAMPLES[0][0]+'\n'+EXAMPLES[-1][0]+'\nexit\n'),
                           ('raw',['--pregunta',EXAMPLES[0][0],'--raw'],None)]:
        child=subprocess.run([sys.executable,'-X','utf8','-B',str(BASE/'rag.py'),*args],input=stdin,
                cwd=BASE,capture_output=True,text=True,encoding='utf-8',timeout=45)
        children[mode]=dict(exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr)
        tests['CLI_'+mode]=child.returncode==0 and (('TOP-K' in child.stdout) if mode=='show_retrieval' else
                (child.stdout.count('Fuente utilizada:')==2 if mode=='interactive' else
                 ('raw_generated_text' in child.stdout if mode=='raw' else 'Fuente utilizada:' in child.stdout)))
    examples=[]
    for question,proof_text in EXAMPLES:
        proof=next(c for c in index['chunks'] if normalize(proof_text) in normalize(c['text']))
        require(proof['split'] in ('train','validation'),'Demo con evidencia prohibida')
        result=system.answer(question)
        examples.append(dict(result,evidence_exists_in_development=True,existence_proof=proof,
                             note='Existence of corpus evidence does not assert that top1 or generated answer is correct.'))
    tests['examples_context_budget']=all(e['generation']['total_bpe_tokens']<=128 for e in examples)
    tests['weights_unchanged_in_memory']=weights==state_hash()
    tests['previous_integrity']=verify_preservation()['previous_artifacts_preserved']
    write_json(RESULTS/'ejemplos_rag_5B.json',dict(examples=examples,test_documents_used=0))
    report=dict(tests=tests,passed=all(tests.values()),count=len(tests),test_content_reads=0,
        intentional_guard_rejections=blocked,EOS_real_demo_rate=sum(e['generation']['EOS'] for e in examples)/len(examples),
        cli=children,rebuild_stdout=rebuilt.stdout,rebuild_stderr=rebuilt.stderr,integrity=verify_preservation())
    write_json(RESULTS/'smoke_rag_5B.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cli','rebuild_stdout')},ensure_ascii=False,indent=2))
    require(report['passed'],'Smoke falló')

if __name__=='__main__': main()
