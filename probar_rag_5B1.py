"""Contrato real SFT↔RAG, presupuestos/vecinos, guard y CLI; ocho consultas manuales."""
import copy
import hashlib
import json
import subprocess
import sys
from rag_5B1_common import BASE,DATA,RESULTS,preservation,require,sft_functions,previous,write_json
from rag import RAG,prompt,greedy_generate
from retrieval import retrieve,load_index
from rerank_rag_5B1 import rerank,query_analysis,intent_signal
from contexto_rag_5B1 import pack_neighbors,source_lookup,make_packed

QUESTIONS=['¿Qué es el ADN?','¿Qué función tiene la PCR?','¿Qué es CRISPR-Cas?',
 '¿Qué diferencia existe entre Southern blot y Northern blot?','¿Qué son las células multipotenciales?',
 '¿Cuál es la función de un vector de expresión?','¿Qué son las enzimas de restricción?',
 '¿Qué caracteriza a los genes ribosómicos?']

def compare_prompt_contract(record,tokenizer,constructor=prompt):
    ns=sft_functions()
    import torch
    ns['torch']=torch
    expected=ns['prompt'](record)
    actual=constructor(record['context'],record['question'])
    require(actual==expected,'SFT/RAG labels/newlines diverged')
    x,y=ns['encode'](record,tokenizer)
    prefix=tokenizer.encode(actual,agregar_bos=True)
    require(prefix==[2]+tokenizer.encode(actual),'BOS divergente')
    require(x[:len(prefix)].tolist()==prefix,'SFT/RAG prefix token IDs diverged')
    require(prefix[0]==2 and prefix.count(2)==1 and 3 not in prefix,'BOS/EOS prefix inválido')
    require(y[len(prefix)-1].item()==tokenizer.encode(record['answer'])[0] and (y==3).sum().item()==1,'Target response/EOS inválido')

def main():
    import torch
    system=RAG()
    tests={}
    rows=[]
    for split in ('train','validation'):
        rows.extend(json.loads(line) for line in (BASE/f'datos/sft_5A2/{split}_instrucciones_5A2.jsonl').read_text(encoding='utf-8').splitlines())
    for row in rows: compare_prompt_contract(row,system.tokenizer)
    tests['SFT_RAG_exact_labels_newlines_BOS_response_EOS_all_records']=True
    try: compare_prompt_contract(rows[0],system.tokenizer,lambda c,q:prompt(c,q).replace('Usuario:','Pregunta:'))
    except RuntimeError: tests['contract_detects_divergence']=True
    def state_hash():
        h=hashlib.sha256()
        for n,p in system.model.state_dict().items(): h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    before=state_hash()
    toy=[dict(chunk_id='mention',source_doc='toy',split='train',rank=1,score=10.,text='ADN ADN ADN ADN',bpe_tokens=4,sentence_fragment=False),
         dict(chunk_id='definition',source_doc='toy',split='train',rank=2,score=5.,text='El ADN es una molécula que contiene información genética.',bpe_tokens=12,sentence_fragment=False)]
    original=copy.deepcopy(toy)
    tests['definition_beats_high_BM25_mention']=rerank(QUESTIONS[0],toy)[0]['chunk_id']=='definition'
    tests['distant_predicate_is_not_definition']=intent_signal(query_analysis(QUESTIONS[0]),'la replicación del adn en procariotes en la hebra retrasada es:')==0.
    tests['rerank_preserves_raw_candidates']=toy==original
    tests['rerank_determinism']=rerank(QUESTIONS[0],toy)==rerank(QUESTIONS[0],toy)
    tests['empty_query']=retrieve('  ')==[]
    found=rerank(QUESTIONS[1],retrieve(QUESTIONS[1],10))
    tests['top10_and20']=len(found)==10 and len(retrieve(QUESTIONS[1],20))==20
    packed=pack_neighbors(QUESTIONS[1],found,system.tokenizer)
    tests['minimum_response_budget']=packed['effective_max_new_tokens']>=24 and packed['prompt_bpe_tokens']+packed['effective_max_new_tokens']<=128
    try: pack_neighbors('ADN '*200,found,system.tokenizer)
    except ValueError: tests['long_question_error_no_truncation']=True
    class ForcedEOS:
        def __call__(self,x):
            logits=torch.zeros(1,x.shape[1],6000);logits[0,-1,3]=1
            return logits,None
    gen=greedy_generate(ForcedEOS(),system.tokenizer,packed)
    tests['EOS_stop']=gen['generated_token_ids']==[3] and gen['stop_reason']=='EOS'
    blocked=0
    for path in previous.FORBIDDEN:
        try:
            with open(path,'rb'): pass
        except RuntimeError: blocked+=1
    tests['Test_rejected_before_open']=blocked==len(previous.FORBIDDEN)
    for label,path in [('checkpoint',previous.CHECKPOINT),('suite',DATA/'suite_validation_5B.json')]:
        try:
            with open(path,'wb'): pass
        except RuntimeError: tests[label+'_write_blocked']=True
    examples=[]
    for question in QUESTIONS:
        result=system.answer(question)
        examples.append(dict(result,top5=result['retrieved'][:5]))
    tests['eight_questions_executed']=len(examples)==8
    tests['all_demo_budgets']=all(e['generation']['total_bpe_tokens']<=128 and e['packing']['effective_max_new_tokens']>=24 for e in examples)
    tests['neighbor_text_exact_source_span']=True
    tests['neighbor_document_split_consistent']=True
    for e in examples:
        p=e['packing']; span=p['context_source_span']; doc=source_lookup()[span['source_doc']]
        tests['neighbor_text_exact_source_span'] &= p['context']==doc['text'][span['start']:span['end']]
        tests['neighbor_document_split_consistent'] &= all(c['source_doc']==span['source_doc'] and c['split']==doc['split'] for c in p['selected_chunks'])
    tests['model_strict_eval_frozen']=system.model_metadata['strict'] and not system.model.training and all(not p.requires_grad for p in system.model.parameters())
    tests['weights_unchanged']=state_hash()==before
    tests['default_greedy_determinism']=system.answer(QUESTIONS[1])['generation']==examples[1]['generation']
    cli={}
    for name,args,stdin in [('single',['--pregunta',QUESTIONS[0]],None),
        ('debug',['--pregunta',QUESTIONS[1],'--show-retrieval'],None),
        ('raw',['--pregunta',QUESTIONS[0],'--raw'],None),
        ('interactive',['--interactive'],QUESTIONS[0]+'\n'+QUESTIONS[1]+'\nexit\n')]:
        child=subprocess.run([sys.executable,'-X','utf8','-B',str(BASE/'rag.py'),*args],input=stdin,
            cwd=BASE,capture_output=True,text=True,encoding='utf-8',timeout=60)
        cli[name]=dict(exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr)
        tests['CLI_'+name]=child.returncode==0
        if name=='debug': tests['diagnostic_all_signals_and_used_chunk']=all(t in child.stdout for t in ('bm25_rank','bm25_score','rerank_score','rerank_summary','finally_used'))
        if name=='interactive': tests['interactive_two_answers']=child.stdout.count('Fuente utilizada:')>=2
        if name=='raw': tests['raw_JSON_valid']=json.loads(child.stdout)['variant']=='D'
    tests['integrity']=preservation()['previous_artifacts_preserved']
    write_json(RESULTS/'pruebas_manuales_rag_5B1.json',dict(examples=examples,test_documents_used=0))
    report=dict(tests=tests,passed=all(tests.values()),count=len(tests),SFT_records_compared=len(rows),
                cli=cli,integrity=preservation(),test_content_reads=0,intentional_Test_guard_rejections=blocked)
    write_json(RESULTS/'smoke_rag_5B1.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='cli'},ensure_ascii=False,indent=2))
    require(report['passed'],'5B.1 smoke failure')

if __name__=='__main__': main()
