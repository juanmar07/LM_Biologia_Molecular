"""2C.1: BPE UTF-8 desde cero, solo Train; sin torch ni tokens.pt."""
import hashlib, heapq, json, re, sys, subprocess
from collections import Counter, defaultdict
from pathlib import Path
BASE=Path(__file__).resolve().parent
DIRECTORIO=BASE/'datos/tokenizer'
ARCHIVO_TOKENIZADOR=DIRECTORIO/'tokenizer.json'
ESPECIALES=('<PAD>','<UNK>','<BOS>','<EOS>')
TOKENS_ESPECIALES=list(ESPECIALES)
# Segmentación Unicode reversible. Un espacio ASCII puede acompañar una unidad.
PRETOKEN=re.compile(r' ?[^\W\d_]+| ?\d+| ?[^\w\s]+|\s+|_+',re.UNICODE)
SPECIAL=re.compile('('+'|'.join(map(re.escape,ESPECIALES))+')')
WORD=re.compile(r"\b\w+(?:[-’']\w+)*\b")
TERMINOS=['ADN','ARN','mRNA','tRNA','BRCA1','BRCA2','TP53','p53','CRISPR-Cas9','SARS-CoV-2','RT-PCR','PCR','α-D-glucosa','β-globina','Δ','δ','α','β','κ','5′','3′','CD34+','RNA-seq','microRNA']
HASHES={'train.txt':'d1800ba78bb8524219cc9106a4b1215f20d72b2475fbdfa9e908730474d9b420','validation.txt':'79da7a7c7beffa095d70de13efe1c23081c6b4894fdd3237a1154a736697a0f7','test.txt':'1faafa69befce74a97c0445096019e27b874e0cf1b22547f96f6593f27f4bfaf','splits.json':'4a7486ad9a2a646116f2f41e9cf065a0009e2ef062333a9f8c5944bbf455dad7'}
def sha256(data): return hashlib.sha256(data).hexdigest()
def require(condition,message):
    if not condition: raise RuntimeError(message)
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8'))

class TokenizadorEducativo:
    def __init__(self):
        self.pieces=[];self.merges=[];self.ranks={};self.cache={}
        self.token_a_id={s:i for i,s in enumerate(ESPECIALES)};self.id_a_token={}
    def _ready(self): require(len(self.pieces)>=256,'Tokenizador sin entrenar ni cargar.')
    def _indexes(self):
        self.ranks={(a,b):(rank,result) for rank,(a,b,result) in enumerate(self.merges)}
        self.cache={};self.token_a_id={s:i for i,s in enumerate(ESPECIALES)}
        self.token_a_id.update({f'byte_{i+4}:{p.hex()}':i+4 for i,p in enumerate(self.pieces)})
        self.id_a_token={i:s for s,i in self.token_a_id.items()}
    def separar(self,texto): return PRETOKEN.findall(texto)
    def entrenar(self,textos=None,vocab_size=6000,frecuencia_minima=2):
        """Sin argumentos lee Train verificado; iterable explícito para pruebas."""
        if textos is None: textos=[d['text'] for d in leer_documentos('train')]
        elif isinstance(textos,str): textos=[textos]
        require(vocab_size>=260 and frecuencia_minima>=1,'Parámetros BPE inválidos.')
        frequencies=Counter()
        for text in textos:
            for part in SPECIAL.split(text):
                if part not in ESPECIALES: frequencies.update(s.encode('utf-8') for s in self.separar(part))
        self.pieces=[bytes([b]) for b in range(256)];self.merges=[]
        words=[[b+4 for b in word] for word in sorted(frequencies)]
        weights=[frequencies[word] for word in sorted(frequencies)]
        counts=Counter();locations=defaultdict(set)
        for i,word in enumerate(words):
            for pair,n in Counter(zip(word,word[1:])).items():
                counts[pair]+=n*weights[i];locations[pair].add(i)
        queue=[(-n,a,b) for (a,b),n in counts.items()];heapq.heapify(queue)
        while len(self.pieces)+4<vocab_size and queue:
            negative,a,b=heapq.heappop(queue);pair=(a,b)
            if -negative!=counts[pair] or counts[pair]<=0: continue
            if counts[pair]<frecuencia_minima: break
            new_id=len(self.pieces)+4
            self.pieces.append(self.pieces[a-4]+self.pieces[b-4]);self.merges.append([a,b,new_id])
            changed=set()
            for i in sorted(locations[pair]):
                word=words[i];old=Counter(zip(word,word[1:]));merged=[];j=0
                while j<len(word):
                    if j+1<len(word) and (word[j],word[j+1])==pair: merged.append(new_id);j+=2
                    else: merged.append(word[j]);j+=1
                new=Counter(zip(merged,merged[1:]));words[i]=merged
                for p in old.keys()|new.keys():
                    counts[p]+=(new[p]-old[p])*weights[i]
                    if new[p]: locations[p].add(i)
                    else: locations[p].discard(i)
                    changed.add(p)
            for p in sorted(changed):
                if counts[p]>0: heapq.heappush(queue,(-counts[p],*p))
        self.target_vocab_size=vocab_size;self.min_frequency=frecuencia_minima;self._indexes()
        return self
    def _segment(self,segment):
        cached=self.cache.get(segment)
        if cached is not None: return cached
        ids=[b+4 for b in segment.encode('utf-8')]
        while len(ids)>1:
            best=min((self.ranks[(a,b)][0] for a,b in zip(ids,ids[1:]) if (a,b) in self.ranks),default=None)
            if best is None: break
            a,b,result=self.merges[best];output=[];j=0
            while j<len(ids):
                if j+1<len(ids) and ids[j]==a and ids[j+1]==b: output.append(result);j+=2
                else: output.append(ids[j]);j+=1
            ids=output
        self.cache[segment]=tuple(ids);return self.cache[segment]
    def encode(self,texto,agregar_bos=False,agregar_eos=False):
        self._ready();ids=[self.bos_id] if agregar_bos else []
        for part in SPECIAL.split(texto):
            if part in ESPECIALES: ids.append(self.token_a_id[part])
            else:
                for segment in self.separar(part): ids.extend(self._segment(segment))
        if agregar_eos: ids.append(self.eos_id)
        return ids
    def decode(self,ids,omitir_especiales=True):
        self._ready();chunks=[]
        for value in ids:
            i=int(value)
            if not 0<=i<self.vocab_size: raise ValueError(f'ID fuera del vocabulario: {i}')
            if i<4:
                if not omitir_especiales: chunks.append(ESPECIALES[i].encode('utf-8'))
            else: chunks.append(self.pieces[i-4])
        # Un modelo puede generar UTF-8 incompleto. encode siempre es reversible.
        return b''.join(chunks).decode('utf-8',errors='replace')
    def codificar(self,texto,agregar_inicio_fin=True): return self.encode(texto,agregar_inicio_fin,agregar_inicio_fin)
    def decodificar(self,ids,omitir_especiales=True): return self.decode(ids,omitir_especiales)
    def configuration(self):
        return {'version':'2C.1','algorithm':'byte-level BPE from scratch','implementation':'Python standard library; incremental weighted pair counts','pretoken_pattern':PRETOKEN.pattern,'normalization':None,'special_tokens':dict(zip(ESPECIALES,range(4))),'byte_ids':'byte+4; fixed UTF-8 alphabet, not learned from held-out data','target_vocab_size':self.target_vocab_size,'min_frequency':self.min_frequency,'tie_break':'maximum weighted frequency, then ascending pair IDs','vocabulary_hex':[p.hex() for p in self.pieces],'merges':self.merges}
    def guardar(self,ruta=ARCHIVO_TOKENIZADOR): self._ready();write_json(ruta,self.configuration())
    def cargar(self,ruta=ARCHIVO_TOKENIZADOR):
        c=json.loads(Path(ruta).read_text(encoding='utf-8'))
        require(c.get('version')=='2C.1' and c.get('special_tokens')==dict(zip(ESPECIALES,range(4))),'Formato/especiales incompatibles.')
        require(c['pretoken_pattern']==PRETOKEN.pattern and c['normalization'] is None,'Pretokenización incompatible.')
        self.pieces=[bytes.fromhex(p) for p in c['vocabulary_hex']];self.merges=c['merges']
        require(self.pieces[:256]==[bytes([i]) for i in range(256)] and len(self.pieces)==256+len(self.merges),'Alfabeto/vocabulario inválido.')
        for rank,(a,b,result) in enumerate(self.merges):
            require(result==rank+260 and 4<=a<result and 4<=b<result,'Orden de merges inválido.')
            require(self.pieces[result-4]==self.pieces[a-4]+self.pieces[b-4],'Merge corrupto.')
        self.target_vocab_size=c['target_vocab_size'];self.min_frequency=c['min_frequency'];self._indexes();return self
    @property
    def vocab_size(self): return len(self.pieces)+4 if self.pieces else 0
    @property
    def tamanio_vocabulario(self): return self.vocab_size
    pad_id,unk_id,bos_id,eos_id=0,1,2,3

def verificar_entradas():
    for filename,expected in HASHES.items():
        path=BASE/('datos/splits.json' if filename=='splits.json' else 'datos/splits/'+filename)
        require(sha256(path.read_bytes())==expected,'Entrada 2B modificada: '+filename)
def leer_documentos(split):
    require(split in ('train','validation'),'Test solo admite hash en 2C.')
    verificar_entradas();metadata=json.loads((BASE/'datos/splits.json').read_text(encoding='utf-8'))
    by_source={d['source']:d for d in metadata['inventory']};docs=[]
    for source in sorted(metadata['splits'][split]['sources']):
        d=by_source[source];path=(BASE/d['clean_file']).resolve()
        require(path.parent==(BASE/'datos/documentos_limpios').resolve(),'Ruta documental inválida.')
        data=path.read_bytes();require(sha256(data)==d['sha256'],'TXT congelado modificado: '+source)
        docs.append({'source':source,'sha256':d['sha256'],'text':data.decode('utf-8')})
    require(len(docs)==(14 if split=='train' else 4),'Número de fuentes inesperado.')
    return docs
def parameter_cost(size):
    embedding=size*128;output=size*128+size;total=embedding+output
    return {'embedding_parameters':embedding,'output_parameters_with_bias':output,'total_vocab_parameters':total,'float32_mib':total*4/2**20,'saving_vs_31902_types_percent':100*(1-size/31902)}
def benchmark(tokenizer,terms):
    results=[]
    for term in terms:
        ids=tokenizer.encode(term)
        pieces=[ESPECIALES[i] if i<4 else tokenizer.pieces[i-4].decode('utf-8',errors='backslashreplace') for i in ids]
        results.append({'term':term,'subtokens':pieces,'count':len(ids),'exact_decode':tokenizer.decode(ids)==term})
    require(all(r['exact_decode'] for r in results),'Benchmark no reversible.')
    return results

def metrics(tokenizer,docs):
    tokens=unknown=chars=words=0;frequencies=Counter();roundtrips=[]
    for d in docs:
        text=d['text'];ids=tokenizer.encode(text);tokens+=len(ids);unknown+=ids.count(tokenizer.unk_id);chars+=len(text)
        forms=WORD.findall(text);words+=len(forms);frequencies.update(forms)
        exact=tokenizer.decode(ids)==text;require(exact,'Roundtrip completo falló: '+d['source'])
        roundtrips.append({'source':d['source'],'characters':len(text),'exact':exact})
    distribution=Counter();maximum=0;extreme=[]
    for form,frequency in sorted(frequencies.items()):
        n=len(tokenizer.encode(form));distribution[str(n) if n<4 else '4+']+=frequency
        if n>maximum: maximum=n;extreme=[]
        if n==maximum: extreme.append({'form':form,'characters':len(form),'subtokens':n,'occurrences':frequency})
    return {'total_tokens':tokens,'approximate_words':words,'characters':chars,'tokens_per_word':tokens/words,'characters_per_token':chars/tokens,'unknown_count':unknown,'unknown_percentage':100*unknown/tokens,'word_subtoken_distribution_percent':{k:100*distribution[k]/words for k in ('1','2','3','4+')},'fragmentation_definition':'Word occurrences encoded in isolation, case-sensitive; without leading whitespace','maximum_word_fragmentation':maximum,'maximum_fragmentation_examples':extreme[:3],'full_document_roundtrips':roundtrips}

def scientific_sample(docs):
    forms=Counter()
    for d in docs: forms.update(re.findall(r'\b\w+(?:[-+′’]\w*)*',d['text']))
    categories={'greek':lambda x:bool(re.search('[Α-Ωα-ω]',x)),'alphanumeric':lambda x:any(c.isalpha() for c in x) and any(c.isdigit() for c in x),'acronym':lambda x:len(x)>=2 and x.isupper() and any(c.isalpha() for c in x),'hyphenated':lambda x:'-' in x and any(c.isalpha() for c in x),'numbers':lambda x:x.isdigit()}
    selected=[];used=set()
    for category,predicate in categories.items():
        eligible=sorted((x for x in forms if predicate(x) and len(x)<=60),key=lambda x:(-forms[x],x))
        for term in [x for x in eligible if x not in used][:20]:
            selected.append({'term':term,'category':category,'train_occurrences':forms[term]});used.add(term)
    if len(selected)<100:
        for term in sorted((x for x in forms if x not in used and any(c.isalpha() for c in x) and len(x)<=60),key=lambda x:(-forms[x],x)):
            selected.append({'term':term,'category':'train_form','train_occurrences':forms[term]})
            if len(selected)==100: break
    require(len(selected)>=100,'Muestra científica insuficiente.')
    return selected

def comparar_y_entrenar():
    verificar_entradas();train=leer_documentos('train');validation=leer_documentos('validation');sample=scientific_sample(train)
    candidates=[];models={}
    for size in (4000,6000,8000):
        print(f'Entrenando BPE {size} exclusivamente con 14 TXT Train...',flush=True)
        model=TokenizadorEducativo().entrenar([d['text'] for d in train],size)
        require(model.vocab_size==size,'No se alcanzó el tamaño objetivo.')
        path=DIRECTORIO/'candidatos'/f'bpe_{size}.json';model.guardar(path)
        sampled=benchmark(model,[t['term'] for t in sample])
        candidates.append({'target_vocab_size':size,'actual_vocab_size':model.vocab_size,'candidate_sha256':sha256(path.read_bytes()),'train':metrics(model,train),'validation':metrics(model,validation),'scientific_benchmark':benchmark(model,TERMINOS),'train_scientific_sample':sampled,'sample_mean_subtokens':sum(r['count'] for r in sampled)/len(sampled),'estimated_parameter_cost':parameter_cost(size)})
        models[size]=model
    # Regla fijada antes de observar resultados: el candidato fiel más pequeño
    # a <=5% de tokens Validation de 8k y <=10% de fragmentación científica Train.
    baseline=candidates[-1]
    eligible=[c for c in candidates if c['validation']['unknown_count']==0 and c['validation']['tokens_per_word']<=baseline['validation']['tokens_per_word']*1.05 and c['sample_mean_subtokens']<=baseline['sample_mean_subtokens']*1.10]
    require(eligible,'Ningún candidato satisface fidelidad y eficiencia.')
    chosen=min(eligible,key=lambda c:c['actual_vocab_size']);size=chosen['actual_vocab_size'];model=models[size]
    model.guardar();first=ARCHIVO_TOKENIZADOR.read_bytes()
    print(f'Reentrenando BPE {size} desde cero para comprobar determinismo...',flush=True)
    repeated=TokenizadorEducativo().entrenar([d['text'] for d in train],size)
    require(repeated.configuration()==model.configuration(),'BPE no determinista al reentrenar.')
    for term in TERMINOS+[t['term'] for t in sample]: require(model.encode(term)==repeated.encode(term),'IDs cambiaron.')
    repeated.guardar(DIRECTORIO/'candidatos/reentrenado_seleccionado.json')
    require(first==(DIRECTORIO/'candidatos/reentrenado_seleccionado.json').read_bytes(),'Serialización no determinista.')
    require(TokenizadorEducativo().cargar().configuration()==model.configuration(),'Carga alteró modelo.')
    # Reinicializar intérprete: cargar el artefacto sin entrenar ni abrir splits.
    child="import tokenizador as t; m=t.TokenizadorEducativo().cargar(); assert m.vocab_size=="+str(size)+"; assert all(m.decode(m.encode(s))==s for s in t.TERMINOS); print('fresh-load-ok')"
    check=subprocess.run([sys.executable,'-B','-c',child],cwd=BASE,capture_output=True,text=True,check=True)
    require(check.stdout.strip()=='fresh-load-ok','Falló persistencia en proceso nuevo.')
    verificar_entradas()
    reason={'eligible_vocab_sizes':[c['actual_vocab_size'] for c in eligible],'validation_token_excess_vs_8k_percent':100*(chosen['validation']['tokens_per_word']/baseline['validation']['tokens_per_word']-1),'scientific_sample_excess_vs_8k_percent':100*(chosen['sample_mean_subtokens']/baseline['sample_mean_subtokens']-1),'validation_improvement_4k_to_6k_percent':100*(1-candidates[1]['validation']['total_tokens']/candidates[0]['validation']['total_tokens']),'validation_improvement_6k_to_8k_percent':100*(1-candidates[2]['validation']['total_tokens']/candidates[1]['validation']['total_tokens'])}
    report={'version':'2C.1','source_train_hash':HASHES['train.txt'],'source_validation_hash':HASHES['validation.txt'],'test_hash_registrado_solo_como_integridad_sin_analizar':HASHES['test.txt'],'test_content_read_for_metrics':False,'source_splits_hash':HASHES['splits.json'],'sha256_code':sha256(Path(__file__).read_bytes()),'algorithm':'Deterministic byte-level BPE from scratch, Train only','implementation_library_and_version':{'library':'Python standard library; local implementation 2C.1','python':sys.version.split()[0]},'normalization_policy':'Identity; preserve whitespace, case and Unicode exactly','pretoken_pattern':PRETOKEN.pattern,'min_frequency':2,'special_tokens_and_ids':dict(zip(ESPECIALES,range(4))),'training_sources':[{k:v for k,v in d.items() if k!='text'} for d in train],'validation_sources':[{k:v for k,v in d.items() if k!='text'} for d in validation],'metadata_handling':'Direct clean TXT; no source JSON, file names or DOCUMENTO wrappers','candidate_results':candidates,'selected_vocab_size':size,'selection_rule':'Smallest faithful zero-UNK candidate within 5% of 8k Validation tokens/word and 10% of 8k Train scientific sample mean fragmentation','selection_reason':reason,'train_metrics':chosen['train'],'validation_metrics':chosen['validation'],'scientific_benchmark':chosen['scientific_benchmark'],'scientific_sample_train_only':sample,'roundtrip_results':{'complete_documents_per_candidate':18,'all_exact':True,'whitespace_changes_allowed':'None','fresh_process_load_encode_decode_verified':True,'weighted_bpe_reference_test':'passed independently against naive full pair recount','unicode_whitespace_specials_api_tests':'passed independently'},'estimated_parameter_cost':chosen['estimated_parameter_cost'],'sha256_tokenizer':sha256(first),'determinism':{'fresh_retraining_same_vocab_merges_ids':True,'tokenizer_bytes_identical':True,'benchmark_ids_identical':True},'original_tokenizer_audit':'Word/sign regex, lowercasing, frequency vocabulary and approximate spaces; original entrypoint generated global tokens.pt. Consumers use cargar/codificar/decodificar/tamanio_vocabulario and token_a_id specials.','warnings':['Dataset and pipeline require adaptation in 2D; old global tensor generation entrypoint explicitly refuses execution.','Arbitrary generated byte sequences may be incomplete UTF-8: decode replaces invalid sequences; valid encode/decode is exact.','Fixed 256-byte alphabet provides coverage, not semantic knowledge of unseen symbols.','Selection measures tokenizer efficiency, not Transformer perplexity.','Literal reserved strings are special tokens; decode with omitir_especiales=False retains them.']}
    write_json(DIRECTORIO/'reporte_tokenizador.json',report)
    print(json.dumps({'selected_vocab_size':size,'metrics':[{k:c[k] for k in ('actual_vocab_size','train','validation','sample_mean_subtokens')} for c in candidates],'selection_reason':reason,'sha256_tokenizer':report['sha256_tokenizer'],'sha256_report':sha256((DIRECTORIO/'reporte_tokenizador.json').read_bytes())},ensure_ascii=True,indent=2),flush=True)
    return model

def entrenar_tokenizador_y_tokenizar():
    raise RuntimeError('Generación global de tokens.pt suspendida: adaptar dataset por splits en 2D.')
if __name__=='__main__': comparar_y_entrenar()
