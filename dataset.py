"""2D.1: document-local causal datasets, compact tensors and window indices.
Run python -B dataset.py to analyze, freeze settings, build and verify .pt files.
This module does not import the model or perform training.
"""
import hashlib
import io
import json
import math
import re
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from config import (LONGITUD_CONTEXTO, STRIDE_ENTRENAMIENTO, STRIDE_EVALUACION,
                    IGNORE_INDEX_DATASET, CARPETA_DATASET, TAMANIO_BATCH, SEMILLA)
from tokenizador import TokenizadorEducativo

BASE = Path(__file__).resolve().parent
VERSION = '2D.1'
TOKENIZER_HASH = '97953e99660fdebcf071b8327cd79f9cfa62e2ba38eae85d69f9f0e718d55b7e'
SPLIT_HASHES = {'train.txt':'d1800ba78bb8524219cc9106a4b1215f20d72b2475fbdfa9e908730474d9b420',
 'validation.txt':'79da7a7c7beffa095d70de13efe1c23081c6b4894fdd3237a1154a736697a0f7',
 'test.txt':'1faafa69befce74a97c0445096019e27b874e0cf1b22547f96f6593f27f4bfaf',
 'splits.json':'4a7486ad9a2a646116f2f41e9cf065a0009e2ef062333a9f8c5944bbf455dad7'}

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def verify_inputs():
    for name, expected in SPLIT_HASHES.items():
        p = BASE / ('datos/splits.json' if name == 'splits.json' else 'datos/splits/' + name)
        require(sha256(p.read_bytes()) == expected, 'Frozen input changed: ' + name)
    require(sha256((BASE/'datos/tokenizer/tokenizer.json').read_bytes()) == TOKENIZER_HASH, 'Tokenizer changed.')

def encode_documents(split, tokenizer, test_allowed=False):
    require(split in ('train', 'validation', 'test'), 'Unknown split.')
    require(split != 'test' or test_allowed, 'Freeze configuration before reading Test.')
    meta = json.loads((BASE/'datos/splits.json').read_text(encoding='utf-8'))
    inventory = {d['source']: d for d in meta['inventory']}
    docs = []
    for source in sorted(meta['splits'][split]['sources']):
        row = inventory[source]
        path = (BASE/row['clean_file']).resolve()
        require(path.parent == (BASE/'datos/documentos_limpios').resolve(), 'Invalid document path.')
        data = path.read_bytes()
        require(sha256(data) == row['sha256'], 'Clean TXT changed: ' + source)
        text = data.decode('utf-8')
        body = tokenizer.encode(text)
        require(body and all(4 <= t < 6000 for t in body), 'Empty body or unexpected special/invalid ID.')
        require(tokenizer.decode(body) == text, 'Document roundtrip failed: ' + source)
        ids = torch.tensor([2] + body + [3], dtype=torch.int32)
        docs.append({'source':source, 'sha256_text':sha256(data), 'body_tokens':len(body),
                     'tokens':ids, 'text':text})
    require(len(docs) == {'train':14, 'validation':4, 'test':3}[split], 'Wrong document count.')
    return docs

def starts_for_document(n, context, stride):
    require(n >= 2 and context >= 1 and 1 <= stride <= context, 'Invalid window parameters.')
    if n < context + 1:
        return [0]
    last = n - context - 1
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts

def window_index(lengths, context, stride, evaluation):
    rows = []
    for doc_id, n in enumerate(lengths):
        covered_end = 0  # Absolute target positions 1..n-1; BOS is position 0.
        for start in starts_for_document(n, context, stride):
            valid = min(context, n-start-1)
            ignored_prefix = min(valid, max(0, covered_end-start)) if evaluation else 0
            rows.append([doc_id, start, ignored_prefix])
            covered_end = max(covered_end, start+valid)
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 3)

class DatasetSecuencias(Dataset):
    """Returns (X,Y), both long [L]. Y=-100 means no loss contribution."""
    def __init__(self, split='train', ruta=None, data=None):
        require(split in ('train', 'validation', 'test'), 'Unknown split.')
        self.data = data if data is not None else torch.load(
            ruta or CARPETA_DATASET/(split+'.pt'), map_location='cpu', weights_only=True)
        require(self.data['version'] == VERSION and self.data['split'] == split, 'Incompatible dataset format/split.')
        require(self.data['tokenizer_sha256'] == TOKENIZER_HASH, 'Wrong tokenizer provenance.')
        self.split = split
        self.context = self.data['context_length']
        self.documents = self.data['documents']
        self.windows = self.data['window_index']
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, index):
        doc_id, start, prefix = self.windows[index].tolist()
        seq = self.documents[doc_id]['tokens'][start:start+self.context+1].to(torch.long)
        x = torch.full((self.context,), 0, dtype=torch.long)
        y = torch.full((self.context,), -100, dtype=torch.long)
        x[:min(self.context, len(seq))] = seq[:self.context]
        y[:len(seq)-1] = seq[1:]
        y[:prefix] = -100
        return x, y

def crear_dataloader(split='train', batch_size=TAMANIO_BATCH, shuffle=None, seed=SEMILLA, ruta=None):
    if shuffle is None:
        shuffle = split == 'train'
    require(split == 'train' or not shuffle, 'Evaluation requires shuffle=False.')
    require(batch_size > 0, 'Batch size must be positive.')
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(DatasetSecuencias(split, ruta), batch_size=batch_size,
                      shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)

class GestorDataset:
    """Compatibility facade; future trainer must use len(loader) and target counts."""
    def __init__(self, batch_size=TAMANIO_BATCH, dispositivo='cpu'):
        self.loaders = {s:crear_dataloader(s, batch_size) for s in ('train', 'validation')}
        self.train = self.loaders['train'].dataset
        self.val = self.loaders['validation'].dataset
        self.pasos_por_epoca = len(self.loaders['train'])
        self.device = dispositivo
        self.iterators = {}
    def obtener_batch(self, division='train'):
        split = 'validation' if division in ('val','validation') else division
        require(split in self.loaders, 'No implicit Test loader or Train fallback.')
        if split not in self.iterators:
            self.iterators[split] = iter(self.loaders[split])
        iterator = self.iterators[split]
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(self.loaders[split]); self.iterators[split] = iterator
            x, y = next(iterator)
        return x.to(self.device), y.to(self.device)

def index_statistics(lengths, context, stride, evaluation):
    idx = window_index(lengths, context, stride, evaluation)
    unmasked = raw_targets = padded_windows = aligned = 0
    for doc_id, start, prefix in idx.tolist():
        valid = min(context, lengths[doc_id]-start-1)
        raw_targets += valid
        unmasked += valid-prefix
        padded_windows += lengths[doc_id]-start < context+1
    for n in lengths:
        if n >= context+1 and (n-context-1) % stride:
            aligned += 1
    unique = sum(n-1 for n in lengths)
    return {'context':context, 'stride':stride, 'windows':len(idx),
            'raw_target_exposures':raw_targets, 'effective_nonignored_targets':unmasked,
            'unique_target_positions':unique, 'raw_redundancy_factor':raw_targets/unique,
            'repeated_targets_masked':raw_targets-unmasked,
            'padding_windows':padded_windows, 'padding_window_percentage':100*padded_windows/len(idx),
            'documents_with_full_window':sum(n >= context+1 for n in lengths),
            'documents_with_full_window_percentage':100*sum(n >= context+1 for n in lengths)/len(lengths),
            'documents_with_regular_partial_tail':aligned, 'documents_requiring_actual_partial_window':padded_windows,
            'end_aligned_documents':aligned, 'batches':{str(b):math.ceil(len(idx)/b) for b in (8,16,32)},
            'attention_relative_per_sequence':(context/64)**2}

def paragraph_statistics(documents, tokenizer):
    lengths = sorted(len(tokenizer.encode(p)) for d in documents
                     for p in re.split(r'\n\s*\n',d['text']) if p.strip())
    def percentile(q):
        position = (len(lengths)-1)*q
        lo, hi = math.floor(position), math.ceil(position)
        return lengths[lo]+(lengths[hi]-lengths[lo])*(position-lo)
    return {'paragraph_definition':'Nonempty spans separated by existing blank lines; no file modification',
            'paragraphs':len(lengths), 'median':percentile(.5), 'p75':percentile(.75),
            'p90':percentile(.9), 'p95':percentile(.95),
            'coverage_percentage':{str(L):100*sum(n<=L for n in lengths)/len(lengths) for L in (64,128,256)}}

def make_payload(split, docs, context, stride):
    documents = [{k:v for k,v in d.items() if k != 'text'} for d in docs]
    return {'version':VERSION, 'split':split, 'context_length':context, 'stride':stride,
            'evaluation_unique_targets':split != 'train', 'ignore_index':-100,
            'tokenizer_sha256':TOKENIZER_HASH, 'vocab_size':6000,
            'source_split_sha256':SPLIT_HASHES[split+'.txt'],
            'documents':documents, 'window_index':window_index([len(d['tokens']) for d in docs],context,stride,split!='train')}

def canonical_hash(payload):
    digest = hashlib.sha256()
    metadata = {k:v for k,v in payload.items() if k not in ('documents','window_index')}
    metadata['documents'] = [{k:v for k,v in d.items() if k != 'tokens'} for d in payload['documents']]
    digest.update(json.dumps(metadata,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8'))
    for tensor in [d['tokens'] for d in payload['documents']] + [payload['window_index']]:
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(json.dumps(list(tensor.shape)).encode('ascii'))
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()

def serialize(payload):
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()

def validate_dataset(payload):
    ds = DatasetSecuencias(payload['split'], data=payload)
    counts = [torch.zeros(len(d['tokens']),dtype=torch.int32) for d in ds.documents]
    eos_targets = [0]*len(ds.documents)
    padding_positions = padded_windows = 0
    for i, (doc_id, start, prefix) in enumerate(ds.windows.tolist()):
        x, y = ds[i]
        require(x.dtype == y.dtype == torch.long and x.shape == y.shape == (ds.context,), 'Wrong dtype/shape.')
        doc = ds.documents[doc_id]['tokens'].to(torch.long)
        valid_x = min(ds.context,len(doc)-start)
        valid_y = min(ds.context,len(doc)-start-1)
        require(torch.equal(x[:valid_x],doc[start:start+valid_x]), 'X crosses/mismatches document.')
        require(bool((x[valid_x:] == 0).all()), 'Padding not trailing.')
        expected = doc[start+1:start+1+valid_y]
        require(torch.equal(y[prefix:valid_y],expected[prefix:]), 'Next-token shift incorrect.')
        require(bool((y[:prefix] == -100).all()) and bool((y[valid_y:] == -100).all()), 'Wrong ignore mask.')
        require(bool(((x>=0)&(x<6000)).all()), 'Invalid input ID.')
        real = y[y != -100]
        require(bool(((real>=1)&(real<6000)&(real!=2)).all()), 'PAD/BOS/invalid target.')
        counts[doc_id][start+1+prefix:start+1+valid_y] += 1
        eos_targets[doc_id] += int((real == 3).sum())
        padding_positions += ds.context-valid_x
        padded_windows += len(doc)-start < ds.context+1
    rows = []
    for i,d in enumerate(ds.documents):
        ids=d['tokens']
        require(int(ids[0]) == 2 and int(ids[-1]) == 3, 'Wrong boundaries.')
        require(int((ids==2).sum()) == int((ids==3).sum()) == 1, 'Multiple BOS/EOS.')
        require(int((ids==0).sum()) == int((ids==1).sum()) == 0, 'Unexpected PAD/UNK in document.')
        require(int(counts[i][0]) == 0 and bool((counts[i][1:]>=1).all()), 'Targets omitted/BOS evaluated.')
        if payload['split'] != 'train':
            require(bool((counts[i][1:]==1).all()) and eos_targets[i]==1, 'Evaluation repeats targets.')
        require(eos_targets[i]>=1, 'EOS not targeted.')
        rows.append({'source':d['source'], 'body_tokens':d['body_tokens'], 'tokens_with_bos_eos':len(ids),
                     'windows':int((ds.windows[:,0]==i).sum()), 'unique_targets':len(ids)-1,
                     'nonignored_target_exposures':int(counts[i].sum()), 'eos_targets':eos_targets[i]})
    return {'documents':len(ds.documents), 'body_tokens':sum(d['body_tokens'] for d in ds.documents),
            'tokens_with_bos_eos':sum(len(d['tokens']) for d in ds.documents), 'windows':len(ds),
            'unique_targets':sum(len(d['tokens'])-1 for d in ds.documents),
            'nonignored_targets':sum(int(c.sum()) for c in counts), 'BOS_count':len(ds.documents),
            'EOS_count':len(ds.documents), 'UNK_count':0, 'padding_input_positions':padding_positions,
            'padded_windows':padded_windows,
            'documents_detail':rows, 'all_integrity_tests_passed':True}

def synthetic_padding_tests():
    # Independent position-coverage tests across short/exact/nonmultiple lengths.
    checks=0
    for n in range(2,45):
        for L in (4,8,16):
            for stride in (L,L//2):
                for evaluation in (False,True):
                    ids=torch.tensor([2]+[5]*(n-2)+[3],dtype=torch.int32)
                    payload={'version':VERSION,'split':'validation' if evaluation else 'train','context_length':L,
                             'tokenizer_sha256':TOKENIZER_HASH,'documents':[{'source':'synthetic','tokens':ids,'body_tokens':n-2}],
                             'window_index':window_index([n],L,stride,evaluation)}
                    validate_dataset(payload);checks+=1
    return {'length_stride_evaluation_cases':checks,'padding_and_unique_target_tests_passed':True}

def manual_examples(payload, tokenizer):
    ds=DatasetSecuencias('train',data=payload)
    examples=[]
    for doc_id,mode in ((0,'bos'),(1,'middle'),(2,'eos')):
        candidates=(ds.windows[:,0]==doc_id).nonzero().flatten().tolist()
        index=candidates[0] if mode=='bos' else candidates[-1] if mode=='eos' else candidates[len(candidates)//2]
        x,y=ds[index]
        examples.append({'source':ds.documents[doc_id]['source'],'mode':mode,'window_index':index,
                         'start':int(ds.windows[index,1]),'x_first':x[:6].tolist(),'y_first':y[:6].tolist(),
                         'x_last':x[-6:].tolist(),'y_last':y[-6:].tolist(),
                         'short_decode':tokenizer.decode(x.tolist())[:65]})
    synthetic={'version':VERSION,'split':'train','context_length':LONGITUD_CONTEXTO,'tokenizer_sha256':TOKENIZER_HASH,
               'documents':[{'source':'synthetic_padding_only','tokens':torch.tensor([2,5,6,3],dtype=torch.int32),'body_tokens':2}],
               'window_index':window_index([4],LONGITUD_CONTEXTO,STRIDE_ENTRENAMIENTO,False)}
    x,y=DatasetSecuencias('train',data=synthetic)[0]
    examples.append({'source':'synthetic_padding_only; not in real datasets','mode':'padding_demo',
                     'x_first':x[:7].tolist(),'y_first':y[:7].tolist(),'x_last':x[-3:].tolist(),'y_last':y[-3:].tolist()})
    return examples

def generate_datasets():
    verify_inputs()
    tokenizer=TokenizadorEducativo().cargar()
    require(tokenizer.vocab_size==6000 and (tokenizer.pad_id,tokenizer.unk_id,tokenizer.bos_id,tokenizer.eos_id)==(0,1,2,3),'Unexpected tokenizer API.')
    print('Encoding only Train and Validation for context analysis...',flush=True)
    docs={s:encode_documents(s,tokenizer) for s in ('train','validation')}
    require(sum(d['body_tokens'] for d in docs['train'])==799670 and sum(d['body_tokens'] for d in docs['validation'])==111640,'2C token counts changed.')
    paragraphs={s:paragraph_statistics(ds,tokenizer) for s,ds in docs.items()}
    candidates={s:[index_statistics([len(d['tokens']) for d in ds],L,stride,s!='train')
                   for L in (64,128,256) for stride in (L,L//2)] for s,ds in docs.items()}
    # Decide exclusively from Train/Validation; Test has not been opened as text.
    coverage=paragraphs['validation']['coverage_percentage']
    selected=min((L for L in (64,128,256) if coverage[str(L)]>=75),default=256)
    selected_stride=selected//2
    selected_train_stats=next(c for c in candidates['train'] if c['context']==selected and c['stride']==selected_stride)
    require(selected_train_stats['raw_redundancy_factor']<=2.1,'Excessive overlap.')
    require((LONGITUD_CONTEXTO,STRIDE_ENTRENAMIENTO,STRIDE_EVALUACION,IGNORE_INDEX_DATASET)==(selected,selected_stride,selected,-100),'config.py does not match Train/Validation decision.')
    frozen_configuration=(selected,selected_stride,selected)
    print('Configuration frozen before Test: context=%d train stride=%d evaluation stride=%d'%frozen_configuration,flush=True)
    payloads={s:make_payload(s,ds,selected,selected_stride if s=='train' else selected) for s,ds in docs.items()}
    # Test content is first read AFTER all selection decisions have been frozen.
    docs['test']=encode_documents('test',tokenizer,test_allowed=True)
    payloads['test']=make_payload('test',docs['test'],frozen_configuration[0],frozen_configuration[2])
    sources=[d['source'] for ds in docs.values() for d in ds]
    require(len(sources)==len(set(sources))==21,'Sources cross splits.')
    statistics={s:validate_dataset(p) for s,p in payloads.items()}
    synthetic=synthetic_padding_tests()
    blobs={s:serialize(p) for s,p in payloads.items()}
    second_docs={s:encode_documents(s,tokenizer,test_allowed=(s=='test')) for s in ('train','validation','test')}
    second={s:make_payload(s,ds,selected,selected_stride if s=='train' else selected) for s,ds in second_docs.items()}
    repeat_blobs={s:serialize(p) for s,p in second.items()}
    byte_identical={s:blobs[s]==repeat_blobs[s] for s in blobs}
    canonical={s:canonical_hash(p) for s,p in payloads.items()}
    require(all(canonical[s]==canonical_hash(second[s]) for s in second),'Canonical dataset contents changed.')
    for s,p in payloads.items():
        a,b=DatasetSecuencias(s,data=p),DatasetSecuencias(s,data=second[s])
        for index in (0,len(a)//2,len(a)-1):
            require(all(torch.equal(x,y) for x,y in zip(a[index],b[index])),'Repeated samples differ.')
        require(torch.equal(p['window_index'],second[s]['window_index']),'Window indices differ.')
    storage={}
    for s,p in payloads.items():
        token_bytes=sum(d['tokens'].numel()*d['tokens'].element_size() for d in p['documents'])
        index_bytes=p['window_index'].numel()*p['window_index'].element_size()
        materialized=len(p['window_index'])*selected*2*8
        storage[s]={'document_tokens_bytes':token_bytes,'window_indices_bytes':index_bytes,
                    'compact_tensor_bytes':token_bytes+index_bytes,'materialized_xy_int64_bytes':materialized,
                    'serialized_bytes':len(blobs[s]),'canonical_sha256':canonical[s]}
    batches={str(b):{'shape_x': [b,selected],'shape_y':[b,selected],'input_tokens_per_batch':b*selected,
                    'xy_int64_bytes':2*b*selected*8,'train_batches_per_epoch':math.ceil(statistics['train']['windows']/b)} for b in (8,16,32)}
    reason={'rule':'Smallest candidate covering >=75% of Validation paragraphs; Train stride L/2 with exposure redundancy <=2.1',
            'validation_paragraph_coverage_64':coverage['64'],'validation_paragraph_coverage_128':coverage['128'],
            'validation_paragraph_coverage_256':coverage['256'],'train_windows_selected':statistics['train']['windows'],
            'train_target_redundancy':selected_train_stats['raw_redundancy_factor'],
            'attention_cost_256_vs_128':4,'note':'Extra overlapping windows provide shifted contexts, not new unique text. Paragraph spans are auxiliary evidence.'}
    report={'version':VERSION,'torch_version':str(torch.__version__),'source_split_hashes':SPLIT_HASHES,'tokenizer_sha256':TOKENIZER_HASH,
            'sha256_dataset_code':sha256(Path(__file__).read_bytes()),'sha256_config':sha256((BASE/'config.py').read_bytes()),
            'vocab_size':6000,'special_token_ids':{'PAD':0,'UNK':1,'BOS':2,'EOS':3},'ignore_index':-100,
            'context_candidates':[64,128,256],'paragraph_length_statistics':paragraphs,'window_statistics_by_candidate':candidates,
            'relative_attention_cost':{'64':1,'128':4,'256':16},'selected_context_length':selected,'selected_train_stride':selected_stride,
            'selected_evaluation_stride':selected,'selection_reason':reason,'configuration_frozen_before_test_content':True,'test_used_for_selection':False,
            'evaluation_window_protocol':'Regular stride L, then unique end-aligned full window. For targets at absolute document positions <= previous covered_end, set Y=-100. Every position 1..N-1 contributes exactly once; BOS is never a target; EOS is once.',
            'train_statistics':statistics['train'],'validation_statistics':statistics['validation'],'test_integrity_statistics':statistics['test'],
            'documents_per_split':{s:len(ds) for s,ds in docs.items()},
            'tokens_per_document':{s:[{'source':d['source'],'before_bos_eos':d['body_tokens'],'after_bos_eos':len(d['tokens'])} for d in ds] for s,ds in docs.items()},
            'windows_per_document':{s:statistics[s]['documents_detail'] for s in statistics},
            'padding_statistics':{s:{'input_pad_positions':statistics[s]['padding_input_positions'],'padded_windows':statistics[s]['padded_windows']} for s in statistics},
            'evaluable_targets':{s:statistics[s]['nonignored_targets'] for s in statistics},
            'BOS_count':21,'EOS_count':21,'UNK_count':0,
            'storage_format':{'version':VERSION,'document_tokens_dtype':'int32','window_index_dtype':'int64','window_columns':['document_index','start','ignored_target_prefix'],'getitem_dtype':'torch.long','loading':'torch.load(weights_only=True, map_location=cpu)'},
            'estimated_storage':storage,'batch_preanalysis':batches,'manual_examples':manual_examples(payloads['train'],tokenizer),
            'integrity_tests':{'all_windows_checked':True,'all_documents_exact_roundtrip':True,'no_cross_document_windows':True,'evaluation_each_target_once':True,**synthetic},
            'determinism':{'two_fresh_document_encodings':True,'byte_identical_pt':byte_identical,'canonical_content_sha256':canonical,'first_middle_last_samples_identical':True,'window_indices_identical':True},
            'sha256_train_dataset':sha256(blobs['train']),'sha256_validation_dataset':sha256(blobs['validation']),'sha256_test_dataset':sha256(blobs['test']),
            'warnings':['entrenar.py remains unchanged: its len(train)/(batch*L) epoch calculation is obsolete; use len(DataLoader).',
                        'Evaluation must accumulate summed cross-entropy over nonignored targets and divide by total target count; do not average unequal masked batches blindly.',
                        'Paragraphs are blank-line spans from PDF extraction; Train spans often resemble long page blocks. Context choice is structural, not validated by Transformer training.',
                        'Causal masking already prevents trailing padding from influencing preceding valid tokens; no extra attention padding mask is added.',
                        'No final batch-size decision or GPU training benchmark was made; verified runtime is PyTorch CPU.']}
    verify_inputs()
    CARPETA_DATASET.mkdir(parents=True,exist_ok=True)
    for s,data in blobs.items():
        (CARPETA_DATASET/(s+'.pt')).write_bytes(data)
        loaded=torch.load(CARPETA_DATASET/(s+'.pt'),map_location='cpu',weights_only=True)
        require(canonical_hash(loaded)==canonical[s],'Saved content changed.')
    # DataLoader smoke checks only; never invoke the Transformer.
    for s in ('train','validation','test'):
        x,y=next(iter(crear_dataloader(s,batch_size=8)))
        require(x.dtype==y.dtype==torch.long and x.shape[1]==y.shape[1]==selected,'DataLoader failed.')
    text=(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8')
    (CARPETA_DATASET/'reporte_dataset.json').write_bytes(text)
    require(text==(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8'),'Report serialization unstable.')
    print(json.dumps({'selected_context':selected,'train_stride':selected_stride,'evaluation_stride':selected,
                      'statistics':statistics,'storage':storage,'byte_identical_pt':byte_identical,
                      'sha256_report':sha256(text)},ensure_ascii=True,indent=2),flush=True)
    return report

if __name__=='__main__':
    generate_datasets()
