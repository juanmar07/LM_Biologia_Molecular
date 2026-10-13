"""5C: evidencia local, candidato MiniGPT y entrega conservadora con procedencia."""
import seguridad_5C_common as integrity  # Instalar guard antes de cargar el runtime.
import re
from collections import Counter
from retrieval import retrieve, tokens, normalize
from rerank_rag_5B1 import rerank, query_analysis, intent_signal, PREDICATE, GENERIC
from contexto_rag_5B1 import pack_neighbors, source_lookup
from rag import RAG, greedy_generate
from evaluar_rag_5B import response_metrics

ABSTENTION = 'No encontré información suficientemente clara en los documentos disponibles para responder con seguridad.'
POLICY = dict(version='5C-v1', top_k=10, variant='D', minimum_topic_coverage=0.8,
              minimum_new_content_terms=3, minimum_candidate_precision=0.95,
              candidate_proposition='literal complete selected proposition; no semantic paraphrase verifier',
              context_scan='first sufficient context in unchanged rerank order, within original top10')
CONTENT_EXCLUDED = GENERIC | frozenset('tamano tecnica tecnicas metodo metodos proceso procesos puede pueden permite permiten utiliza utilizan emplea emplean consiste capaces ejemplo observa observe actualidad acuerdo herramienta basica basico'.split())

def clean(text):
    """Única transformación de la respuesta extractiva: espacios/saltos de línea."""
    return ' '.join(text.split())

def content(text):
    return [t for t in tokens(text) if t not in CONTENT_EXCLUDED]

def analysis_for(question):
    analysis = query_analysis(question)
    analysis['significant_terms'] = [t for t in analysis['significant_terms'] if t not in ('tamano','designa','termino')]
    if re.search(r'que (?:significa|designa|expresa|representa)\b',normalize(question)):
        analysis['intent'] = 'definition'
    if not analysis['scientific_terms']:
        analysis['principal_term'] = next(iter(analysis['significant_terms']), '')
    return analysis

def units(context):
    """Unidades literales con offsets locales; encabezados separados de su proposición."""
    result = []
    for match in re.finditer(r'.+?(?:[.!?]["»”\)]*(?=\s+[A-ZÁÉÍÓÚÜÑ¿¡]|$)|$)', context, re.S):
        raw = match.group(); start = match.start()
        leading = len(raw) - len(raw.lstrip()); raw = raw.lstrip(); start += leading
        headings = []
        while '\n' in raw:
            first, rest = raw.split('\n', 1)
            if not first.strip() or (not PREDICATE.search(normalize(first)) and len(tokens(first, False)) <= 14):
                headings.append(first.strip()); start += len(first) + 1; raw = rest
            else:
                break
        leading = len(raw) - len(raw.lstrip()); start += leading; raw = raw.lstrip().rstrip()
        if raw:
            result.append(dict(text=raw, cleaned=clean(raw), start=start, end=start+len(raw),
                               heading=' '.join(headings), after_sentence_boundary=match.start()>0))
    return result

def unit_support(unit, analysis, packed):
    text = normalize(unit['text']); heading = normalize(unit['heading'])
    topic = set(analysis['significant_terms']); present = set(tokens(text))
    heading_binding = bool(topic and topic <= set(tokens(heading)) and re.match(r'^(?:son|es|consiste|se trata)\b', text))
    effective = present | (set(tokens(heading)) if heading_binding else set())
    coverage = len(topic & effective)/len(topic) if topic else 0.
    principal = bool(analysis['principal_term'] and analysis['principal_term'] in effective)
    # Binding must be in the grammatical subject, rather than an incidental later mention.
    words = tokens(text, False)
    first_predicate = PREDICATE.search(text)
    subject = text[:first_predicate.start()] if first_predicate else ''
    subject_terms = set(tokens(subject))
    ordered_topic = analysis['scientific_terms'] or analysis['significant_terms']
    # A query entity must start the subject. «fragmento de ADN» does not define ADN;
    # «esta modalidad de PCR» does not state the general function of PCR.
    entity = r'\b' + r'\s+(?:(?:de|del|la|las|el|los|y)\s+)*'.join(re.escape(t) for t in ordered_topic) + r'\b'
    entity_at_start = bool(ordered_topic and re.match(r'^(?:(?:el|la|los|las|un|una)\s+)?'+entity, subject))
    subject_bound = (bool(topic and topic <= subject_terms) and entity_at_start) or heading_binding
    # No scientific answer templates: only grammatical subject/predicate patterns.
    intent = analysis['intent']
    direct_definition = subject_bound and bool(first_predicate and re.match(
        r'(?:es|son|consiste|designa|representa|expresa|implica|refiere|se define)\b(?!\s*:)', text[first_predicate.start():]))
    if intent == 'definition': matched = direct_definition
    elif intent in ('function', 'characteristic'):
        matched = subject_bound and bool(intent_signal(analysis, text))
    elif intent == 'comparison': matched = bool(intent_signal(analysis, text)) and topic <= effective
    else: matched = bool(PREDICATE.search(text)) and coverage >= POLICY['minimum_topic_coverage']
    new_content = sorted(set(content(text))-topic)
    informative = bool(first_predicate) and len(words) >= 6 and len(new_content) >= POLICY['minimum_new_content_terms']
    complete = bool(re.search(r'[.!?]["»”\)]*$', unit['text']))
    complete = complete and unit['text'].count('(')==unit['text'].count(')') and not re.search(r'\b(?:[a-z]|ej|etc)\.$',text)
    # The first unit may begin inside a sentence; later units have a real sentence boundary.
    complete = complete and (packed['completeness']['start_boundary'] or unit['after_sentence_boundary'] or heading_binding)
    mention_only = bool(topic & effective) and not (matched and informative)
    safe = coverage >= POLICY['minimum_topic_coverage'] and principal and matched and informative and complete and not mention_only
    return dict(sufficient=safe, query_term_coverage=coverage, principal_term_present=principal,
                intent_match=bool(matched), informative=informative, mention_only=mention_only,
                structural_completeness=complete, heading_binding=heading_binding,
                subject_bound=subject_bound, new_content_terms=new_content,
                existing_intent_signal=intent_signal(analysis, text))

def evidence_support(question, packed, anchor):
    analysis = analysis_for(question)
    inspected = [dict(u, support=unit_support(u, analysis, packed)) for u in units(packed['context'])]
    useful = [u for u in inspected if u['support']['sufficient']]
    useful.sort(key=lambda u: (-u['support']['query_term_coverage'],
                              -u['support']['existing_intent_signal'], -int(u['support']['principal_term_present']), u['start']))
    return dict(sufficient=bool(useful), analysis=analysis, units=inspected,
                useful_units=useful, rerank_score=anchor['rerank_score'],
                existing_rerank_signals=anchor['rerank_signals'],
                context_structural_completeness=packed['completeness'])

def validate_candidate(generation, context, evidence):
    answer = generation['answer']; a = Counter(content(answer)); b = Counter(content(context))
    overlap = sum((a & b).values()); topic = set(evidence['analysis']['significant_terms'])
    principal = evidence['analysis']['principal_term']
    repetition = response_metrics(generation, '', context)['repetition_rate']
    useful = evidence['useful_units']
    # Literal proposition verification preserves order, negation, quantities and qualifiers.
    # Lexical overlap alone never licenses a scientific paraphrase.
    preserved = next((u for u in useful if clean(answer).casefold() == u['cleaned'].casefold()), None)
    metrics = dict(normalized_content_token_precision_vs_context=overlap/sum(a.values()) if a else 0.,
                   normalized_content_token_recall_vs_context=overlap/sum(b.values()) if b else 0.,
                   principal_term_preserved=bool(principal and principal in set(tokens(answer))),
                   novel_content_terms=sorted(set(a)-set(b)), repetition=repetition,
                   useful_proposition_preserved=bool(preserved), EOS=generation['EOS'])
    reasons = []
    if repetition: reasons.append('repetition')
    if not answer.strip(): reasons.append('empty')
    if len(a) < 3: reasons.append('too_short_without_content')
    if len(set(a)-topic) < POLICY['minimum_new_content_terms']: reasons.append('tautology_or_contentless')
    if metrics['normalized_content_token_precision_vs_context'] < POLICY['minimum_candidate_precision']: reasons.append('low_context_precision')
    if metrics['novel_content_terms']: reasons.append('novel_content_terms')
    if not metrics['principal_term_preserved']: reasons.append('missing_principal')
    if not preserved: reasons.append('no_verified_useful_proposition')
    if not generation['EOS']: reasons.append('missing_EOS')
    metrics.update(rejection_reasons=reasons, accepted=not reasons)
    return metrics

class ChatSeguro:
    def __init__(self):
        runtime = RAG()
        self.model, self.tokenizer, self.model_metadata = runtime.model, runtime.tokenizer, runtime.model_metadata

    def answer(self, question):
        if not isinstance(question, str): raise TypeError('La pregunta debe ser texto')
        found = retrieve(question, POLICY['top_k'])
        ranked = rerank(question, found)
        inspected = []; chosen = None
        for anchor in ranked:
            try: packed = pack_neighbors(question, [anchor], self.tokenizer, 1)
            except ValueError as exc:
                inspected.append(dict(chunk_id=anchor['chunk_id'], sufficient=False, packing_error=str(exc))); continue
            evidence = evidence_support(question, packed, anchor)
            inspected.append(dict(chunk_id=anchor['chunk_id'], context=packed['context'], evidence=evidence))
            if evidence['sufficient']:
                chosen = (anchor, packed, evidence); break
        row = dict(question=question, decision='ABSTAIN', generator_candidate=None, candidate_raw=None,
                   candidate_token_ids=[], generator_accepted=False, fallback_used=False, abstention_used=True,
                   generation_executed=False, final_answer=ABSTENTION, source_doc=None, chunk_id=None,
                   BM25_score=None, rerank_score=None, support_metrics=None, evidence=None,
                   generation=None, packing=None, fallback_sentence=None, fallback_source_spans=[],
                   bm25_retrieved=found, retrieval=ranked, inspected_contexts=inspected, policy=POLICY)
        if chosen is None: return row
        anchor, packed, evidence = chosen
        generation = greedy_generate(self.model, self.tokenizer, packed)
        metrics = validate_candidate(generation, packed['context'], evidence)
        unit = evidence['useful_units'][0]
        span = packed['context_source_span']
        start, end = span['start']+unit['start'], span['start']+unit['end']
        original = source_lookup()[span['source_doc']]['text'][start:end]
        integrity.require(original == unit['text'], 'Fallback no coincide con offsets originales')
        row.update(decision='GENERATOR_ACCEPTED' if metrics['accepted'] else 'EXTRACTIVE_FALLBACK',
                   generator_candidate=generation['answer'], candidate_raw=generation['raw_generated_text'],
                   candidate_token_ids=generation['generated_token_ids'], generator_accepted=metrics['accepted'],
                   fallback_used=not metrics['accepted'], abstention_used=False, generation_executed=True,
                   final_answer=generation['answer'] if metrics['accepted'] else unit['cleaned'],
                   source_doc=anchor['source_doc'], chunk_id=anchor['chunk_id'], BM25_score=anchor['bm25_score'],
                   rerank_score=anchor['rerank_score'], support_metrics=metrics, evidence=evidence,
                   generation=generation, packing=packed, fallback_sentence=unit['cleaned'] if not metrics['accepted'] else None,
                   fallback_source_spans=[dict(source_doc=span['source_doc'], start=start, end=end,
                                               raw=original, heading_binding=unit['support']['heading_binding'])] if not metrics['accepted'] else [])
        return row
