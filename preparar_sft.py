"""Construcción extractiva conservadora 5A, sin APIs externas ni acceso a Test.

Una afirmación autónoma por concepto; no se completan fuentes con conocimiento externo.
La auditoría prueba soporte literal, no certifica la verdad científica del documento.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from statistics import mean

from dataset_sft import (BASE, SFT, ABSTENTION, TASKS, require, sha256, write_json,
                         source_documents, prompt, encode_record, install_test_guard)
from tokenizador import TokenizadorEducativo

SEED = 42
RELATION = re.compile(r'^((?:El|La|Los|Las|Un|Una) .{3,85}?) (se define(?:n)? como|se denomina(?:n)?|son|es|permite(?:n)?|presenta(?:n)?|contiene(?:n)?|produce(?:n)?|forman|forma|se caracteriza(?:n)?|se clasifica(?:n)?|se encuentra(?:n)?|se replica(?:n)?|se transmite(?:n)?|se utiliza(?:n)?|se convierte(?:n)?|separa(?:n)?|participa(?:n)?|regula(?:n)?|origina(?:n)?|genera(?:n)?|depende(?:n)?|puede(?:n)?|tiene(?:n)?|cambia(?:n)?|codifica(?:n)?|actúa(?:n)?|actua(?:n)?|consiste(?:n)?|reconoce(?:n)?|representa(?:n)?|incluye(?:n)?|protege(?:n)?|requiere(?:n)?|evita(?:n)?|induce(?:n)?|inhibe(?:n)?|cataliza(?:n)?|resulta(?:n)?|se une(?:n)?|se divide(?:n)?|se modifica(?:n)?|se obtiene(?:n)?|se aísla(?:n)?|se sintetiza(?:n)?|se forma(?:n)?|interviene(?:n)?)\b (.+)\.$', re.I)
SCIENCE = re.compile(r'\b(?:ADN|DNA|ARN|RNA|ARNm|gen(?:es|oma|ética|etica|ómico|ómica|otipo)?|prote[íi]na(?:s)?|enzima(?:s)?|c[ée]lula(?:s)?|cromosoma(?:s)?|pl[áa]smido(?:s)?|vector(?:es)?|nucle[óo]tido(?:s)?|nucle[óo]sido(?:s)?|transcripci[óo]n|traducci[óo]n|replicaci[óo]n|metilaci[óo]n|acetilaci[óo]n|histona(?:s)?|epigen[ée]tica|epigenoma|microARN|CRISPR|Cas9|mutaci[óo]n|intr[óo]n|ex[óo]n|cromatina|ribosoma(?:s)?|retrovirus|bacteri[óo]fago(?:s)?|endonucleasa(?:s)?|ligasa(?:s)?|polimerasa(?:s)?|c[óo]smido(?:s)?|tel[óo]mero(?:s)?|oper[óo]n|alelo(?:s)?|mitosis|meiosis|blastocisto|PCR|recombinaci[óo]n|terapia g[ée]nica|diferenciaci[óo]n)\b', re.I)
DEPENDENT = re.compile(r'\b(?:figura|cuadro|tabla|diapositiva|p[áa]g|cap[íi]tulo|anterior|anteriormente|siguiente|este|esta|estos|estas|ese|esa|esos|esas|ambos|ambas|dicho|dicha|ello|as[íi]|aqu[íi]|citado|autor|doi|copyright)\b|https?://|[�•♦]|[a-záéíóú]-\s+[a-záéíóú]', re.I)
EXTRA_RELATION = re.compile(r'^((?:El|La|Los|Las|Un|Una) .{3,85}?) (corresponde(?:n)?|consta(?:n)?|posee(?:n)?|se realiza(?:n)?|se agrupa(?:n)?|se asocia(?:n)?|se basa(?:n)?|se emplea(?:n)?|se incorpora(?:n)?|se extrae(?:n)?|se introduce(?:n)?|se conoce(?:n)?|mantiene(?:n)?|se mantiene(?:n)?|se localiza(?:n)?|se separa(?:n)?|se desarrolla(?:n)?|se establece(?:n)?|se expresa(?:n)?|se inicia(?:n)?|se determina(?:n)?|se a[ñn]ade(?:n)?|aumenta(?:n)?|disminuye(?:n)?|no incrementa(?:n)?|pertenece(?:n)?|desempeña(?:n)?|necesita(?:n)?|reconocen|han permitido|ha permitido|ha demostrado|ha acelerado|ha ampliado|ha abierto|permanece(?:n)?|a menudo permanece(?:n)?)\b (.+)\.$', re.I)


def parse_relation(sentence):
    matches = [m for m in (RELATION.fullmatch(sentence), EXTRA_RELATION.fullmatch(sentence)) if m]
    return min(matches, key=lambda m: len(m[1])) if matches else None


BAD_SUBJECT = re.compile(r'\b(?:que|cual|no|se|es|son|está|están|permanece|permanecen|sufren|sintetiza|ha|solo|suelen|contiene|especifica|hacen|hace|sugieren|primero|primeros|segundo|segundos|vez|muestra|individuo|paciente)\b|[,;:]|\b(?:se|y|o|de|del|en|por|con|a|d)\s*$', re.I)
NOISE = re.compile(r'\b(?:fi gura|fi g|efi ciente|infl uir|transcripciona l|acualquier|producir se|induc iendo|codifi cadora|específi ca|identifi cación|polimorfi smos|pres enta|vent ajas|muyrelacionados|especiesdiferentes|induciblegen|inductorinductor|activadoractivador|ARNmensajero|ambas situaciones|dos modificaciones|varios mecanismos|distintos mecanismos|varias vent|muy variados|mayores obstáculos)\b|\d\.$', re.I)
AMBIGUOUS_SUBJECT = re.compile(r'^(?:la proteína|el fragmento de ADN|las tres enzimas|una sola mutación|el ARN viral|el gen responsable.*|la enzima más empleada|la inserción completa del vector|la preparación de partículas víricas.*)$', re.I)


def normalize(text):
    return ' '.join(text.split())


def canonical(text):
    return ' '.join(re.findall(r'\w+', text.lower()))


def shingles(text):
    words = canonical(text).split()
    return set(zip(words, words[1:], words[2:]))


def question_for(subject, verb, body, index):
    # La etiqueta describe la operación pedida y se asigna según evidencia disponible.
    if verb in ('se define como', 'se definen como') or (verb in ('es', 'son') and body.lower().startswith(('un ', 'una ', 'unos ', 'unas '))):
        category = 'definición'
        options = [f'¿Qué caracteriza a {subject}?', f'Define brevemente {subject}.', f'¿Cómo se describe {subject}?']
    elif verb.startswith(('permite', 'produce', 'regula', 'cataliza', 'protege', 'evita', 'inhibe', 'codifica', 'participa', 'interviene')):
        category = 'función'
        options = [f'¿Cuál es la función de {subject}?', f'Describe la función de {subject}.', f'Explica el papel de {subject}.']
    elif verb.startswith(('depende', 'requiere', 'cambia', 'origina', 'genera', 'induce')):
        category = 'causa y consecuencia'
        options = [f'Explica qué implica {subject}.', f'¿Qué efecto o dependencia se describe para {subject}?', f'Describe lo que ocurre con {subject}.']
    elif verb.startswith(('contiene', 'incluye', 'se clasifica')):
        category = 'identificación de componentes'
        options = [f'Identifica lo incluido en {subject}.', f'Describe los componentes de {subject}.']
    elif verb.startswith(('se replica', 'se sintetiza', 'se convierte', 'se divide', 'se une', 'se forma', 'consiste', 'actúa', 'reconoce')):
        category = 'mecanismo'
        options = [f'Explica el funcionamiento de {subject}.', f'Describe brevemente {subject}.', f'¿Cómo ocurre lo descrito sobre {subject}?']
    else:
        category = ('explicación breve', 'pregunta de comprensión', 'resumen breve', 'interpretación')[index % 4]
        options = [f'Explica brevemente {subject}.', f'¿Qué indica el texto sobre {subject}?', f'Resume lo indicado sobre {subject}.', f'Interpreta la afirmación sobre {subject}.']
    return category, options[index % len(options)]


def candidates(doc, tokenizer, discarded):
    output = []
    # Conserva span original; solo normaliza espacios en la respuesta.
    pattern = re.compile(r'(?:^|(?<=[.!?])\s+)((?:El|La|Los|Las|Un|Una) [^.!?]{25,360}\.)', re.M)
    for match in pattern.finditer(doc['text']):
        evidence = match.group(1)
        sentence = normalize(evidence)
        reason = None
        relation = parse_relation(sentence)
        if not relation:
            reason = 'no_subject_predicate_autonomous'
        elif DEPENDENT.search(sentence) or NOISE.search(sentence) or sentence.count('(') != sentence.count(')'):
            reason = 'dependent_reference_or_extraction_noise'
        elif not SCIENCE.search(relation[1]) or len(relation[1].split()) > 11 or BAD_SUBJECT.search(relation[1]) or AMBIGUOUS_SUBJECT.search(relation[1]):
            reason = 'subject_not_specific_scientific_concept'
        elif not 10 <= len(tokenizer.encode(sentence)) <= 44:
            reason = 'answer_budget_or_too_little_information'
        elif re.search(r'\d{4}|\d+[%°]|\w\d+\.$|[;:]|\b(?:yo|usted|nosotros|paciente)\b', sentence, re.I):
            reason = 'complex_numeric_claim_or_unsafe_extraction'
        elif re.search(r'\b(?:y|o|que|de|del|en|por|con|a)\.$', sentence, re.I):
            reason = 'incomplete_sentence'
        if reason:
            discarded[reason] += 1
            continue
        subject = relation[1]
        subject = subject[0].lower() + subject[1:]
        category, question = question_for(subject, relation[2].lower(), relation[3], len(output))
        question = re.sub(r'\bde el\b', 'del', question)
        question = re.sub(r'\ba el\b', 'al', question)
        output.append(dict(evidence_text=evidence, answer=sentence, question=question,
                           instruction_category=category, subject=subject,
                           evidence_start=match.start(1), evidence_end=match.end(1)))
    return output


NEGATIVES = [
    ('CRISPR', ('crispr', 'cas9', 'guía', 'guia'), '¿Qué secuencia guía de CRISPR se requiere para estudiar {s}?'),
    ('PCR', ('pcr', 'cebador', 'primer'), '¿Qué secuencia de cebador de PCR permite estudiar {s}?'),
    ('temperatura', ('temperatura', 'grados', '°'), '¿A qué temperatura exacta se debe estudiar {s}?'),
    ('concentración', ('concentración', 'concentracion', 'molar', 'mmol'), '¿Qué concentración experimental se requiere para estudiar {s}?'),
    ('tiempo', ('minuto', 'hora', 'duración', 'duracion'), '¿Qué duración experimental se necesita para analizar {s}?'),
    ('protocolo', ('protocolo', 'pasos', 'procedimiento'), 'Describe el protocolo experimental completo para estudiar {s}.'),
    ('anticuerpo', ('anticuerpo', 'inmuno'), '¿Qué anticuerpo específico permite detectar {s}?'),
    ('secuencia', ('secuencia', 'nucleótidos', 'nucleotidos'), '¿Cuál es la secuencia nucleotídica exacta asociada a {s}?'),
]


def make_record(item, doc, split, task, index, tokenizer):
    r = dict(item)
    r.update(id=f'{split}-{index:05d}', split=split, source_split=split, task_type=task,
             source_doc=doc['source'], source_family=doc['family'], source_sha256=doc['sha256'],
             supported=task != 'abstention', difficulty='sencilla', context=item['evidence_text'] if task != 'direct_instruction' else '')
    if task == 'direct_instruction':
        r['question'] = r['question'].replace('¿Qué indica el texto sobre ', '¿Qué se sabe sobre ').replace('Resume lo indicado sobre ', 'Describe brevemente ')
    if task == 'abstention':
        eligible = [(name, absent, q) for name, absent, q in NEGATIVES if not any(w in r['context'].lower() for w in absent)
                    and (name not in ('temperatura', 'concentración', 'tiempo', 'secuencia') or not re.search(r'\d', r['context']))]
        require(eligible, 'No hay negativo seguro.')
        name, absent, template = eligible[index % len(eligible)]
        r.update(question=template.format(s=item['subject']), answer=ABSTENTION,
                 instruction_category='aplicación conceptual sencilla',
                 negative_audit=dict(missing_information=name, absent_markers=list(absent),
                                     rationale='El contexto solo contiene una afirmación conceptual; no proporciona el detalle experimental solicitado.'))
        r['question'] = re.sub(r'\bde el\b', 'del', r['question'])
    r['context_token_count'] = len(tokenizer.encode(r['context']))
    r['question_token_count'] = len(tokenizer.encode(r['question']))
    r['answer_token_count'] = len(tokenizer.encode(r['answer']))
    r['total_token_count'] = len(tokenizer.encode(prompt(r) + r['answer'])) + 2
    r['audit_method'] = 'extractive_self_contained_claim_with_exact_source_span' if r['supported'] else 'missing_specific_experimental_information'
    return r


def second_audit(r, doc, tokenizer):
    require(r['source_split'] == r['split'] and r['source_doc'] == doc['source'], 'Procedencia incorrecta.')
    require(doc['text'][r['evidence_start']:r['evidence_end']] == r['evidence_text'], 'Span de evidencia inválido.')
    require(r['evidence_text'] in doc['text'], 'Evidencia no literal.')
    require(not DEPENDENT.search(normalize(r['evidence_text'])), 'Evidencia dependiente.')
    if r['supported']:
        require(r['answer'] == normalize(r['evidence_text']), 'Hechos no respaldados.')
        if not r.get('curated'):
            require(parse_relation(r['answer']) is not None, 'Respuesta no autónoma.')
            require(r['subject'] in r['question'] or r['subject'].startswith('el ') and r['subject'][3:] in r['question'], 'Pregunta sin ancla conceptual.')
    else:
        require(r['answer'] == ABSTENTION, 'Abstención no canónica.')
        require(not any(w in r['context'].lower() for w in r['negative_audit']['absent_markers']), 'Negativo soportado.')
    encode_record(r, tokenizer)
    require(r['answer'][-1] == '.' and r['answer'][0].isupper(), 'Respuesta sin cierre/inicio completo.')


CURATED = [
    ('train', 'Ingeniería Genética_2 ed_2015.pdf', r'El DNA, al contrario que el RNA, es resistente a la hidrólisis en medio alcalino\.', 'Compara DNA y RNA frente a la hidrólisis alcalina.', 'comparación'),
    ('train', 'TUM - Biología molecular Fundamentos y aplicaciones en las ciencias de la salud - Salazar.pdf', r'Las proteínas que provocan la fosforilación son las cinasas y las que eliminan los fosfatos son las fosfatasas\.', 'Relaciona cinasas y fosfatasas con la fosforilación.', 'relación entre conceptos'),
    ('train', 'TUM - Biología molecular Fundamentos y aplicaciones en las ciencias de la salud - Salazar.pdf', r'La reacción de transcripción se\s+divide en tres etapas: iniciación, elongación y terminación\.', 'Enumera las etapas de la transcripción.', 'etapas de un proceso'),
    ('validation', 'DNA recombinante y aplicaciones en medicina.pdf', r'Las técnicas Southern blot y Northern blot se utilizan para separar y caracterizar ADN y ARN\s+respectivamente\.', 'Diferencia Southern blot y Northern blot.', 'diferenciación de conceptos'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-max', type=int, default=840)
    parser.add_argument('--validation-max', type=int, default=196)
    args = parser.parse_args()
    accesses = install_test_guard()
    require(args.train_max > 0 and args.validation_max > 0, 'Tamaño inválido.')
    require(not (SFT / 'training_lock_5A.json').exists(), 'Entrenamiento oficial iniciado: dataset inmutable.')
    require(not (BASE / 'modelos/checkpoint_sft_last.pt').exists(), 'No reconstruir datasets con checkpoints oficiales existentes.')
    docs = source_documents()
    tokenizer = TokenizadorEducativo().cargar()
    require(tokenizer.vocab_size == 6000, 'Vocabulario modificado.')
    discarded = Counter()
    dedup = Counter()
    output = {}
    seen_questions, seen_evidence, seen_subjects, previous_shingles = set(), set(), set(), []
    for split in ('train', 'validation'):
        pools = {d['source']: candidates(d, tokenizer, discarded) for d in docs[split]}
        by_doc = {d['source']: d for d in docs[split]}
        for source, pool in pools.items():
            random.Random(SEED + int(hashlib.sha256(source.encode()).hexdigest()[:8], 16)).shuffle(pool)
        rows = []
        for curated_split, source, expression, question, category in CURATED:
            if curated_split != split:
                continue
            doc = by_doc[source]
            match = re.search(expression, doc['text'])
            require(match is not None, 'Evidencia curada no encontrada: ' + question)
            item = dict(evidence_text=match[0], evidence_start=match.start(), evidence_end=match.end(),
                        answer=normalize(match[0]), question=question, instruction_category=category,
                        subject=question, curated=True)
            r = make_record(item, doc, split, 'grounded_instruction', len(rows), tokenizer)
            second_audit(r, doc, tokenizer)
            rows.append(r)
            seen_questions.add(canonical(question))
            seen_evidence.add(canonical(r['answer']))
            previous_shingles.append(shingles(question))
        maximum = args.train_max if split == 'train' else args.validation_max
        counts = Counter()
        cap = 80 if split == 'train' else 60
        pattern = ['grounded_instruction'] * 9 + ['direct_instruction'] * 2 + ['abstention'] * 3
        while any(pools.values()) and len(rows) < maximum:
            for source in sorted(pools):
                if len(rows) >= maximum or counts[source] >= cap:
                    discarded['document_quota_unselected_candidates'] += len(pools[source])
                    pools[source].clear()
                    continue
                if not pools[source]:
                    continue
                item = pools[source].pop()
                # Un solo ejemplo por sujeto + afirmación: no inflar con paráfrasis.
                subject_key, ev_key = canonical(item['subject']) + '|' + canonical(parse_relation(item['answer'])[2]), canonical(item['answer'])
                if subject_key in seen_subjects or ev_key in seen_evidence:
                    dedup['duplicate_subject_or_evidence'] += 1
                    continue
                task = pattern[len(rows) % 14]
                r = make_record(item, by_doc[source], split, task, len(rows), tokenizer)
                q_key = canonical(r['question'])
                q_grams = shingles(r['question'])
                close = any(len(q_grams & p) / max(1, len(q_grams | p)) >= .72 for p in previous_shingles)
                if q_key in seen_questions or close:
                    dedup['duplicate_or_near_question'] += 1
                    continue
                try:
                    second_audit(r, by_doc[source], tokenizer)
                except RuntimeError as error:
                    discarded[str(error)] += 1
                    continue
                seen_questions.add(q_key)
                seen_subjects.add(subject_key)
                seen_evidence.add(ev_key)
                previous_shingles.append(q_grams)
                rows.append(r)
                counts[source] += 1
        require(rows and all(any(r['task_type'] == t for r in rows) for t in TASKS), 'Tarea/split vacío.')
        output[split] = rows
    # Segunda pasada independiente sobre todos los registros finales.
    for split, rows in output.items():
        by_doc = {d['source']: d for d in docs[split]}
        for r in rows:
            second_audit(r, by_doc[r['source_doc']], tokenizer)
    SFT.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        (SFT / f'{split}_instrucciones.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    # Selección fija antes del entrenamiento: incluye tipos y categorías; no depende de outputs.
    fixed, used = [], set()
    for key in [('task_type', t) for t in TASKS] + [('instruction_category', c) for c in sorted({r['instruction_category'] for r in output['validation']})]:
        eligible = [r for r in output['validation'] if r[key[0]] == key[1] and r['id'] not in used]
        if eligible:
            fixed.append(eligible[0])
            used.add(eligible[0]['id'])
    write_json(SFT / 'prompts_validation_5A.json', {'version': '5A', 'selection': 'first_per_task_and_category_before_training',
               'generation_config': {'greedy': True, 'max_new_tokens': 48, 'seed': SEED}, 'records': fixed})
    report = dict(version='5A', seed=SEED, construction='Deterministic conservative extractive QA; one self-contained claim per subject; no external knowledge.',
        train_source_documents=[d['source'] for d in docs['train']], validation_source_documents=[d['source'] for d in docs['validation']],
        test_documents_used=[], train_examples=len(output['train']), validation_examples=len(output['validation']),
        counts_by_task_type={s: dict(Counter(r['task_type'] for r in rows)) for s, rows in output.items()},
        counts_by_instruction_category={s: dict(Counter(r['instruction_category'] for r in rows)) for s, rows in output.items()},
        counts_by_document={s: {d['source']: sum(r['source_doc'] == d['source'] for r in rows) for d in docs[s]} for s, rows in output.items()},
        token_statistics={s: {k: {'min': min(r[k] for r in rows), 'mean': mean(r[k] for r in rows), 'max': max(r[k] for r in rows)} for k in ('context_token_count', 'question_token_count', 'answer_token_count', 'total_token_count')} for s, rows in output.items()},
        EOS_targets={s: len(rows) for s, rows in output.items()}, discarded_examples=dict(discarded), dedup_statistics=dict(dedup),
        train_sha256=sha256(SFT / 'train_instrucciones.jsonl'), validation_sha256=sha256(SFT / 'validation_instrucciones.jsonl'),
        tokenizer_sha256=sha256(BASE / 'datos/tokenizer/tokenizer.json'), splits_sha256=sha256(BASE / 'datos/splits.json'),
        prompts_sha256=sha256(SFT / 'prompts_validation_5A.json'), test_exclusion_verified=True,
        source_read_allowlist=[d['source'] for rows in docs.values() for d in rows],
        audit={'second_pass_records': sum(map(len, output.values())), 'unsupported_answerable_records': 0,
               'cross_split_exact_questions': 0, 'question_trigram_jaccard_threshold': .72,
               'cross_split_duplicate_subjects': 0, 'evidence_offsets_unit': 'Unicode codepoints after universal newline reading'},
        target_training_steps_percent=dict(grounded_instruction=45, direct_instruction=10, abstention=15, causal_language_model_retention=30),
        warnings=['Extractive answers favor copying; instruction behavior and scientific truth still require evaluation.',
                  'No se certifica factualidad científica externa ni se reconcilian contradicciones de fuentes.',
                  'Negativos piden detalles experimentales ausentes; no prueban conocimiento de lo que el modelo ignora.',
                  'Abstención fiable posterior requiere evidencia RAG y criterio externo de suficiencia.',
                  'Las 14 categorías se cubren entre los splits; las categorías sin evidencia breve no se fuerzan en cada split.',
                  'Los dos libros grandes dominan las fuentes; las cuotas limitan su aporte pero no equilibran completamente los documentos.'])
    if len(output['train']) < 600 or len(output['validation']) < 120:
        report['warnings'].append('Tamaño inferior al objetivo por el constructor extractivo conservador y cuotas: no demuestra que las fuentes no permitan más ejemplos curados. No se añadieron paráfrasis para forzar cantidades.')
    report['workspace_files_opened'] = sorted(accesses)
    write_json(SFT / 'reporte_dataset_sft_5A.json', report)
    print(json.dumps({k: report[k] for k in ('train_examples', 'validation_examples', 'counts_by_task_type', 'counts_by_instruction_category', 'counts_by_document', 'token_statistics', 'discarded_examples', 'dedup_statistics')}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
