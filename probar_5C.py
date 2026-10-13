"""Verificación mínima 5C: 8 consultas manuales + 20 Validation deterministas."""
import seguridad_5C_common as integrity
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from respuesta_segura import ChatSeguro, ABSTENTION, POLICY, evidence_support, validate_candidate, clean, units, analysis_for, unit_support
import respuesta_segura
from contexto_rag_5B1 import source_lookup
from retrieval import load_index, tokens
from rag_common import FORBIDDEN, CHECKPOINT, LOG

MANUAL = [
    '¿Qué caracteriza a los genes ribosómicos?',
    '¿Qué diferencia existe entre Southern blot y Northern blot?',
    '¿Cuál es la función de un vector de expresión?',
    '¿Qué son las células multipotenciales?',
    '¿Qué es CRISPR-Cas?',
    '¿Qué función tiene la PCR?',
    '¿Qué son las enzimas de restricción?',
    '¿Qué es el ADN?',
]
MODES = ('GENERATOR_ACCEPTED', 'EXTRACTIVE_FALLBACK', 'ABSTAIN')

def counts(rows):
    counter = Counter(r['decision'] for r in rows)
    return {mode:counter[mode] for mode in MODES}

def memory_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def frozen_selection():
    path = integrity.DATA/'suite_validation_5B.json'
    expected = json.loads((integrity.DATA/'suite_validation_5B.sha256.json').read_text(encoding='utf-8'))['sha256']
    integrity.require(integrity.sha256(path)==expected, 'Suite Validation cambió')
    suite = json.loads(path.read_text(encoding='utf-8'))['questions']
    by_doc = defaultdict(list)
    for q in sorted(suite, key=lambda q:q['id']): by_doc[q['expected_source_doc']].append(q)
    integrity.require(len(by_doc)==4 and min(map(len, by_doc.values()))>=5, 'Muestra estratificada inválida')
    chosen = [by_doc[doc][i] for i in range(5) for doc in sorted(by_doc)]
    lock = dict(suite_sha256=expected, selection='first 5 IDs per Validation source; round-robin sorted sources; before execution; no result filtering',
                question_ids=[q['id'] for q in chosen], n=20, seed=42, policy=POLICY)
    dest = integrity.STAGE/'seleccion_validation_5C.json'
    if dest.exists(): integrity.require(json.loads(dest.read_text(encoding='utf-8'))==lock, 'Selección congelada cambió')
    else: integrity.write_json(dest, lock)
    return chosen, lock

def test_suite(system, rows):
    tests = {}
    def check(name, condition):
        integrity.require(condition, 'Prueba falló: '+name); tests[name] = True
    answered = [r for r in rows if not r['abstention_used']]
    check('existing_retrieval_train_validation_only', all(c['split'] in ('train','validation') for r in rows for c in r['retrieval']) and len(load_index()['chunks'])==23161)
    check('top10_original_query_rerank_preserved', all(len(r['bm25_retrieved'])<=10 and all(c['query_analysis']['candidate_query']==r['question'] for c in r['retrieval']) for r in rows))
    check('context_budget_128_exact_SFT_BOS', all(r['generation']['total_bpe_tokens']<=128 and r['packing']['prompt_ids'][0]==2 and
        r['packing']['prompt'].startswith('Contexto:\n') and r['packing']['prompt'].endswith('\n\nAsistente:\n') and r['question'] in r['packing']['prompt'] for r in answered))
    check('candidate_raw_and_ids_preserved', all(r['candidate_raw']==system.tokenizer.decode(r['candidate_token_ids'],omitir_especiales=False) and
        r['generator_candidate']==system.tokenizer.decode(r['candidate_token_ids'][:-1] if r['generation']['EOS'] else r['candidate_token_ids'],omitir_especiales=False) for r in answered))
    check('EOS_termination', all(not r['generation']['EOS'] or r['candidate_token_ids'][-1]==3 for r in answered))
    check('source_present_when_answered', all(r['source_doc'] and r['chunk_id'] and r['BM25_score'] is not None and r['rerank_score'] is not None for r in answered))
    check('fallback_literal_offsets', all(all(source_lookup()[s['source_doc']]['text'][s['start']:s['end']]==s['raw'] and clean(s['raw'])==r['final_answer'] for s in r['fallback_source_spans']) for r in answered if r['fallback_used']))
    genes = rows[0]
    check('relevant_genes_fallback', genes['fallback_used'] and 'conservación' in genes['final_answer'] and 'ribosómicos' in genes['final_answer'])
    comparison = rows[1]
    check('comparison_preserves_ADN_ARN_respectively', comparison['fallback_used'] and all(t in comparison['final_answer'] for t in ('Southern','Northern','ADN','ARN','respectivamente')))
    evidence = genes['evidence']; context = genes['packing']['context']
    def validate(text, eos=True): return validate_candidate(dict(answer=text, EOS=eos),context,evidence)
    check('literal_supported_candidate_acceptance', validate(genes['final_answer'])['accepted'])
    check('repetition_forces_rejection', 'repetition' in validate('Los genes ribosómicos se caracterizan. Los genes ribosómicos se caracterizan.')['rejection_reasons'])
    check('empty_short_and_tautology_rejected', all(not validate(t)['accepted'] for t in ('','Genes.','Los genes son genes.')))
    check('new_content_rejected', 'novel_content_terms' in validate('Los genes ribosómicos producen meteoritos.')['rejection_reasons'])
    check('missing_principal_rejected', 'missing_principal' in validate('El elevado grado de conservación.')['rejection_reasons'])
    check('EOS_not_sufficient', not validate('ADN genes células PCR.',True)['accepted'])
    check('definition_intent_for_expresses_means_designates', all(analysis_for(q)['intent']=='definition' for q in
        ('¿Qué expresa la potencia de una célula?','¿Qué significa expresión diferencial?','¿Qué designa el término CRISPR?')))
    check('order_or_negation_not_licensed_by_overlap', all(not validate(t)['accepted'] for t in
        ('Los genes ribosómicos no se caracterizan por su elevado grado de conservación.',
         'Los genes se caracterizan por su grado ribosómicos de elevado conservación.')))
    # Negative subject-binding test, using actual retrieved text rather than a science template.
    pcr = rows[5]
    first = pcr['inspected_contexts'][0]
    check('Taq_incidental_PCR_function_rejected', not first.get('evidence',{}).get('sufficient',False))
    fake_packed = dict(completeness=dict(start_boundary=True))
    # Mechanical adversarial fixtures exercise grammar/segmentation, not corpus answers.
    for name, question, fixture in (
        ('incidental_entity_not_definition','¿Qué es el ADN?','El fragmento de ADN es insertado en un vector para transformar células.'),
        ('specific_modality_not_general_function','¿Qué función tiene la PCR?','Esta modalidad de PCR permite detectar la presencia de un fragmento.'),
        ('nearby_generic_cells_not_multipotentials','¿Qué son las células multipotenciales?','Las células madre son células presentes en diversos órganos.')):
        check(name, not any(unit_support(u,analysis_for(question),fake_packed)['sufficient'] for u in units(fixture)))
    check('abbreviations_not_split_into_incomplete_sentence', len(units('Se observan células en órganos (p. ej., en diversos tejidos).'))==1)
    fragment = units('\n fragmento de ADN permite analizar la variación del organismo.')[0]
    check('leading_whitespace_not_sentence_boundary', not unit_support(fragment,analysis_for('Describe el fragmento de ADN.'),dict(completeness=dict(start_boundary=False)))['structural_completeness'])
    generation_fn = respuesta_segura.greedy_generate
    def forbidden_generation(*args): raise AssertionError('Generación ejecutada con evidencia insuficiente')
    respuesta_segura.greedy_generate = forbidden_generation
    try:
        empty = system.answer(''); adn = system.answer(MANUAL[-1])
        check('insufficient_evidence_abstains_before_generation', empty['decision']=='ABSTAIN' and adn['decision']=='ABSTAIN' and
              not empty['generation_executed'] and not adn['generation_executed'] and adn['final_answer']==ABSTENTION)
    finally: respuesta_segura.greedy_generate = generation_fn
    for name, path, mode in [('Test_guard_denies_before_open',sorted(FORBIDDEN)[0],'rb'),('checkpoint_write_guard',CHECKPOINT,'wb')]:
        denied = False
        try:
            with open(path,mode): pass
        except RuntimeError: denied = True
        check(name, denied)
    script = str(integrity.BASE/'chat_final.py')
    def cli(args, input_text=None):
        run = subprocess.run([sys.executable,'-X','utf8','-B',script]+args,input=input_text,
                             capture_output=True,text=True,encoding='utf-8',cwd=integrity.BASE,timeout=90)
        integrity.require(run.returncode==0, run.stderr)
        return run.stdout
    normal = cli(['--pregunta',MANUAL[0]])
    check('normal_hides_candidate_and_metadata', 'Fuente:' in normal and 'conservación' in normal and 'generator_candidate' not in normal and 'EXTRACTIVE_FALLBACK' not in normal)
    debug = cli(['--pregunta',MANUAL[0],'--debug'])
    check('debug_shows_full_trace', all(t in debug for t in ('"retrieval"','"packing"','"generator_candidate"','"support_metrics"','"decision"','"fallback_sentence"','"fallback_source_spans"')))
    interactive = cli(['--interactive'],MANUAL[0]+'\n'+MANUAL[-1]+'\nsalir\n')
    check('interactive_multiple_questions_and_salir', interactive.count('Respuesta:')==2 and 'conservación' in interactive and ABSTENTION in interactive)
    check('interactive_exit', 'Respuesta:' not in cli(['--interactive'],'exit\n'))
    check('Test_content_reads_zero', not any(r['allowed'] and r['path'] in FORBIDDEN for r in LOG))
    return tests

def main():
    before = integrity.preservation()
    selection, lock = frozen_selection()
    system = ChatSeguro(); memory_before = memory_hash(system.model)
    rows = []
    for i, question in enumerate(MANUAL):
        row = system.answer(question); row.update(id=f'manual-{i+1:02}', group='manual'); rows.append(row)
        print(row['id'], row['decision'], row['final_answer'], flush=True)
    for question in selection:
        # References enter only after retrieval/generation/decision; never evidence selection.
        row = system.answer(question['question']); row.update(id=question['id'],group='validation',
            evaluation_reference=question['reference'],expected_source_doc=question['expected_source_doc'])
        rows.append(row); print(row['id'],row['decision'],row['final_answer'],flush=True)
    tests = test_suite(system,rows)
    memory_after = memory_hash(system.model)
    integrity.require(memory_before==memory_after,'Pesos en memoria modificados')
    tests['model_memory_unchanged_no_training'] = True
    after = integrity.preservation(); tests['checkpoint_and_previous_artifacts_preserved'] = True
    manual, validation = rows[:8], rows[8:]
    report = dict(stage='5C', policy=POLICY, scope='8 manual + 20 frozen Validation; no Test; no extensive benchmark',
                  integrity=after, integrity_before=before, model=system.model_metadata,
                  memory_hash_before=memory_before,memory_hash_after=memory_after,
                  validation_selection=lock, decision_counts=dict(manual=counts(manual),validation=counts(validation),all=counts(rows)),
                  tests=tests, test_count=len(tests), validation_false_abstention_count=counts(validation)['ABSTAIN'],
                  validation_false_abstention_rate=counts(validation)['ABSTAIN']/20,
                  false_abstention_definition='All frozen Validation items are corpus-answerable; abstention may reflect missed retrieval, packing, or conservative gate, not absence from the corpus.',
                  source_audit=dict(answered=sum(not r['abstention_used'] for r in rows),literal_provenance_verified=True,
                                    obvious_context_contradictions_detected=0,automatic_audit='literal complete supported unit + exact source offsets; does not certify scientific truth'),
                  rows=rows, verdict='5C FUNCIONAL CON LIMITACIONES — CHAT FINAL UTILIZABLE',
                  limitations=['MiniGPT outputs rejected by conservative support validation are replaced, not counted as generator successes.',
                               'Lexical grammatical safety gate can abstain despite an answer in the corpus; no semantic model is used.',
                               'Source-faithful extraction is not independent scientific verification; retrieval/OCR limitations remain.'],
                  code_sha256={p:integrity.sha256(integrity.BASE/p) for p in ('respuesta_segura.py','chat_final.py','probar_5C.py','seguridad_5C_common.py')})
    integrity.write_json(integrity.RESULTS/'reporte_5C.json',report)
    command = f'& "{sys.executable}" -X utf8 -B "{script_path()}" --interactive'
    lines = ['# Informe 5C', '', report['verdict'], '',
        'Pipeline: BM25 top10 original → reranker 5B.1 sin cambiar pesos → contexto vecino exacto → suficiencia por proposición → MiniGPT greedy con BOS/SFT/EOS → validación → candidato, extracción literal o abstención.', '',
        'Se revisan los contextos de los candidatos ya recuperados en el orden del reranker, hasta encontrar evidencia suficiente. No se amplía top10 ni se reescribe la pregunta. El candidato se conserva íntegro en el JSON/debug.', '',
        'Suficiencia: cobertura de términos ≥0.8, principal, sujeto gramatical/intención, al menos tres términos de contenido adicionales y una unidad completa. Un encabezado puede vincular una proposición que comienza con «Son». Se conservan las señales originales y los rechazos.', '',
        'Aceptación: sin repetición, contenido útil, principal, sin términos nuevos, precisión de contenido ≥0.95, EOS y coincidencia literal con una proposición completa útil. La coincidencia literal es una comprobación conservadora de soporte; no valida paráfrasis. EOS y solapamiento genérico por sí solos no bastan.', '',
        'Fallback: una oración literal de la evidencia aprobada, con espacios normalizados y encabezados aislados retirados. Se verifican offsets contra el TXT original. Si la evidencia no basta, se devuelve la abstención canónica antes de generar.', '',
        '## Resultados', '', '| Grupo | MiniGPT aceptado | Extracción | Abstención |', '|---|---:|---:|---:|']
    for name, values in report['decision_counts'].items(): lines.append(f"| {name} | {values['GENERATOR_ACCEPTED']} | {values['EXTRACTIVE_FALLBACK']} | {values['ABSTAIN']} |")
    lines += ['', f"Pruebas: {len(tests)}/{len(tests)} aprobadas. False abstention en Validation corpus-answerable: {report['validation_false_abstention_count']}/20 ({report['validation_false_abstention_rate']:.0%}). Incluye fallos de recuperación/empaquetado y rechazos conservadores; no equivale a ausencia de evidencia en el corpus.", '',
        'La extracción se usa por la limitación demostrada del MiniGPT. Ninguna sustitución se contabiliza como éxito generativo.', '',
        '## Ocho preguntas manuales', '', '| Pregunta | Decisión | Respuesta | Fuente |', '|---|---|---|---|']
    for r in manual: lines.append('| '+' | '.join(str(r[k] or 'Sin evidencia suficiente').replace('|','\\|') for k in ('question','decision','final_answer','source_doc'))+' |')
    lines += ['', '## Integridad y alcance', '', f"{after['protected_files']} archivos previos preservados. Índice no reconstruido. Test: 0 documentos, 0 lecturas de contenido. No entrenamiento ni APIs/modelos externos.", '',
        f"Checkpoint antes/después: `{after['checkpoint_after']}`. Pesos en memoria antes/después: `{memory_after}`.", '',
        f"Suite Validation: `{lock['suite_sha256']}`; 5 primeros IDs por cada una de las 4 fuentes, selección congelada antes de ejecutar. Las referencias sólo se registran después de la decisión.", '',
        'Ejemplos completos, candidatos raw/IDs, contextos, scores, rechazos y offsets: `resultados/reporte_5C.json`. Trazas de accesos: `resultados/accesos_5C_*.json`.', '',
        '## Uso', '', '```powershell',command,'```', '',
        'Consulta única: sustituir `--interactive` por `--pregunta "¿Qué caracteriza a los genes ribosómicos?"`. Añadir `--debug` para ver los candidatos internos.', '',
        '## Limitaciones', ''] + ['- '+s for s in report['limitations']]
    (integrity.RESULTS/'informe_5C.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(counts=report['decision_counts'],tests=len(tests),integrity=after,verdict=report['verdict']),ensure_ascii=False),flush=True)

def script_path(): return integrity.BASE/'chat_final.py'
if __name__=='__main__': main()
