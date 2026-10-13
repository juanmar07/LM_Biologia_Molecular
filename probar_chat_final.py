"""Smoke tests de presentación y CLI; sólo las cuatro consultas manuales, sin Test."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import Mock, patch
import chat_final as cli

BASE = Path(__file__).resolve().parent
REPORT = BASE / 'resultados/verificacion_chat_final.json'


class PresentationTests(unittest.TestCase):
    def row(self, decision='EXTRACTIVE_FALLBACK'):
        return dict(question='¿Pregunta con tildes y α?', final_answer='Una oración\n con   espacios.',
                    source_doc=r'C:\documentos\Fuente con tildes.pdf', decision=decision,
                    abstention_used=decision == 'ABSTAIN', generator_candidate='candidato interno',
                    generator_accepted=decision == 'GENERATOR_ACCEPTED', fallback_used=decision == 'EXTRACTIVE_FALLBACK',
                    chunk_id='chunk-interno', BM25_score=2, rerank_score=3, support_metrics={'repetition': 0},
                    packing={'context': 'contexto original'}, generation={'EOS': True, 'stop_reason': 'EOS'})

    def invoke(self, entries=(), *, args=None, system=None, load_error=None):
        system = system if system is not None else Mock(model_metadata={'checkpoint': 'ruta interna'})
        if system.answer.side_effect is None and not isinstance(system.answer.return_value, dict):
            system.answer.return_value = self.row()
        out, err = io.StringIO(), io.StringIO()
        with patch.object(cli, 'configure_console'), patch.object(cli, 'load_system', return_value=system, side_effect=load_error) as load:
            with patch('builtins.input', side_effect=list(entries)), redirect_stdout(out), redirect_stderr(err):
                code = cli.main(args or ['--interactive'])
        return code, out.getvalue(), err.getvalue(), system, load

    def test_all_response_modes_preserve_raw_result(self):
        for mode in ('GENERATOR_ACCEPTED', 'EXTRACTIVE_FALLBACK', 'ABSTAIN'):
            with self.subTest(mode=mode):
                row = self.row(mode); before = copy.deepcopy(row); out = io.StringIO()
                with redirect_stdout(out): cli.show(row)
                self.assertEqual(row, before)
                self.assertIn('MiniGPT: Una oración con espacios.', out.getvalue())
                self.assertEqual('Fuente:' in out.getvalue(), mode != 'ABSTAIN')
                for internal in (mode, 'candidato interno', 'chunk-interno', 'C:\\documentos', 'BM25', 'rerank'):
                    self.assertNotIn(internal, out.getvalue())

    def test_sources_only_from_final_provenance(self):
        row = self.row(); row['source_doc'] = ['/datos/A.pdf', r'C:\docs\B.pdf', '/otro/A.pdf']
        row['retrieval'] = [{'source_doc': 'No utilizada.pdf'}]
        self.assertEqual(cli.displayed_sources(row), ['A.pdf', 'B.pdf'])
        row['decision'] = 'ABSTAIN'
        self.assertEqual(cli.displayed_sources(row), [])

    def test_empty_questions_and_single_question_trim(self):
        code, out, err, system, load = self.invoke(['', '   ', '/salir'])
        self.assertEqual(code, 0); system.answer.assert_not_called(); load.assert_called_once()
        code, out, err, system, load = self.invoke(args=['--pregunta', '   '])
        self.assertEqual(code, 2); load.assert_not_called(); self.assertIn('no vacía', err)
        code, out, err, system, load = self.invoke(args=['--pregunta', '  ¿Pregunta con tildes y α?  '])
        self.assertEqual(code, 0); system.answer.assert_called_once_with('¿Pregunta con tildes y α?')

    def test_help_clear_and_unknown_command_keep_session(self):
        with patch.object(cli.os, 'system') as clear:
            code, out, err, system, load = self.invoke(['/ayuda', '/limpiar', '/debug quiz', '/salir'])
        self.assertEqual(code, 0); load.assert_called_once(); system.answer.assert_not_called()
        self.assertIn('Cada pregunta se procesa de manera independiente.', out)
        self.assertIn('Comando no reconocido', out); self.assertEqual(out.count('Asistente de Biología Molecular'), 2)
        clear.assert_not_called()  # Salida redirigida: no ejecuta cls ni códigos de terminal.

    def test_clear_in_terminal(self):
        with patch.object(cli.sys.stdout, 'isatty', return_value=True), patch.object(cli.os, 'system') as clear, patch.object(cli, 'header') as header:
            cli.clear_screen()
        clear.assert_called_once_with('cls' if cli.os.name == 'nt' else 'clear'); header.assert_called_once()

    def test_debug_toggles_full_trace_without_changing_answer(self):
        code, out, err, system, load = self.invoke(['/debug on', 'primera', '/debug off', 'segunda', '/exit'])
        self.assertEqual(code, 0); load.assert_called_once(); self.assertEqual(system.answer.call_count, 2)
        first, second = out.split('Información técnica desactivada.')
        for field in ('decision', 'generator_candidate', 'generator_accepted', 'fallback_used', 'abstention_used',
                      'chunk_id', 'source_doc', 'BM25_score', 'rerank_score', 'support_metrics', 'packing', 'EOS', 'stop_reason', 'repetition'):
            self.assertIn('"' + field + '"', first)
            self.assertNotIn('"' + field + '"', second)
        self.assertEqual(out.count('MiniGPT: Una oración con espacios.'), 2)

    def test_exit_aliases_and_eof(self):
        for entry in ('/salir', '/exit', 'salir', 'exit', EOFError()):
            with self.subTest(entry=entry):
                code, out, err, system, load = self.invoke([entry])
                self.assertEqual(code, 0); self.assertIn('Hasta luego.', out); self.assertNotIn('Traceback', err)

    def test_interrupt_during_input_loading_and_answer(self):
        scenarios = [dict(entries=[KeyboardInterrupt()]), dict(load_error=KeyboardInterrupt()),
                     dict(entries=['pregunta'], system=Mock(model_metadata={}, answer=Mock(side_effect=KeyboardInterrupt())))]
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                code, out, err, system, load = self.invoke(**scenario)
                self.assertEqual(code, 0); self.assertIn('Hasta luego.', out); self.assertNotIn('Traceback', err)

    def test_expected_error_recovers_with_same_system(self):
        for debug in (False, True):
            with self.subTest(debug=debug):
                system = Mock(model_metadata={}, answer=Mock(side_effect=[ValueError('detalle técnico'), self.row()]))
                code, out, err, system, load = self.invoke(['primera', 'segunda', '/salir'], args=['--interactive'] + (['--debug'] if debug else []), system=system)
                self.assertEqual(code, 0); load.assert_called_once(); self.assertEqual(system.answer.call_count, 2)
                self.assertIn('No se pudo procesar', err); self.assertIn('MiniGPT:', out)
                self.assertEqual('Traceback' in err, debug); self.assertEqual('detalle técnico' in err, debug)

    def test_load_failure_and_single_question_failure(self):
        for debug in (False, True):
            code, out, err, system, load = self.invoke(args=['--interactive'] + (['--debug'] if debug else []), load_error=FileNotFoundError('ruta interna'))
            self.assertEqual(code, 1); self.assertIn('No se pudo cargar', err); self.assertEqual('Traceback' in err, debug)
        system = Mock(model_metadata={}, answer=Mock(side_effect=RuntimeError('fallo interno')))
        code, out, err, system, load = self.invoke(args=['--pregunta', 'pregunta'], system=system)
        self.assertEqual(code, 1); self.assertNotIn('fallo interno', err)

    def test_internal_logs_and_warnings_only_in_debug(self):
        for debug in (False, True):
            out = io.StringIO(); err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                with cli.runtime_output(debug):
                    print('ruta interna'); print('mensaje interno', file=sys.stderr)
                    cli.warnings.warn('warning interno', UserWarning)
            for text in ('ruta interna', 'mensaje interno', 'warning interno'):
                self.assertEqual(text in out.getvalue(), debug)
            self.assertEqual(err.getvalue(), '')

    def test_session_opt_in_unicode_timestamp_and_no_debug(self):
        with tempfile.TemporaryDirectory() as temp:
            session = cli.SessionLog(); session.path = Path(temp) / 'sesion.jsonl'
            with patch.object(cli, 'SessionLog', return_value=session) as factory:
                self.invoke(['pregunta', '/salir']); factory.assert_not_called()
                self.assertFalse(session.path.exists())
                code, out, err, _, _ = self.invoke(['pregunta', '/salir'], args=['--interactive', '--debug', '--guardar-sesion'])
                self.assertEqual(code, 0)
            saved = json.loads(session.path.read_text(encoding='utf-8'))
            self.assertEqual(set(saved), {'timestamp', 'pregunta', 'respuesta', 'fuentes'})
            self.assertEqual(saved['respuesta'], 'Una oración con espacios.'); self.assertEqual(saved['fuentes'], ['Fuente con tildes.pdf'])
            self.assertIsNotNone(cli.datetime.fromisoformat(saved['timestamp']).tzinfo)
            session.save(self.row('ABSTAIN'))
            self.assertEqual(json.loads(session.path.read_text(encoding='utf-8').splitlines()[1])['fuentes'], [])

    def test_session_write_failure_does_not_break_chat(self):
        with patch.object(cli, 'SessionLog', return_value=Mock(save=Mock(side_effect=OSError('disco')))) as factory:
            code, out, err, system, load = self.invoke(['primera', 'segunda', '/salir'], args=['--interactive', '--guardar-sesion'])
        self.assertEqual(code, 0); self.assertEqual(system.answer.call_count, 2)
        factory.return_value.save.assert_called_once(); self.assertIn('se desactivó el registro', err)


def real_smoke():
    import seguridad_5C_common as integrity
    from rag_common import LOG, FORBIDDEN
    from retrieval import load_index
    from contexto_rag_5B1 import source_lookup, neighbor_lookup
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    questions = [r['question'] for r in report['rows_before']]
    system = cli.load_system()
    def memory_hash():
        digest = hashlib.sha256()
        for name, value in sorted(system.model.state_dict().items()):
            digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    before_memory = memory_hash()
    rows = [system.answer(q) for q in questions]
    assert rows == report['rows_before'], 'El resultado completo del pipeline cambió'
    assert before_memory == memory_hash(), 'Pesos en memoria modificados'
    assert all(cache.cache_info().misses == 1 for cache in (load_index, source_lookup, neighbor_lookup))
    checks = dict(full_pipeline_rows_unchanged=True, model_memory_unchanged=True, resources_cached_once=True)
    def command(args, input_text=None, expected=0):
        proc = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(BASE/'chat_final.py'), *args],
                              input=input_text, capture_output=True, text=True, encoding='utf-8', cwd=BASE, timeout=90)
        assert proc.returncode == expected, (proc.returncode, proc.stderr)
        assert not proc.stderr, proc.stderr
        return proc.stdout
    normal = command(['--interactive'], '\n  \n/ayuda\n/limpiar\n' + '\n'.join(questions) + '\n/salir\n')
    assert normal.count('Cargando MiniGPT...') == 1 and normal.count('Listo.') == 1
    assert normal.count('MiniGPT:') == 4 and normal.count('Fuente:') == 3
    for row in rows:
        assert 'MiniGPT: ' + cli.compact(row['final_answer']) in normal
    for hidden in ('EXTRACTIVE_FALLBACK', 'ABSTAIN', 'generator_candidate', 'BM25', 'rerank', 'checkpoint', 'chunk_id'):
        assert hidden not in normal
    debug = command(['--interactive', '--debug'], '\n'.join(questions) + '\n/exit\n')
    decoded = [json.JSONDecoder().raw_decode(part)[0] for part in debug.split('Debug: ')[1:]]
    assert decoded == rows, 'Debug alteró la trazabilidad'
    single = command(['--pregunta', '  ' + questions[0] + '  '])
    assert 'Tú: ' + questions[0] in single and cli.compact(rows[0]['final_answer']) in single
    assert 'Hasta luego.' in command(['--interactive'], '')
    checks.update(real_interactive_commands=True, normal_output_clean=True, real_debug_full_rows_unchanged=True,
                  single_question_unicode_trim=True, real_eof_clean=True)
    # Verificar también el archivo real opt-in, sin dejar un turno sintético.
    sessions_dir = BASE/'resultados/sesiones'
    existing = set(sessions_dir.glob('*.jsonl'))
    command(['--pregunta', questions[-1], '--guardar-sesion'])
    created = set(sessions_dir.glob('*.jsonl')) - existing
    assert len(created) == 1
    session_path = created.pop(); saved = json.loads(session_path.read_text(encoding='utf-8'))
    assert saved['fuentes'] == [] and saved['respuesta'] == rows[-1]['final_answer']
    assert set(saved) == {'timestamp', 'pregunta', 'respuesta', 'fuentes'}
    checks['real_session_log'] = True
    report.update(rows_after=rows, pipeline_identical_before_after=True, integration_checks=checks,
                  memory_hash_before=before_memory, memory_hash_after=memory_hash(),
                  central_hashes_after={p:integrity.sha256(BASE/p) for p in report['central_hashes_before']},
                  after=integrity.preservation(), session_smoke_file=session_path.relative_to(BASE).as_posix(),
                  test_content_reads=sum(r['allowed'] and r['path'] in FORBIDDEN for r in LOG))
    assert report['central_hashes_before'] == report['central_hashes_after']
    assert report['test_content_reads'] == 0
    checks.update(central_hashes_unchanged=True, protected_247_files_preserved=True, test_content_reads_zero=True)
    integrity.write_json(REPORT, report)
    print(json.dumps(dict(integration_checks=checks, manual=[{k:r[k] for k in ('question','decision','final_answer','source_doc')} for r in rows]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PresentationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): sys.exit(1)
    real_smoke()
    data = json.loads(REPORT.read_text(encoding='utf-8'))
    data['presentation_tests'] = dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors))
    REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
