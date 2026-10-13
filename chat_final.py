"""Interfaz CLI de MiniGPT; reutiliza el pipeline local 5C sin modificarlo."""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import traceback
import warnings

SEPARATOR = '=' * 60
EXPECTED_ERRORS = (OSError, ValueError, RuntimeError, ImportError)
HELP = """Este asistente responde utilizando información recuperada de los documentos del proyecto.
Cuando no encuentra evidencia suficientemente clara, lo indica en lugar de inventar una respuesta.
Cada pregunta se procesa de manera independiente.

Comandos:
/ayuda      mostrar esta ayuda
/limpiar    limpiar la pantalla
/debug on   mostrar información técnica
/debug off  ocultar información técnica
/salir      terminar (/exit también está disponible)
"""


def configure_console():
    """UTF-8 también al redirigir entrada/salida en Windows."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')


def compact(text):
    """Normalización exclusivamente visual; conserva palabras y puntuación."""
    return ' '.join(text.split())


def displayed_sources(result):
    if result.get('abstention_used') or result.get('decision') == 'ABSTAIN':
        return []
    # source_doc identifica la procedencia final; retrieval no implica uso.
    sources = result.get('source_doc') or []
    if isinstance(sources, str):
        sources = [sources]
    names = []
    for source in sources:
        name = compact(str(source).replace('\\', '/').rsplit('/', 1)[-1])
        if name and name not in names:
            names.append(name)
    return names


def header():
    print(SEPARATOR)
    print(' MiniGPT — Asistente de Biología Molecular')
    print(SEPARATOR)
    print('Realiza una pregunta sobre los documentos disponibles.')
    print('Escribe /ayuda para ver los comandos o /salir para terminar.\n')


def show(result, debug=False, *, echo_question=True):
    if echo_question:
        print('Tu pregunta:', result['question'])
    print('\nMiniGPT Biologia Molecular:', compact(result['final_answer']))
    if debug:
        sources = displayed_sources(result)
        if sources:
            print('Fuente:', '; '.join(sources))
        print('\nDebug:', json.dumps(result, ensure_ascii=False, indent=2))
    print('\n' + '-' * 60 + '\n')


@contextmanager
def runtime_output(debug):
    """Oculta mensajes internos sin cambiar el resultado ni silenciar errores."""
    output = io.StringIO()
    with warnings.catch_warnings(record=True) as captured:
        try:
            with redirect_stdout(output), redirect_stderr(output):
                yield
        finally:
            if debug:
                if output.getvalue():
                    print('Debug — salida interna:\n' + output.getvalue().rstrip())
                for warning in captured:
                    print(f'Debug — {warning.category.__name__}: {warning.message}')


def load_system():
    # El guard se instala antes de importar torch o cargar artefactos.
    import seguridad_5C_common  # noqa: F401
    from respuesta_segura import ChatSeguro
    from retrieval import load_index
    from contexto_rag_5B1 import neighbor_lookup, source_lookup
    system = ChatSeguro()
    load_index()
    source_lookup()
    neighbor_lookup()
    return system


def show_model_debug(system):
    print('Debug — modelo:', json.dumps(system.model_metadata, ensure_ascii=False, indent=2))


def show_error(message, debug):
    print(message, file=sys.stderr)
    if debug:
        traceback.print_exc()


def clear_screen():
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        print('\n' + SEPARATOR + '\n')
    header()


class SessionLog:
    """Un archivo UTF-8 por ejecución, creado sólo al guardar un turno exitoso."""
    def __init__(self):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.path = Path(__file__).resolve().parent / 'resultados' / 'sesiones' / f'sesion_{stamp}_{os.getpid()}.jsonl'

    def save(self, result):
        row = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                   pregunta=result['question'], respuesta=compact(result['final_answer']),
                   fuentes=displayed_sources(result))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def run_chat(args):
    if args.pregunta is not None and not args.pregunta.strip():
        print('Escribe una pregunta no vacía.', file=sys.stderr)
        return 2
    header()
    print('Cargando MiniGPT...', flush=True)
    try:
        with runtime_output(args.debug):
            system = load_system()
    except EXPECTED_ERRORS:
        show_error('No se pudo cargar MiniGPT. Revisa los recursos locales; usa --debug para más detalles.', args.debug)
        return 1
    print('Listo.\n')
    debug = args.debug
    if debug:
        show_model_debug(system)
    session = SessionLog() if args.guardar_sesion else None

    def run(question, echo_question=True):
        nonlocal session
        try:
            with runtime_output(debug):
                result = system.answer(question)
        except EXPECTED_ERRORS:
            show_error('No se pudo procesar la pregunta. Puedes intentarlo de nuevo o activar /debug on.', debug)
            print()
            return False
        show(result, debug, echo_question=echo_question)
        if session is not None:
            try:
                session.save(result)
            except EXPECTED_ERRORS:
                show_error('No se pudo guardar la sesión. El chat sigue disponible; se desactivó el registro.', debug)
                session = None
        return True

    if args.pregunta is not None:
        return 0 if run(args.pregunta.strip()) else 1
    while True:
        try:
            question = input('Tu pregunta: ' if sys.stdin.isatty() else '').strip()
        except EOFError:
            print('\nHasta luego.')
            return 0
        if not question:
            continue
        command = ' '.join(question.casefold().split())
        if command in ('/salir', '/exit', 'salir', 'exit'):
            print('\nHasta luego.')
            return 0
        if command == '/ayuda':
            print('\n' + HELP)
        elif command == '/limpiar':
            clear_screen()
        elif command in ('/debug on', '/debug off'):
            debug = command == '/debug on'
            print('\nInformación técnica ' + ('activada.' if debug else 'desactivada.') + '\n')
            if debug:
                show_model_debug(system)
        elif command.startswith('/'):
            print('\nComando no reconocido. Escribe /ayuda para ver los comandos.\n')
        else:
            # input ya muestra la pregunta en una terminal; se repite en redirecciones.
            run(question, echo_question=not sys.stdin.isatty())


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--pregunta', help='procesar una única pregunta')
    modes.add_argument('--interactive', action='store_true', help='iniciar una sesión de chat')
    parser.add_argument('--debug', action='store_true', help='mostrar la trazabilidad técnica completa')
    parser.add_argument('--guardar-sesion', action='store_true', help='guardar los turnos en resultados/sesiones/')
    args = parser.parse_args(argv)
    try:
        return run_chat(args)
    except KeyboardInterrupt:
        print('\nHasta luego.')
        return 0


if __name__ == '__main__':
    sys.exit(main())
