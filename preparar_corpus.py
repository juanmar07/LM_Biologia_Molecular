"""Etapa 2A: PDF por fuente, limpieza conservadora y métricas reproducibles."""

import hashlib
import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

# Rutas equivalentes a config.py, sin importar torch en la preparación de datos.
BASE_DIR = Path(__file__).resolve().parent
CARPETA_CORPUS = BASE_DIR / "corpus"
ARCHIVO_CORPUS_LIMPIO = BASE_DIR / "datos" / "corpus_limpio.txt"
CARPETA_DOCUMENTOS = BASE_DIR / "datos" / "documentos_limpios"
ARCHIVO_REPORTE = BASE_DIR / "datos" / "reporte_corpus.json"
NUMERO_PDF_ESPERADO = 21
VERSION_LIMPIEZA = "2A.2"
LIGADURAS = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
PALABRA = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+"
PARTIDA = re.compile(r"\b([\w]+)[-\u00ad][ \t]*\n[ \t]*([\w]+)")
CONTADOR = re.compile(r"^(?:Page|Página|Pág\.?)\s+(\d+)\s*(?:/|de)\s*(\d+)\s*$", re.I)
FRACCION = re.compile(r"^(\d+)\s*/\s*(\d+)$")
NUMERO = re.compile(r"^[-–—]?\s*(\d{1,4})\s*[-–—]?$", re.ASCII)
EDITORIAL = re.compile(r"copyright|©|todos los derechos|all rights reserved|terms of use|términos de uso|privacy|privacidad|access provided|acceso proporcionado|downloaded|your ip|su dirección ip|silverchair|^ISSN\b|^eISSN\b|^\[REV\.|^Departamento de Genética, Universidad", re.I)
TITULO = re.compile(r"^(?:introducción|conclusiones|resumen|abstract|referencias(?: bibliográficas)?|bibliografía|objetivos|metodología|capítulo\b.*)$", re.I)


def sha256(datos):
    return hashlib.sha256(datos).hexdigest()


def es_uso_privado(c):
    n = ord(c)
    return 0xE000 <= n <= 0xF8FF or 0xF0000 <= n <= 0xFFFFD or 0x100000 <= n <= 0x10FFFD


def es_control_invalido(c):
    return unicodedata.category(c) == "Cc" and c not in "\n\r\t"


def normalizar_representacion(texto):
    texto = unicodedata.normalize("NFC", texto).translate(LIGADURAS)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    # El espacio impide concatenar palabras al retirar un control inválido.
    texto = "".join(" " if es_control_invalido(c) or unicodedata.category(c) == "Zs" else c for c in texto)
    return re.sub(r"[ \t]+", " ", texto)


def posiciones_borde(lineas, cantidad=3):
    utiles = [i for i, linea in enumerate(lineas) if linea.strip()]
    return set(utiles[:cantidad] + utiles[-cantidad:])


def bloque_pie_plataforma(lineas):
    """Reconoce el bloque de exportación, no títulos científicos aislados."""
    utiles = [i for i, s in enumerate(lineas) if s.strip()]
    finales = utiles[-7:]
    accesos = [i for i in finales if re.match(r"^Access Provided by:", lineas[i].strip(), re.I)]
    if not accesos or not any("copyright" in lineas[i].lower() or "©" in lineas[i] for i in finales):
        return set()
    inicio = accesos[0]
    anterior = next((i for i in reversed(utiles) if i < inicio), None)
    if anterior is not None and re.match(r"^Universidad\b", lineas[anterior].strip()):
        inicio = anterior
    return {i for i in finales if i >= inicio}


def clave_pagina(texto, numeracion=None):
    """Representación solo para comparar; nunca sustituye al texto de salida."""
    lineas = normalizar_representacion(texto).splitlines()
    bordes = posiciones_borde(lineas)
    numeracion = numeracion or set()
    lineas = [s for i, s in enumerate(lineas) if not (i in bordes and
              (CONTADOR.fullmatch(s.strip()) or i in numeracion))]
    return " ".join(" ".join(lineas).split())


def clave_cuerpo(texto, numeracion=None):
    lineas = normalizar_representacion(texto).splitlines()
    bordes = posiciones_borde(lineas, 6)
    # Sustituir por líneas vacías mantiene los índices de numeración detectados.
    pie = bloque_pie_plataforma(lineas)
    cuerpo = "\n".join("" if i in pie or (i in bordes and EDITORIAL.search(s)) else s for i, s in enumerate(lineas))
    clave, completa = clave_pagina(cuerpo, numeracion), clave_pagina(texto, numeracion)
    return clave if len(clave) >= 1000 and len(clave) >= 0.90 * len(completa) else completa


def detectar_duplicados(paginas):
    conservadas, duplicadas, vistas, cuerpos = [], [], {}, {}
    claves_por_pagina = {}
    normalizadas = [{"numero": p["numero"], "texto": normalizar_representacion(p["texto"])} for p in paginas]
    _, numeracion = detectar_margenes(normalizadas)
    for pagina in paginas:
        ignoradas = {i for numero, i in numeracion if numero == pagina["numero"]}
        # Las fracciones se ignoran solo si coinciden con número real y total del PDF.
        lineas = normalizar_representacion(pagina["texto"]).splitlines()
        for i in posiciones_borde(lineas):
            m = FRACCION.fullmatch(lineas[i].strip())
            if m and int(m[1]) == pagina["numero"] and int(m[2]) == len(paginas):
                ignoradas.add(i)
        clave, cuerpo = clave_pagina(pagina["texto"], ignoradas), clave_cuerpo(pagina["texto"], ignoradas)
        claves_por_pagina[pagina["numero"]] = cuerpo
        anterior = vistas.get(clave) if len(clave) >= 1000 else None
        metodo = "texto_normalizado_y_contador"
        if anterior is None and len(cuerpo) >= 1000:
            anterior = cuerpos.get(cuerpo)
            metodo = "cuerpo_igual_con_variacion_editorial_en_bordes"
        if anterior is not None:
            duplicadas.append({"pagina": pagina["numero"], "copia_de": anterior,
                               "metodo": metodo, "caracteres_omitidos": len(pagina["texto"])})
        else:
            conservadas.append(pagina)
            # No deduplicar páginas breves de títulos, ejercicios o tablas.
            if len(clave) >= 1000:
                vistas[clave] = pagina["numero"]
                cuerpos[cuerpo] = pagina["numero"]
    # Algunos PDF exportan mitades repetidas y una página con ambas mitades.
    # Solo se omite un bloque entero largo, idéntico y ya repetido >=3 veces.
    copias = Counter(d["copia_de"] for d in duplicadas)
    incluidas = {}
    for pagina in conservadas:
        numero = pagina["numero"]
        clave = claves_por_pagina[numero]
        if len(clave) < 10000 or copias[numero] < 2:
            continue
        mayores = [p for p in conservadas if len(claves_por_pagina[p["numero"]]) > len(clave)
                   and clave in claves_por_pagina[p["numero"]]]
        if mayores:
            mayor = max(mayores, key=lambda p: (len(claves_por_pagina[p["numero"]]), -p["numero"]))
            incluidas[numero] = mayor["numero"]
            duplicadas.append({"pagina": numero, "copia_de": mayor["numero"],
                               "metodo": "bloque_largo_repetido_incluido_integro_en_otra_pagina",
                               "caracteres_omitidos": len(pagina["texto"])})
    for duplicada in duplicadas:
        while duplicada["copia_de"] in incluidas:
            duplicada["copia_de"] = incluidas[duplicada["copia_de"]]
    conservadas = [p for p in conservadas if p["numero"] not in incluidas]
    duplicadas.sort(key=lambda d: d["pagina"])
    return conservadas, duplicadas


class CapturadorAdvertencias(logging.Handler):
    def __init__(self):
        super().__init__()
        self.mensajes = Counter()

    def emit(self, registro):
        self.mensajes[registro.getMessage()] += 1


def extraer_paginas(ruta):
    from pypdf import PdfReader

    datos = {"paginas": 0, "paginas_sin_texto": [], "errores": [], "avisos_pypdf": []}
    paginas, capturador = [], CapturadorAdvertencias()
    logger = logging.getLogger("pypdf")
    propagacion = logger.propagate
    logger.addHandler(capturador)
    logger.propagate = False
    try:
        lector = PdfReader(str(ruta))
        datos["paginas"] = len(lector.pages)
        for i in range(datos["paginas"]):
            try:
                texto = lector.pages[i].extract_text() or ""
                if texto.strip():
                    paginas.append({"numero": i + 1, "texto": texto})
                else:
                    datos["paginas_sin_texto"].append(i + 1)
            except Exception as exc:
                datos["errores"].append({"pagina": i + 1, "tipo": type(exc).__name__, "mensaje": str(exc)})
    except Exception as exc:
        datos["errores"].append({"pagina": None, "tipo": type(exc).__name__, "mensaje": str(exc)})
    finally:
        logger.removeHandler(capturador)
        logger.propagate = propagacion
    datos["avisos_pypdf"] = [{"mensaje": s, "veces": n} for s, n in sorted(capturador.mensajes.items())]
    return paginas, datos


def detectar_margenes(paginas):
    ocurrencias, interiores, numeros = defaultdict(set), set(), defaultdict(list)
    for pagina in paginas:
        lineas = pagina["texto"].splitlines()
        bordes = posiciones_borde(lineas)
        for i, linea in enumerate(lineas):
            s = linea.strip()
            if not s:
                continue
            if i in bordes:
                ocurrencias[s].add(pagina["numero"])
                m = NUMERO.fullmatch(s)
                if m:
                    lado = "inicio" if i < len(lineas) / 2 else "fin"
                    numeros[(lado, int(m[1]) - pagina["numero"])].append((pagina["numero"], i))
            else:
                interiores.add(s)
    umbral = max(3, math.ceil(0.60 * len(paginas)))
    repetidas = {s for s, apariciones in ocurrencias.items()
                 if len(apariciones) >= umbral and s not in interiores and not TITULO.fullmatch(s)
                 and not NUMERO.fullmatch(s) and len(s) <= 180
                 and (EDITORIAL.search(s) or (len(s.split()) <= 18 and not s.endswith((".", ";", ":"))))}
    numeracion = set()
    for posiciones in numeros.values():
        if len({p for p, _ in posiciones}) >= umbral:
            numeracion.update(posiciones)
    return repetidas, numeracion


def quitar_margenes(pagina, repetidas, numeracion, total):
    lineas = pagina["texto"].splitlines()
    bordes, editoriales = posiciones_borde(lineas), posiciones_borde(lineas, 6)
    pie = bloque_pie_plataforma(lineas)
    salida, eliminadas = [], Counter()
    for i, linea in enumerate(lineas):
        s = linea.strip()
        contador = CONTADOR.fullmatch(s) or FRACCION.fullmatch(s)
        seguro = contador and int(contador[1]) == pagina["numero"] and int(contador[2]) == total
        motivo = None
        if i in pie:
            motivo = "bloque_pie_de_plataforma"
        elif i in bordes and (seguro or (pagina["numero"], i) in numeracion):
            motivo = "numeracion_de_pagina"
        elif i in bordes and s in repetidas:
            motivo = "linea_repetida_en_margenes"
        elif i in editoriales and EDITORIAL.search(s) and not TITULO.fullmatch(s):
            motivo = "metadato_editorial_en_margen"
        if motivo:
            eliminadas[motivo] += 1
        else:
            salida.append(linea)
    return "\n".join(salida), eliminadas


def evidencia_lexica(texto):
    return Counter(re.findall(rf"\b{PALABRA}\b", texto)), set(re.findall(rf"\b{PALABRA}-{PALABRA}\b", texto))


def reconstruir_guiones(texto, palabras, compuestas):
    reconstruidas, preservados = 0, 0

    def reemplazar(match):
        nonlocal reconstruidas, preservados
        izquierda, derecha = match.groups()
        unida, compuesta = izquierda + derecha, izquierda + "-" + derecha
        # Una aparición intacta en la misma fuente es evidencia de división tipográfica.
        tipografico = (re.fullmatch(PALABRA, izquierda) and re.fullmatch(PALABRA, derecha)
                       and izquierda.islower() and derecha.islower()
                       and palabras[unida] > 0 and compuesta not in compuestas)
        if tipografico:
            reconstruidas += 1
            return unida
        preservados += 1
        return compuesta

    texto = PARTIDA.sub(reemplazar, texto)
    return texto, reconstruidas, preservados


def normalizar_espacios(texto):
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = "\n".join(s.strip() for s in texto.split("\n"))
    return unicodedata.normalize("NFC", re.sub(r"\n{3,}", "\n\n", texto).strip())


def limpiar_texto(texto):
    texto = normalizar_representacion(texto)
    palabras, compuestas = evidencia_lexica(texto)
    texto, _, _ = reconstruir_guiones(texto, palabras, compuestas)
    return normalizar_espacios(texto)


def senales_editoriales(texto):
    return {
        "doi": len(re.findall(r"\b10\.\d{4,9}/[^\s]+", texto)),
        "urls": len(re.findall(r"https?://|www\.", texto, re.I)),
        "correos": len(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", texto)),
        "titulos_referencias": len(re.findall(r"(?im)^\s*(?:referencias(?: bibliográficas)?|bibliografía|references)\s*$", texto)),
        "marcas_abstract_ingles": len(re.findall(r"\b(?:ABSTRACT|Abstract|Key words|Keywords)\b", texto)),
        "lineas_con_metadatos_editoriales": sum(bool(EDITORIAL.search(s)) for s in texto.splitlines()),
    }


def nombre_salida(relativa):
    base = unicodedata.normalize("NFKD", Path(relativa).stem)
    base = "".join(c for c in base if not unicodedata.combining(c))
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")[:80] or "documento"
    return "doc_" + base + "_" + sha256(relativa.encode("utf-8"))[:12] + ".txt"


PRIVADOS_VERIFICADOS = {
    "\ue92f": "chevrón editorial al final del título del libro",
    "\uf0a7": "viñeta cuadrada", "\uf0b7": "viñeta redonda",
    "\uf644": "dígito 1 en glucagón-1 (TUM, página 236)",
    "\uf64a": "dígito 7 en P7 (TUM, página 251)",
    "\uf64b": "dígito 8 en por lo menos 8 horas (TUM, página 220)",
}


def corregir_unicode_verificado(texto, fuente):
    """Correcciones posteriores a deduplicar; glifos contrastados con el PDF.

    U+00AD representa guiones visibles en los capítulos exportados: nunca
    eliminarlo ciegamente entre letras, cifras o símbolos científicos.
    """
    antes = texto
    # Portada TUM, página 3: el glifo de U+00AD es É en MÉXICO,
    # dentro de una línea con mapa de fuente defectuoso. No es un guion.
    glifos_portada = 0
    if fuente.startswith("TUM -"):
        glifos_portada = texto.count(".\u00ad9*$0")
        texto = texto.replace(".\u00ad9*$0", ".É9*$0")
    palabras, compuestas = evidencia_lexica(texto)
    eliminados = 0

    def unir(match):
        nonlocal eliminados
        a, b = match.groups()
        if a.islower() and b.islower() and palabras[a + b] and a + "-" + b not in compuestas:
            eliminados += 1
            return a + b
        return match[0]

    texto = re.sub(rf"\b({PALABRA})\u00ad({PALABRA})\b", unir, texto)
    convertidos = texto.count("\u00ad")
    # En los capítulos exportados los restantes son guiones visibles.
    texto = texto.replace("\u00ad", "-")
    texto = re.sub(r"(?m)^((?:Biología|Tratado|Harper|Harrison)[^\n]*)\ue92f[ \t]*$", r"\1", texto)
    if fuente.startswith("Manual de problemas"):
        texto = re.sub(r"(?m)^[\uf0a7\uf0b7](?=\s)", "-", texto)
    if fuente.startswith("TUM -"):
        texto = texto.replace("glucagón-\uf644", "glucagón-1")
        texto = re.sub(r"(?m)^P\uf64a$", "P7", texto)
        texto = texto.replace("por lo menos \uf64b horas", "por lo menos 8 horas")
    privados = {}
    for c, significado in PRIVADOS_VERIFICADOS.items():
        n, restantes = antes.count(c), texto.count(c)
        if n:
            privados[f"U+{ord(c):04X}"] = {
                "antes_correccion": n, "corregidos": n - restantes,
                "restantes": restantes, "significado": significado,
                "confianza": "alta: glifo y contexto comprobados visualmente en PDF",
                "contexto": antes[max(0, antes.index(c) - 55):antes.index(c) + 56],
            }
    return unicodedata.normalize("NFC", texto), {
        "antes_correccion": antes.count("\u00ad"),
        "eliminados_con_evidencia_lexica": eliminados,
        "glifos_portada_corregidos_a_E_aguda": glifos_portada,
        "convertidos_a_guion": convertidos, "restantes": texto.count("\u00ad"),
    }, privados


def validar_capitulo_2(paginas, conservadas):
    """Cobertura exhaustiva de contenido, independiente de hashes y longitudes."""
    cuerpos = {p["numero"]: clave_cuerpo(p["texto"]) for p in paginas}
    retenidos = [cuerpos[p["numero"]] for p in conservadas]

    def bloques(texto):
        palabras = texto.split()
        return {tuple(palabras[i:i + 24]) for i in range(max(0, len(palabras) - 23))}

    todos = set().union(*(bloques(t) for t in cuerpos.values()))
    presentes = set().union(*(bloques(t) for t in retenidos))
    faltantes = todos - presentes
    grupos = defaultdict(list)
    for numero, cuerpo in cuerpos.items():
        grupos[cuerpo].append(numero)
    return {
        "paginas_comparadas": len(paginas), "paginas_retenidas": [p["numero"] for p in conservadas],
        "grupos_canonicos": [{"paginas": nums, "caracteres": len(t)} for t, nums in grupos.items()],
        "longitud_bloque_palabras": 24, "paso_palabras": 1,
        "bloques_unicos": len(todos), "bloques_representados": len(todos & presentes),
        "cobertura_porcentaje": 100 * len(todos & presentes) / len(todos) if todos else 0,
        "bloques_faltantes": len(faltantes),
        "cuerpos_incluidos_integros": sum(any(t in r for r in retenidos) for t in cuerpos.values()),
        "caracteres_de_cuerpos_no_incluidos": sum(len(t) for t in cuerpos.values() if not any(t in r for r in retenidos)),
        "representacion": "clave_cuerpo antes de correcciones Unicode; solo se excluyen márgenes editoriales reconocidos",
    }


def procesar_documento(ruta, relativa):
    paginas, extraccion = extraer_paginas(ruta)
    originales = [p["texto"] for p in paginas]
    problematicos = Counter(f"U+{ord(c):04X}" for t in originales for c in t
                            if es_uso_privado(c) or es_control_invalido(c) or c == "\ufffd" or unicodedata.category(c) == "Cf")
    datos = {"nombre": ruta.name, "fuente": relativa, "sha256_pdf": sha256(ruta.read_bytes()),
             "archivo_limpio": "datos/documentos_limpios/" + nombre_salida(relativa), **extraccion,
             "paginas_con_texto": len(paginas), "caracteres_originales": sum(map(len, originales)),
             "caracteres_unicode_problematicos_detectados": sum(problematicos.values()),
             "unicode_problematico_por_codigo": dict(sorted(problematicos.items())),
             "controles_invalidos_detectados": sum(es_control_invalido(c) for t in originales for c in t),
             "advertencias": []}
    conservadas, duplicadas = detectar_duplicados(paginas)
    if ruta.name.startswith("CAPÍTULO 2_"):
        datos["validacion_exhaustiva_capitulo_2"] = validar_capitulo_2(paginas, conservadas)
    datos.update(duplicaciones_detectadas=duplicadas, paginas_o_bloques_duplicados_omitidos=len(duplicadas),
                 paginas_conservadas=[p["numero"] for p in conservadas],
                 caracteres_tras_deduplicacion=sum(len(p["texto"]) for p in conservadas),
                 controles_invalidos_eliminados=sum(es_control_invalido(c) for p in conservadas for c in p["texto"]))
    preparadas = [{"numero": p["numero"], "texto": normalizar_representacion(p["texto"])} for p in conservadas]
    repetidas, numeracion = detectar_margenes(preparadas)
    partes, eliminadas = [], Counter()
    for pagina in preparadas:
        texto, cuenta = quitar_margenes(pagina, repetidas, numeracion, extraccion["paginas"])
        partes.append(texto)
        eliminadas.update(cuenta)
    texto = "\n\n".join(partes)
    palabras, compuestas = evidencia_lexica(texto)
    texto, reconstruidas, preservados = reconstruir_guiones(texto, palabras, compuestas)
    texto = normalizar_espacios(texto)
    texto, suaves, privados = corregir_unicode_verificado(texto, relativa)
    suaves["extraidos_antes_deduplicacion"] = sum(t.count("\u00ad") for t in originales)
    datos["soft_hyphens"] = suaves
    datos["caracteres_privados_verificados"] = privados
    datos["unicode_privado_restante"] = dict(sorted(Counter(f"U+{ord(c):04X}" for c in texto if es_uso_privado(c)).items()))
    datos.update(caracteres_limpios=len(texto), sha256_texto_limpio=sha256(texto.encode("utf-8")),
                 reconstrucciones_por_guion=reconstruidas, guiones_conservados_al_unir_linea=preservados,
                 cabeceras_pies_eliminados=sum(eliminadas.values()), lineas_eliminadas_por_motivo=dict(sorted(eliminadas.items())),
                 divisiones_por_guion_pendientes=len(re.findall(r"\w[-\u00ad][ \t]*\n", texto)),
                 patrones_repetidos_en_margenes=sorted(repetidas),
                 senales_editoriales=senales_editoriales(texto))
    condiciones = [
        (extraccion["paginas_sin_texto"], "Páginas sin texto extraíble; no se aplicó OCR."),
        (extraccion["errores"], "Extracción incompleta: consultar errores y números de página."),
        (extraccion["avisos_pypdf"], "El lector PDF notificó problemas estructurales; consultar avisos_pypdf."),
        (duplicadas, "Páginas de texto largo duplicado omitidas dentro de esta fuente."),
        (problematicos, "Unicode dudoso detectado: consultar correcciones verificadas y códigos restantes; casos no comprobados se conservan."),
        (preservados, "Guiones sin evidencia suficiente conservados al unir líneas."),
        (datos["divisiones_por_guion_pendientes"], "Divisiones de línea pendientes, incluidos límites de página; se conserva el texto dudoso."),
        (any(datos["senales_editoriales"].values()), "Referencias, URLs, DOI, correos o marcas editoriales/abstract detectados; se conservan en el cuerpo."),
        (not texto, "Documento sin texto limpio útil; revisar antes de tokenizar."),
    ]
    datos["advertencias"] = [mensaje for condicion, mensaje in condiciones if condicion]
    return texto, datos


def descubrir_pdf():
    archivos = [p for p in CARPETA_CORPUS.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
                and not p.name.startswith("._") and "__MACOSX" not in p.parts]
    return sorted(archivos, key=lambda p: p.relative_to(CARPETA_CORPUS).as_posix())


def preparar_corpus():
    archivos = descubrir_pdf()
    if len(archivos) != NUMERO_PDF_ESPERADO:
        inventario = "\n".join(p.relative_to(CARPETA_CORPUS).as_posix() for p in archivos)
        raise RuntimeError(f"Se esperaban 21 PDF; se encontraron {len(archivos)}. No se modificó ninguna salida.\n{inventario}")
    documentos, bloques, salidas = [], [], {}
    for ruta in archivos:
        relativa = ruta.relative_to(CARPETA_CORPUS).as_posix()
        print(f"Procesando: {relativa}", flush=True)
        texto, datos = procesar_documento(ruta, relativa)
        documentos.append(datos)
        salidas[BASE_DIR / datos["archivo_limpio"]] = texto
        fuente = json.dumps({"fuente": relativa}, ensure_ascii=False, sort_keys=True)
        bloques.append(f"<DOCUMENTO> {fuente}\n{texto}\n</DOCUMENTO>")
    corpus = "\n\n".join(bloques) + "\n"
    resumen = {
        "numero_pdf": len(archivos), "paginas_totales": sum(d["paginas"] for d in documentos),
        "paginas_con_texto": sum(d["paginas_con_texto"] for d in documentos),
        "paginas_sin_texto": sum(len(d["paginas_sin_texto"]) for d in documentos),
        "caracteres_extraidos_antes_limpieza": sum(d["caracteres_originales"] for d in documentos),
        "caracteres_finales": sum(d["caracteres_limpios"] for d in documentos),
        "caracteres_corpus_con_delimitadores": len(corpus),
        "documentos_procesados": sum(bool(d["caracteres_limpios"]) for d in documentos),
        "documentos_con_advertencias": sum(bool(d["advertencias"]) for d in documentos),
        "errores": [{"fuente": d["fuente"], **error} for d in documentos for error in d["errores"]],
        "paginas_o_bloques_duplicados_omitidos": sum(d["paginas_o_bloques_duplicados_omitidos"] for d in documentos),
        "palabras_partidas_reconstruidas": sum(d["reconstrucciones_por_guion"] for d in documentos),
        "lineas_de_cabecera_pie_eliminadas": sum(d["cabeceras_pies_eliminados"] for d in documentos),
        "caracteres_unicode_problematicos_detectados": sum(d["caracteres_unicode_problematicos_detectados"] for d in documentos),
        "sha256_corpus_limpio": sha256(corpus.encode("utf-8")),
        "soft_hyphens": {k: sum(d["soft_hyphens"][k] for d in documentos) for k in documentos[0]["soft_hyphens"]},
        "unicode_privado_restante": dict(sorted(sum((Counter(d["unicode_privado_restante"]) for d in documentos), Counter()).items())),
        "caracteres_privados_verificados": {codigo: {
            "extraidos_antes_deduplicacion": sum(d["unicode_problematico_por_codigo"].get(codigo, 0) for d in documentos),
            "antes_correccion": sum(d["caracteres_privados_verificados"].get(codigo, {}).get("antes_correccion", 0) for d in documentos),
            "corregidos": sum(d["caracteres_privados_verificados"].get(codigo, {}).get("corregidos", 0) for d in documentos),
            "documentos": [d["fuente"] for d in documentos if codigo in d["unicode_problematico_por_codigo"]],
            "significado": significado,
        } for c, significado in PRIVADOS_VERIFICADOS.items() for codigo in [f"U+{ord(c):04X}"]},
    }
    reporte = {"version_limpieza": VERSION_LIMPIEZA, "sha256_codigo_preparacion": sha256(Path(__file__).read_bytes()),
               "version_pypdf": version("pypdf"),
               "politicas": {"normalizacion": "NFC", "minusculas_globales": False, "ocr": False,
                   "deduplicacion": "páginas largas de la misma fuente; sin deduplicación entre documentos",
                   "guiones": "evidencia de palabra intacta en la misma fuente; casos dudosos conservan guion",
                   "cabeceras": "bordes repetidos >=60% y >=3 páginas o metadatos editoriales explícitos",
                   "unicode_dudoso": "se contabiliza; privados solo se corrigen en contextos visualmente verificados; otros se conservan",
                   "soft_hyphens": "palabra intacta en la fuente permite unir; resto se convierte a guion visible para preservar notación"},
               "resumen": resumen, "documentos": documentos}
    # Solo artefactos autorizados; bytes UTF-8 garantizan LF incluso en Windows.
    CARPETA_DOCUMENTOS.mkdir(parents=True, exist_ok=True)
    for ruta, texto in salidas.items():
        ruta.write_bytes(texto.encode("utf-8"))
    ARCHIVO_CORPUS_LIMPIO.write_bytes(corpus.encode("utf-8"))
    ARCHIVO_REPORTE.write_bytes((json.dumps(reporte, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(resumen, ensure_ascii=True, indent=2), flush=True)
    if resumen["errores"] or resumen["documentos_procesados"] != len(archivos):
        raise RuntimeError("Preparación incompleta: consultar reporte; no avanzar a tokenización.")
    return corpus


if __name__ == "__main__":
    preparar_corpus()
