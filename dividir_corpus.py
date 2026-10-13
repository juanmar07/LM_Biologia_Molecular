"""Etapa 2B.1: partición por familias, sin modificar ni tokenizar el corpus 2A.2.

Solo biblioteca estándar. Ejecutar: python -B dividir_corpus.py
Los 210 pares se comparan con conjuntos exactos de tuplas de 8 y 12 palabras.
"""

import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
CORPUS = BASE / "datos/corpus_limpio.txt"
REPORT = BASE / "datos/reporte_corpus.json"
OUTPUT = BASE / "datos/splits.json"
EXPECTED_CORPUS = "c10bcaf79faf39179a8f0f61bc3c21c6b27a16dd0f6a0a6e710b4f3be8d66e49"
EXPECTED_REPORT = "cd7d968b9b4d32b267f4d08d1fb78d9f81a48da13780d586a22c50a32d518344"
WORD = re.compile(r"\b\w+(?:[-’']\w+)*\b")
SURFACE = re.compile(r"\w+|[^\w\s]")
SPLITS = ("train", "validation", "test")
TARGET = (0.8, 0.1, 0.1)
RANGES = ((0.75, 0.85), (0.075, 0.15), (0.075, 0.15))
FAMILY_THRESHOLD = 0.15


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_inputs():
    # Fallar antes de analizar o escribir si cambió alguna entrada congelada.
    require(digest(CORPUS.read_bytes()) == EXPECTED_CORPUS, "Hash del corpus 2A.2 distinto; no se escribieron salidas.")
    require(digest(REPORT.read_bytes()) == EXPECTED_REPORT, "Hash del reporte 2A.2 distinto; no se escribieron salidas.")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    require(report["version_limpieza"] == "2A.2", "Versión de limpieza inesperada.")
    rows = sorted(report["documentos"], key=lambda d: d["fuente"])
    require(len(rows) == 21 and len({d["fuente"] for d in rows}) == 21, "Se requieren exactamente 21 fuentes únicas.")
    documents, seen = [], set()
    for row in rows:
        path = (BASE / row["archivo_limpio"]).resolve()
        require(path.parent == (BASE / "datos/documentos_limpios").resolve(), "Ruta TXT fuera del directorio autorizado.")
        require(path not in seen, "Dos fuentes apuntan al mismo TXT.")
        seen.add(path)
        data = path.read_bytes()
        require(digest(data) == row["sha256_texto_limpio"], f"TXT congelado modificado: {row['fuente']}")
        text = data.decode("utf-8")
        require(len(text) == row["caracteres_limpios"], "Longitud TXT inconsistente.")
        words = WORD.findall(unicodedata.normalize("NFC", text).lower())
        documents.append({"id": "D_" + digest(row["fuente"].encode("utf-8"))[:12],
                          "source": row["fuente"], "clean_file": row["archivo_limpio"],
                          "sha256": digest(data), "characters": len(text),
                          "approximate_words": len(words), "text": text,
                          "shingles": {n: {tuple(words[i:i+n]) for i in range(len(words)-n+1)} for n in (8, 12)}})
    require(seen == {p.resolve() for p in (BASE / "datos/documentos_limpios").glob("*.txt")}, "Inventario TXT no coincide con reporte.")
    require(sum(d["characters"] for d in documents) == 3733434, "Volumen congelado inesperado.")
    require(build_text(documents).encode("utf-8") == CORPUS.read_bytes(), "Fuentes y corpus consolidado no coinciden.")
    return documents


def build_text(documents):
    blocks = ["<DOCUMENTO> " + json.dumps({"fuente": d["source"]}, ensure_ascii=False, sort_keys=True)
              + "\n" + d["text"] + "\n</DOCUMENTO>" for d in sorted(documents, key=lambda d: d["source"])]
    return "\n\n".join(blocks) + "\n"


def pair_metrics(a, b, n):
    x, y = a["shingles"][n], b["shingles"][n]
    shared = len(x & y)
    return {"shingles_a": len(x), "shingles_b": len(y), "shingles_compartidos": shared,
            "containment_a_en_b": shared / len(x), "containment_b_en_a": shared / len(y),
            "containment_documento_menor": shared / min(len(x), len(y)),
            "jaccard": shared / (len(x) + len(y) - shared)}


def analyze_pairs(documents):
    pairs = []
    for a, b in itertools.combinations(documents, 2):
        primary, secondary = pair_metrics(a, b, 8), pair_metrics(a, b, 12)
        small = primary["containment_documento_menor"]
        exact = a["sha256"] == b["sha256"]
        classification = ("exact_duplicate" if exact else "strong_overlap" if small >= 0.15
                          else "moderate_overlap" if small >= 0.05 else "low_overlap")
        pairs.append({"documento_a": a["source"], "documento_b": b["source"],
                      "palabras_a": a["approximate_words"], "palabras_b": b["approximate_words"],
                      "exact_sha256_duplicate": exact, **primary, "secondary_12_words": secondary,
                      "clasificacion": classification})
    require(len(pairs) == 210, "Se deben analizar 210 pares.")
    return pairs


def build_families(documents, pairs):
    by_source = {d["source"]: d for d in documents}
    parent = {s: s for s in by_source}
    evidence = []

    def root(s):
        while parent[s] != s:
            s = parent[s]
        return s

    def link(sources, reason, quotes):
        sources = sorted(sources)
        require(len(sources) >= 2, "Familia editorial sin miembros suficientes.")
        roots = sorted({root(s) for s in sources})
        for r in roots[1:]:
            parent[r] = roots[0]
        evidence.append({"documents": sources, "reason": reason, "textual_evidence": quotes})

    # Evidencia de obra/serie, no agrupación por tema ni por empresa editora.
    salazar = []
    roche = []
    epigenetics = []
    quotes = {}
    for d in documents:
        text = d["text"]
        match = re.search(r"Biología [Mm]olecular\. Fundamentos y aplicaciones en las ciencias de la salud[^\n]*", text)
        if d["source"].startswith("TUM -"):
            match = re.search(r"BIOLOGÍA MOLECULAR Salazar[^\n]*", text)
        if match:
            salazar.append(d["source"])
            quotes[d["source"]] = match[0]
        if "Informes Anticipando" in text and "Fundación Instituto Roche" in text:
            roche.append(d["source"])
        if "EPIGENÉTICA EN MEDICINA" in text and "Material didáctico creado por Mar Benito" in text:
            epigenetics.append(d["source"])
    require(len(salazar) == 6 and len(roche) == 2 and len(epigenetics) == 2, "Evidencia editorial cambió; revisar familias.")
    link(salazar, "Misma obra Salazar/Sandoval/Armendáriz, incluyendo capítulos de distintas ediciones; título verificable aunque el capítulo 30 solape poco.",
         {s: quotes[s] for s in sorted(salazar)})
    link(roche, "Misma serie editorial de informes de Fundación Instituto Roche.",
         {s: "Informes Anticipando; Fundación Instituto Roche; www.institutoroche.es" for s in sorted(roche)})
    link(epigenetics, "Sesiones 1 y 2 del mismo material didáctico Epigenética en medicina, creado por Mar Benito.",
         {s: "EPIGENÉTICA EN MEDICINA; Material didáctico creado por Mar Benito; Sesión " + ("1" if s == "epigenetica.pdf" else "2") for s in sorted(epigenetics)})
    for pair in pairs:
        if pair["exact_sha256_duplicate"] or pair["containment_documento_menor"] >= FAMILY_THRESHOLD:
            # La distribución real tiene cuatro pares >=23.37%, con evidencia
            # de la misma obra; el siguiente par es 1.413%. No unir a ciegas.
            a, b = pair["documento_a"], pair["documento_b"]
            require(root(a) == root(b) or pair["exact_sha256_duplicate"], "Nuevo solapamiento sustancial sin revisión editorial.")
            link([a, b], "Duplicado exacto o solapamiento fuerte corroborado por origen editorial.",
                 {"containment_8": pair["containment_documento_menor"],
                  "containment_12": pair["secondary_12_words"]["containment_documento_menor"]})
    groups = {}
    for source in sorted(by_source):
        groups.setdefault(root(source), []).append(source)
    families = []
    for sources in groups.values():
        members = [by_source[s] for s in sources]
        if sources == sorted(salazar):
            fid = "F_salazar"
        elif sources == sorted(roche):
            fid = "F_roche_anticipando"
        elif sources == sorted(epigenetics):
            fid = "F_epigenetica_medicina"
        else:
            fid = "F_" + digest(sources[0].encode("utf-8"))[:12]
        families.append({"id": fid, "documents": sources,
                         "characters": sum(d["characters"] for d in members),
                         "approximate_words": sum(d["approximate_words"] for d in members),
                         "reason": "Familia respaldada por evidencia editorial/textual." if len(sources) > 1 else "Fuente independiente; sin relación editorial demostrada ni solapamiento sustancial.",
                         "evidence": [e for e in evidence if set(e["documents"]).issubset(sources)]})
    return sorted(families, key=lambda f: f["id"])


def assign(families):
    total_c = sum(f["characters"] for f in families)
    total_w = sum(f["approximate_words"] for f in families)
    # Las dos familias/libros grandes permanecen en Train (>=20% cada una).
    fixed = [f for f in families if f["characters"] / total_c >= 0.20]
    free = [f for f in families if f not in fixed]
    best, best_key, explored, valid = None, None, 0, 0
    for choices in itertools.product(range(3), repeat=len(free)):
        explored += 1
        cs = [sum(f["characters"] for f in fixed), 0, 0]
        ws = [sum(f["approximate_words"] for f in fixed), 0, 0]
        counts = [len(fixed), 0, 0]
        for f, k in zip(free, choices):
            cs[k] += f["characters"]
            ws[k] += f["approximate_words"]
            counts[k] += 1
        # Dos familias distintas por evaluación favorecen diversidad de fuentes.
        if min(counts[1:]) < 2 or any(not lo <= c / total_c <= hi for c, (lo, hi) in zip(cs, RANGES)):
            continue
        valid += 1
        score = sum((c / total_c - target)**2 + (w / total_w - target)**2 for c, w, target in zip(cs, ws, TARGET))
        key = (score, choices)  # Familia por ID; train < validation < test.
        if best_key is None or key < best_key:
            best_key, best = key, choices
    require(best is not None, "No hay asignación factible sin romper familias.")
    mapping = {f["id"]: "train" for f in fixed}
    mapping.update({f["id"]: SPLITS[k] for f, k in zip(free, best)})
    return mapping, {"search": "Enumeración exhaustiva de 3^N asignaciones de familias libres",
                     "assignments_examined": explored, "feasible_assignments": valid,
                     "objective": "Suma de desviaciones cuadráticas respecto a 80/10/10 en fracciones de caracteres y palabras, con igual peso",
                     "score": best_key[0], "tie_break": "Tupla de asignaciones por ID de familia ordenado; train < validation < test",
                     "large_families_fixed_in_train": [f["id"] for f in fixed],
                     "constraints": "Rangos de caracteres 75–85/7.5–15/7.5–15%; al menos dos familias por evaluación"}


def tokenizer_preanalysis(parts):
    # Conteos transitorios de formas superficiales, sin vocabulario ni IDs.
    counters = {s: Counter(SURFACE.findall("\n".join(d["text"] for d in docs))) for s, docs in parts.items()}
    train = counters["train"]
    result = {"expression": SURFACE.pattern, "preserves_case": True,
              "delimiters_excluded": True, "types_include_words_numbers_and_signs": True,
              "palabras_aproximadas_train": sum(d["approximate_words"] for d in parts["train"]),
              "tipos_unicos_train": len(train), "tipos_frecuencia_1": sum(n == 1 for n in train.values()),
              "porcentaje_hapax": 100 * sum(n == 1 for n in train.values()) / len(train)}
    result.update({f"tipos_frecuencia_mayor_igual_{n}": sum(v >= n for v in train.values()) for n in (2, 3, 5, 10)})
    for s in ("validation", "test"):
        unseen = counters[s].keys() - train.keys()
        result[f"tipos_{s}_no_vistos_en_train"] = len(unseen)
        result[f"tipos_totales_{s}"] = len(counters[s])
        result[f"ocurrencias_{s}_no_vistas_en_train"] = sum(counters[s][x] for x in unseen)
        result[f"porcentaje_ocurrencias_{s}_no_vistas"] = 100 * sum(counters[s][x] for x in unseen) / sum(counters[s].values())
    return result


def main():
    documents = read_inputs()
    pairs = analyze_pairs(documents)
    families = build_families(documents, pairs)
    mapping, optimization = assign(families)
    membership = {s: mapping[f["id"]] for f in families for s in f["documents"]}
    require(len(membership) == 21, "Cada fuente debe pertenecer a una sola familia.")
    parts = {s: [d for d in documents if membership[d["source"]] == s] for s in SPLITS}
    total_c = sum(d["characters"] for d in documents)
    total_w = sum(d["approximate_words"] for d in documents)
    payloads = {s: build_text(parts[s]).encode("utf-8") for s in SPLITS}
    statistics = {}
    for s in SPLITS:
        chars = sum(d["characters"] for d in parts[s])
        words = sum(d["approximate_words"] for d in parts[s])
        statistics[s] = {"characters": chars, "approximate_words": words,
                         "character_percentage": 100 * chars / total_c, "word_percentage": 100 * words / total_w,
                         "documents": len(parts[s]), "families": sum(mapping[f["id"]] == s for f in families),
                         "characters_with_delimiters": len(payloads[s].decode("utf-8")), "sha256": digest(payloads[s])}
    by_source = {d["source"]: d for d in documents}
    cross = {}
    for left, right in itertools.combinations(SPLITS, 2):
        checked = []
        for a, b in itertools.product(parts[left], parts[right]):
            # Recalcular, no reutilizar solo los valores previos al split.
            primary, secondary = pair_metrics(a, b, 8), pair_metrics(a, b, 12)
            require(a["sha256"] != b["sha256"], "Duplicado exacto cross-split.")
            checked.append({"documento_a": a["source"], "documento_b": b["source"], **primary, "secondary_12_words": secondary})
        checked.sort(key=lambda p: (-p["containment_documento_menor"], p["documento_a"], p["documento_b"]))
        require(not any(p["containment_documento_menor"] >= .05 for p in checked), "Solapamiento cross-split >=5%: requiere inspección antes de publicar salidas.")
        cross[left + "_vs_" + right] = {"pairs_recomputed": len(checked), "maximum_smaller_document_containment": checked[0],
                                         "maximum_left_in_right": max(p["containment_a_en_b"] for p in checked),
                                         "maximum_right_in_left": max(p["containment_b_en_a"] for p in checked),
                                         "pairs_requiring_inspection": [], "top_pairs": checked[:3]}
        maximum = checked[0]
        a, b = by_source[maximum["documento_a"]], by_source[maximum["documento_b"]]
        cross[left + "_vs_" + right]["maximum_pair_shared_examples"] = {
            str(n): [" ".join(t) for t in sorted(a["shingles"][n] & b["shingles"][n])[:12]] for n in (8, 12)}
        cross[left + "_vs_" + right]["residual_review"] = (
            "Train/Validation y Train/Test: frases científicas habituales sobre PCR y procedimientos; sus pares máximos no comparten shingles de 12 palabras. "
            "Validation/Test: una cita bibliográfica común de edición genómica y frases genéricas; 9 shingles de 8 palabras, 3 de 12. "
            "Muestras inspeccionadas en los TXT; no demuestran reproducción sustancial ni origen editorial compartido."
        )
    # Evidencia textual compacta de pares más altos, incluidas coincidencias residuales.
    top = sorted(pairs, key=lambda p: (-p["containment_documento_menor"], p["documento_a"], p["documento_b"]))[:10]
    for p in top:
        a, b = by_source[p["documento_a"]], by_source[p["documento_b"]]
        p["shared_12_word_examples"] = [" ".join(t) for t in sorted(a["shingles"][12] & b["shingles"][12])[:3]]
    report = {"version": "2B.1", "source_corpus_sha256": EXPECTED_CORPUS, "source_report_sha256": EXPECTED_REPORT,
              "sha256_split_code": digest(Path(__file__).read_bytes()),
              "method": "Documentos completos; componentes de obra/serie editorial y solapamiento corroborado; optimización exhaustiva por familias",
              "overlap_configuration": {"primary_shingle_words": 8, "secondary_shingle_words": 12,
                  "word_expression": WORD.pattern, "comparison_only": "NFC, minúsculas y segmentación de palabras; TXT originales intactos",
                  "family_threshold": FAMILY_THRESHOLD, "inspection_threshold": .05,
                  "threshold_justification": "Cuatro pares fuertes: 23.368–81.668%; el siguiente 1.413%. El umbral 15% queda en el intervalo vacío y los cuatro pares tienen misma obra comprobada. Series editoriales se agrupan también por evidencia textual, sin exigir umbral.",
                  "containment_denominator": "Número de shingles distintos del documento; menor significa conjunto de shingles de menor tamaño",
                  "classification": "SHA idéntico: exact_duplicate; containment menor >=15%: strong; >=5%: moderate; resto: low"},
              "inventory": [{k: v for k, v in d.items() if k not in ("text", "shingles")} | {"corpus_percentage": 100*d["characters"]/total_c} for d in documents],
              "families": [f | {"split": mapping[f["id"]]} for f in families],
              "splits": {s: {"family_ids": [f["id"] for f in families if mapping[f["id"]] == s], "sources": [d["source"] for d in parts[s]], "file": "datos/splits/"+s+".txt"} for s in SPLITS},
              "statistics": statistics, "optimization": optimization, "pairwise_metrics": pairs, "top_overlaps": top,
              "cross_split_leakage": cross, "tokenizer_preanalysis": tokenizer_preanalysis(parts),
              "integrity": {"documents_assigned_exactly_once": 21, "pairs_analyzed": len(pairs),
                            "content_characters": total_c, "split_characters_with_delimiters": sum(v["characters_with_delimiters"] for v in statistics.values()),
                            "exact_cross_split_duplicates": 0, "all_families_intact": True},
              "warnings": ["Solo 14 familias; evaluación dependiente de las fuentes seleccionadas, no de una muestra aleatoria grande.",
                           "Libro Ingeniería Genética y familia Salazar dominan Train; se conserva redundancia intrafamilia autorizada.",
                           "Las limitaciones de extracción de 2A.2 permanecen; no se aplicó OCR ni limpieza adicional.",
                           "Las estadísticas superficiales no constituyen un vocabulario ni una elección entre palabras y BPE; en 2C aprender exclusivamente de Train."]}
    require(sum(len(v) for v in parts.values()) == 21, "Documentos omitidos.")
    require(sum(v["characters"] for v in statistics.values()) == 3733434, "Contenido perdido.")
    require(sum(c["pairs_recomputed"] for c in cross.values()) == sum(membership[p["documento_a"]] != membership[p["documento_b"]] for p in pairs), "Pares cross-split incompletos.")
    # Última verificación congelada, antes de escribir únicamente las salidas 2B.
    require(digest(CORPUS.read_bytes()) == EXPECTED_CORPUS and digest(REPORT.read_bytes()) == EXPECTED_REPORT, "Entradas cambiaron durante análisis.")
    (BASE / "datos/splits").mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (BASE / "datos/splits" / (s + ".txt")).write_bytes(payloads[s])
    OUTPUT.write_bytes((json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"statistics": statistics, "sha256_splits_json": digest(OUTPUT.read_bytes()),
                      "tokenizer_preanalysis": report["tokenizer_preanalysis"]}, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
