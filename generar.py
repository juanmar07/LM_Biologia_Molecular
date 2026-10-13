"""4A: continuación causal FP32 con el checkpoint oficial 3B.1.

No es un chatbot ni ha sido instruction-tuned. La calidad factual requiere
un análisis separado. La ventana deslizante conserva los últimos 128 IDs,
renumerados 0..L-1. Defaults preestablecidos, no parámetros óptimos.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import time
import types
import torch
from tokenizador import TokenizadorEducativo

BASE = Path(__file__).resolve().parent
BEST = BASE / "modelos/checkpoint_best.pt"
TOKENIZER = BASE / "datos/tokenizer/tokenizer.json"
CHECKPOINT_SHA256 = "3bcd3074e1020380b75f16ea0917bb0074f1961ed6937d66f93f2f52618c0cde"
TOKENIZER_SHA256 = "97953e99660fdebcf071b8327cd79f9cfa62e2ba38eae85d69f9f0e718d55b7e"
DEFAULTS = dict(max_new_tokens=100, temperature=0.8, top_k=40, seed=42, device="auto")
SPECIAL_IDS = dict(PAD=0, UNK=1, BOS=2, EOS=3)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def elegir_device(device="auto"):
    mps = getattr(torch.backends, "mps", None)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if mps is not None and mps.is_available() else "cpu")
    chosen = torch.device(device)
    require(chosen.type in {"cuda", "mps", "cpu"}, "Device no soportado.")
    require(chosen.type != "cuda" or torch.cuda.is_available(), "CUDA no disponible.")
    require(chosen.type != "mps" or (mps is not None and mps.is_available()), "MPS no disponible.")
    return chosen


def cargar_modelo(checkpoint=BEST, device="auto"):
    """Devuelve (modelo, tokenizer), verificando hashes antes de cargar.

    Namespace privado: sustituye el import de config por constantes del
    checkpoint sin cambiar archivos o módulos globales. No requiere config.py.
    """
    path = Path(checkpoint).resolve()
    require(path == BEST.resolve(), "4A solo autoriza modelos/checkpoint_best.pt.")
    require(sha256(path) == CHECKPOINT_SHA256, "SHA-256 del checkpoint distinto.")
    require(sha256(TOKENIZER) == TOKENIZER_SHA256, "SHA-256 del tokenizer distinto.")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    require(saved["checkpoint_version"] == "3B.1", "Versión de checkpoint distinta.")
    require(saved["epoch_completed"] == 19 and saved["global_step"] == 14839,
            "Época/global_step del checkpoint distintos.")
    require(saved["best_validation_loss"] == 4.725705095024419, "Validation loss distinta.")
    require(not saved["smoke_only"], "Checkpoint de smoke no autorizado.")
    require(saved["tokenizer_sha256"] == TOKENIZER_SHA256, "Tokenizer de metadata distinto.")
    require(saved["special_token_ids"] == SPECIAL_IDS, "IDs especiales distintos.")
    require(sha256(BASE / "modelo.py") == saved["model_code_sha256"], "modelo.py modificado.")
    mc = saved["model_config"]
    require(all(mc[k] == v for k, v in dict(vocab_size=6000, context_length=128,
            d_model=128, n_heads=4, n_layers=3, d_ff=512, dropout=0.1,
            normalization="pre-norm", activation="GELU", weight_tying=True,
            initialization_policy="B").items()), "Arquitectura congelada incompatible.")
    constants = {name: mc[key] for name, key in {
        "LONGITUD_CONTEXTO": "context_length", "DIMENSION_EMBEDDING": "d_model",
        "NUM_CABEZAS": "n_heads", "NUM_CAPAS": "n_layers",
        "DIMENSION_FEED_FORWARD": "d_ff", "DROPOUT": "dropout"}.items()}
    source = ast.parse((BASE / "modelo.py").read_text(encoding="utf-8"))
    source.body = [node for node in source.body
                   if not (isinstance(node, ast.ImportFrom) and node.module == "config")]
    namespace = types.ModuleType("minigpt_generation_4a")
    namespace.__dict__.update(constants)
    exec(compile(source, str(BASE / "modelo.py"), "exec"), namespace.__dict__)
    # Inicialización transitoria reemplazada por strict=True; preserva RNG CPU.
    with torch.random.fork_rng(devices=[]):
        model = namespace.MiniGPT(mc["vocab_size"])
    state = saved["model_state_dict"]
    require(torch.equal(state["embedding_tokens.weight"], state["salida_vocabulario.weight"]),
            "Aliases guardados del weight tying difieren.")
    model.load_state_dict(state, strict=True)
    require(sum(p.numel() for p in model.parameters()) == 1379456, "Conteo de parámetros distinto.")
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Pesos no finitos.")
    chosen = elegir_device(device)
    model.to(device=chosen, dtype=torch.float32).eval()
    require(model.embedding_tokens.weight is model.salida_vocabulario.weight and
            model.embedding_tokens.weight.data_ptr() == model.salida_vocabulario.weight.data_ptr(),
            "Weight tying perdido.")
    model.generation_config = dict(mc)
    tokenizer = TokenizadorEducativo().cargar(TOKENIZER)
    require(tokenizer.vocab_size == mc["vocab_size"], "Vocabulario distinto.")
    return model, tokenizer


def codificar_prompt(tokenizer, prompt, agregar_bos=True):
    require(isinstance(prompt, str), "El prompt debe ser texto.")
    ids = tokenizer.encode(prompt, agregar_bos=False, agregar_eos=False)
    require(all(i >= 4 for i in ids), "El prompt no admite cadenas reservadas PAD/UNK/BOS/EOS.")
    # Vacío necesita una posición válida incluso con --no-bos.
    return ([tokenizer.bos_id] if agregar_bos or not ids else []) + ids


def seleccionar_token(logits, *, greedy, temperature, top_k, generator):
    """Bloqueo -> temperatura -> top-k exacto -> softmax -> multinomial."""
    require(logits.ndim == 1 and logits.numel() >= 4, "Logits de forma inválida.")
    require(bool(torch.isfinite(logits).all()), "Logits no finitos.")
    filtered = logits.clone()
    filtered[:3] = -torch.inf
    if greedy:
        return int(filtered.argmax().item())
    require(math.isfinite(temperature) and temperature > 0, "temperature debe ser finita y >0.")
    require(isinstance(top_k, int) and 1 <= top_k <= logits.numel() - 3,
            "top_k debe estar entre 1 y vocabulario efectivo (5997).")
    # Evitar que un escalar positivo se redondee a cero en FP32. Solo el
    # escalado de selección extremo usa FP64; el forward sigue siendo FP32.
    if temperature < torch.finfo(filtered.dtype).tiny:
        filtered = filtered.double()
    # Centrar mantiene un máximo cero incluso si otros logits pasan a -inf.
    filtered = (filtered - filtered.max()) / temperature
    values, indices = torch.topk(filtered, top_k)
    restricted = torch.full_like(filtered, -torch.inf)
    restricted.scatter_(0, indices, values)
    probabilities = torch.softmax(restricted, dim=-1)
    # RNG CPU local: también funciona en MPS sin alterar el RNG global.
    return int(torch.multinomial(probabilities.cpu(), 1, generator=generator).item())


def hash_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def diagnosticos(ids, texto):
    run = maximum = 0
    previous = None
    for token in ids:
        run = run + 1 if token == previous else 1
        maximum = max(maximum, run)
        previous = token
    return dict(tokens_generated=len(ids), eos_reached=bool(ids and ids[-1] == 3),
                forbidden_special_tokens=sum(i in (0, 1, 2) for i in ids),
                replacement_character_count=texto.count("\ufffd"),
                consecutive_identical_pairs=sum(a == b for a, b in zip(ids, ids[1:])),
                unique_token_percentage=100 * len(set(ids)) / len(ids) if ids else 0.0,
                maximum_identical_token_run=maximum)


def presentar_texto(texto):
    """Conserva hasta el último cierre de oración detectable (. ! ?).

    Heurística de presentación, no revisión gramatical ni factual. No añade
    puntuación faltante: si no existe un cierre devuelve vacío. Un cierre
    con puntos suspensivos se presenta como punto. Evita cortar por decimales,
    iniciales y abreviaturas comunes. Los IDs,
    el texto crudo y sus diagnósticos se conservan por separado.
    """
    abbreviations = {"dr", "dra", "sr", "sra", "srta", "prof", "profa",
                     "fig", "figs", "p", "pp", "ej", "vol", "núm", "cap", "et"}
    result = ""
    for match in re.finditer(r'(?<![.!?])([.!?]+)(["»”’\')\]]*)(?=\s|$)', texto):
        position = match.start()
        if not re.search(r"\w", texto[:position]):
            continue
        if match.group(1) == ".":
            word = re.search(r"(\w+)$", texto[:position])
            if word:
                value = word.group(1)
                if value.lower() in abbreviations or (len(value) == 1 and value.isalpha()):
                    continue
            # Un número seguido de punto al inicio de línea puede ser una lista.
            if re.fullmatch(r"\s*\d+", texto[:position].rsplit("\n", 1)[-1]):
                continue
        punctuation = "." if set(match.group(1)) == {"."} else match.group(1)
        result = texto[:position] + punctuation + match.group(2)
    return result.rstrip()


def sincronizar(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def normalizar_presentacion(texto):
    """Whitespace visual únicamente; no recibe ni modifica los IDs."""
    visual = texto.replace("\r\n", "\n").replace("\r", "\n")

    def linebreak(match):
        count = match.group().count("\n")
        return " " if count == 1 else "\n" * count

    visual = re.sub(r"[ \t]*\n(?:[ \t]*\n)*[ \t]*", linebreak, visual)
    return re.sub(r"[ \t]{2,}", " ", visual).lstrip()


def construir_texto_completo(prompt, presented_text):
    """Une copias visuales sin cambiar prompt ni continuación almacenados.

    Conserva un salto de párrafo inicial como separador entre ambos bloques;
    evita espacios antes de puntuación de cierre y después de signos de apertura.
    """
    left = prompt.rstrip()
    right = normalizar_presentacion(presented_text)
    if not left.strip():
        return right
    if not right:
        return left
    leading = re.match(r"\s*", presented_text.replace("\r\n", "\n").replace("\r", "\n")).group()
    paragraphs = leading.count("\n")
    if paragraphs >= 2:
        separator = "\n" * paragraphs
    elif right[0] in ".,;:!?%)]}»”’" or left[-1] in "([{¿¡«“":
        separator = ""
    else:
        separator = " "
    return left + separator + right


def contar_oraciones_presentadas(texto):
    """Conteo descriptivo con la heurística existente; no controla stopping."""
    previous = ""
    count = 0
    for match in re.finditer(r'(?<![.!?])[.!?]+["»”’\')\]]*(?=\s|$)', texto):
        closed = presentar_texto(texto[:match.end()])
        if closed and closed != previous:
            count += 1
            previous = closed
    return count


@torch.no_grad()
def generar(modelo, tokenizador, prompt, max_new_tokens=100, temperature=0.8,
            top_k=40, seed=42, greedy=False, agregar_bos=True):
    """Devuelve texto crudo/presentado, IDs, diagnósticos y tiempos.

    Decodifica IDs de continuación juntos, omitiendo EOS; UTF-8 incompleto o
    inválido usa U+FFFD. Reproducibilidad garantizada en el mismo runtime/device.
    """
    require(isinstance(max_new_tokens, int) and max_new_tokens >= 0,
            "max_new_tokens debe ser un entero >=0.")
    require(isinstance(seed, int) and 0 <= seed < 2**63, "seed fuera de rango [0, 2**63).")
    if not greedy:
        require(math.isfinite(temperature) and temperature > 0, "temperature debe ser finita y >0.")
        require(isinstance(top_k, int) and 1 <= top_k <= tokenizador.vocab_size - 3,
                "top_k fuera del vocabulario efectivo.")
    require(not modelo.training, "El modelo debe estar en eval().")
    require(all(p.dtype == torch.float32 for p in modelo.parameters()), "Inferencia requiere FP32.")
    device = next(modelo.parameters()).device
    prompt_ids = codificar_prompt(tokenizador, prompt, agregar_bos)
    sequence = list(prompt_ids)
    generated = []
    lengths = []
    limit = modelo.generation_config["context_length"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sincronizar(device)
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        window = sequence[-limit:]
        context = torch.tensor([window], dtype=torch.long, device=device)
        lengths.append(len(window))
        logits, _ = modelo(context)
        token = seleccionar_token(logits[0, -1], greedy=greedy, temperature=temperature,
                                  top_k=top_k, generator=generator)
        generated.append(token)
        sequence.append(token)
        if token == tokenizador.eos_id:
            break
    sincronizar(device)
    seconds = time.perf_counter() - start
    text = tokenizador.decode(generated, omitir_especiales=True)
    result = dict(texto=text, texto_presentado=presentar_texto(text),
                ids_generados=generated, ids_sha256=hash_ids(generated),
                prompt_ids=prompt_ids, context_lengths=lengths,
                diagnosticos=diagnosticos(generated, text), seconds=seconds,
                tokens_per_second=len(generated) / seconds if seconds else 0.0,
                device=str(device), dtype="float32", batch_size=1)
    # Campos adicionales de presentación/auditoría; los existentes conservan
    # exactamente su significado. No participan en la selección ni la parada.
    result.update(prompt=prompt, raw_text=result["texto"],
                  presented_text=result["texto_presentado"],
                  full_presented_text=construir_texto_completo(prompt, result["texto_presentado"]),
                  stop_reason="eos" if result["diagnosticos"]["eos_reached"] else "max_new_tokens",
                  tokens_generated=result["diagnosticos"]["tokens_generated"],
                  eos_reached=result["diagnosticos"]["eos_reached"],
                  sentences_completed=contar_oraciones_presentadas(result["texto_presentado"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Inicio de texto, no instrucción de chat.")
    parser.add_argument("--checkpoint", type=Path, default=BEST)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-bos", action="store_true", help="Continuación estricta; vacío usa BOS.")
    parser.add_argument("--debug", action="store_true", help="Mostrar IDs y diagnósticos.")
    parser.add_argument("--raw", action="store_true", help="Auditar prompt, continuación raw exacta, presentación y parada.")
    args = parser.parse_args()
    try:
        if elegir_device(args.device).type == "cpu":
            torch.set_num_threads(2)
        model, tokenizer = cargar_modelo(args.checkpoint, args.device)
        result = generar(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k, seed=args.seed,
                         greedy=args.greedy, agregar_bos=not args.no_bos)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    if args.raw:
        # JSON conserva espacios y saltos exactos sin confundirlos con etiquetas.
        print(json.dumps({key: result[key] for key in (
            "prompt", "raw_text", "presented_text", "full_presented_text",
            "stop_reason", "tokens_generated", "eos_reached", "sentences_completed")},
            ensure_ascii=False, indent=2))
    else:
        print(f"Texto generado:\n{result['full_presented_text']}")
        if result["raw_text"] and not result["presented_text"]:
            print("[No se generó una oración con cierre detectable.]")
    if args.debug:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
