"""3B: entrenamiento FP32 y reanudacion reproducible al limite de epocas.
--smoke-test nunca crea artefactos oficiales; usa modelos descartables.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
import time
import types
import torch
import torch.nn.functional as F
import config
from dataset import crear_dataloader

BASE = Path(__file__).resolve().parent
VERSION = "3B.1"
EXPECTED = {
    "datos/tokenizer/tokenizer.json": "97953e99660fdebcf071b8327cd79f9cfa62e2ba38eae85d69f9f0e718d55b7e",
    "datos/dataset/train.pt": "563dc91fca06407c124cc25d50d8966c0ce2457fb47099d0927b25b99a203dcc",
    "datos/dataset/validation.pt": "8fe0658708d461799ab5aaf9207aa81601bf734eab28707e1657a839be72afd9",
    "datos/dataset/test.pt": "6d4193e076f18d83bff4c0b45098a25a8bad381f3b6eae0f29cb2f49e44e4938",
    "datos/dataset/reporte_dataset.json": "a060e3abb4d4842c33543518977faf2bdb21484e5b8bde0f717d66172afe69f6",
    "modelo.py": "39229ef8813572cd0de38119f2b765e9dfa357e1016d9691e19615c3ed7850a3",
    "resultados/reporte_inicializacion_3A1.json": "e6d2f4e23d8bbf9c1c408536d14f73c04e40281e327e136cddd1a51a5da694b2",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verificar_integridad():
    """Test solo se lee como bytes para SHA256; no se deserializa."""
    hashes = {}
    for name, expected in EXPECTED.items():
        actual = sha256(BASE / name)
        if actual != expected:
            raise RuntimeError(f"Artefacto congelado incompatible: {name}: {actual} != {expected}")
        hashes[name] = actual
    return hashes


def configuracion_modelo():
    report = json.loads((BASE / "resultados/reporte_inicializacion_3A1.json").read_text(encoding="utf-8"))
    result = copy.deepcopy(report["frozen_architecture"])
    result["initialization_policy"] = "B"
    return result


def construir_modelo(model_config):
    """Configura una copia aislada del codigo congelado desde metadata guardada.
    No cambia modelo.py, el modulo original ni los globals de config.py.
    """
    if sha256(BASE / "modelo.py") != EXPECTED["modelo.py"]:
        raise RuntimeError("Codigo de modelo incompatible.")
    if model_config != configuracion_modelo():
        raise ValueError("model_config incompatible con arquitectura 3A.1.")
    namespace = types.ModuleType("minigpt_checkpoint_architecture")
    namespace.__file__ = str(BASE / "modelo.py")
    exec(compile((BASE / "modelo.py").read_text(encoding="utf-8"), namespace.__file__, "exec"), namespace.__dict__)
    names = {"LONGITUD_CONTEXTO": "context_length", "DIMENSION_EMBEDDING": "d_model",
             "NUM_CABEZAS": "n_heads", "NUM_CAPAS": "n_layers",
             "DIMENSION_FEED_FORWARD": "d_ff", "DROPOUT": "dropout"}
    for name, key in names.items():
        setattr(namespace, name, model_config[key])
    model = namespace.MiniGPT(model_config["vocab_size"])
    if sum(p.numel() for p in model.parameters()) != 1379456:
        raise RuntimeError("Numero de parametros inesperado.")
    if model.embedding_tokens.weight is not model.salida_vocabulario.weight:
        raise RuntimeError("Weight tying perdido.")
    return model


def elegir_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS no disponible.")
    return torch.device(requested)


def fijar_semilla(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def sincronizar(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def configuracion_entrenamiento(args):
    return {"max_epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
            "betas": list(args.betas), "eps": args.eps, "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.clip_norm, "early_stopping_patience": args.patience,
            "early_stopping_min_delta": args.min_delta, "seed": args.seed,
            "scheduler": "none", "precision": "fp32", "num_workers": 0, "drop_last": False,
            "cpu_threads": args.cpu_threads, "log_interval": args.log_interval}


def crear_loaders(tc):
    train = crear_dataloader("train", batch_size=tc["batch_size"], shuffle=True, seed=tc["seed"])
    val = crear_dataloader("validation", batch_size=tc["batch_size"], shuffle=False, seed=tc["seed"])
    for split, loader, windows, stride in (("train", train, 12489, 64), ("validation", val, 875, 128)):
        data = loader.dataset.data
        if len(loader.dataset) != windows or data["context_length"] != 128 or data["stride"] != stride:
            raise RuntimeError(f"Dataset {split} incompatible con 2D.")
        if data["ignore_index"] != -100 or data["vocab_size"] != 6000:
            raise RuntimeError(f"Targets de {split} incompatibles.")
    return train, val


def crear_optimizador(model, tc):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    flattened = decay + no_decay
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(flattened) != len({id(p) for p in flattened}) or {id(p) for p in flattened} != expected:
        raise RuntimeError("Parametros del optimizador duplicados o incompletos.")
    if sum(p.numel() for p in flattened) != 1379456:
        raise RuntimeError("Conteo de parametros del optimizador incorrecto.")
    return torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["learning_rate"], betas=tuple(tc["betas"]), eps=tc["eps"])


def paso_entrenamiento(model, optimizer, x, y, device, clip_norm):
    model.train()
    x, y = x.to(device), y.to(device)
    count = int((y != -100).sum().item())
    if count == 0:
        raise RuntimeError("Batch de Train sin targets validos.")
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(x)
    loss_sum = F.cross_entropy(logits.reshape(-1, 6000), y.reshape(-1), ignore_index=-100, reduction="sum")
    loss = loss_sum / count
    if not torch.isfinite(loss).item():
        raise RuntimeError("Train loss no finita.")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("Pesos no finitos despues de AdamW.")
    return {"ce_sum": loss_sum.detach().item(), "targets": count, "input_tokens": x.numel(),
            "gradient_norm_before_clipping": norm.item(), "batch_loss": loss.detach().item()}


def perplexity(loss):
    try:
        value = math.exp(loss)
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def evaluar(model, loader, device):
    if getattr(loader.dataset, "split", None) != "validation":
        raise ValueError("3B solo permite evaluar Validation.")
    was_training = model.training
    model.eval()
    ce_sum, targets, windows, batches = 0.0, 0, 0, 0
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                count = int((y != -100).sum().item())
                windows += x.size(0)
                batches += 1
                if count == 0:
                    continue
                logits, _ = model(x)
                loss_sum = F.cross_entropy(logits.reshape(-1, 6000), y.reshape(-1), ignore_index=-100, reduction="sum")
                if not torch.isfinite(loss_sum).item():
                    raise RuntimeError("Validation loss no finita.")
                ce_sum += loss_sum.item()
                targets += count
    finally:
        model.train(was_training)
    if targets != 111644 or windows != 875:
        raise RuntimeError(f"Validation incompleta: {windows} ventanas, {targets} targets.")
    loss = ce_sum / targets
    return {"loss": loss, "perplexity": perplexity(loss), "ce_sum": ce_sum, "targets": targets,
            "windows": windows, "batches": batches, "seconds": time.perf_counter() - start}


def escribir_atomico(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def guardar_json_atomico(path, data):
    escribir_atomico(path, lambda temp: temp.write_text(json.dumps(
        data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"))


def crear_checkpoint(model, optimizer, epoch, global_step, best, counter, mc, tc, history, loaders, device, hashes, smoke_only=False):
    return {"checkpoint_version": VERSION, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "epoch_completed": epoch,
            "global_step": global_step, "best_validation_loss": best, "early_stopping_counter": counter,
            "model_config": copy.deepcopy(mc), "training_config": copy.deepcopy(tc),
            "initialization_policy": "B", "special_token_ids": mc["special_token_ids"],
            "tokenizer_sha256": hashes["datos/tokenizer/tokenizer.json"],
            "train_dataset_sha256": hashes["datos/dataset/train.pt"],
            "validation_dataset_sha256": hashes["datos/dataset/validation.pt"],
            "model_code_sha256": hashes["modelo.py"], "training_code_sha256": sha256(Path(__file__)),
            "seed": tc["seed"], "pytorch_version": str(torch.__version__), "device_type": device.type,
            "history": copy.deepcopy(history), "torch_rng_state": torch.get_rng_state(),
            "python_random_state": random.getstate(),
            "dataloader_generator_states": {"train": loaders[0].generator.get_state(),
                                           "validation": loaders[1].generator.get_state()},
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "mps_rng_state": torch.mps.get_rng_state() if device.type == "mps" else None,
            "smoke_only": smoke_only}


def guardar_checkpoint(path, checkpoint):
    escribir_atomico(path, lambda temporary: torch.save(checkpoint, temporary))


def cargar_checkpoint(path, device, allow_smoke=False):
    hashes = verificar_integridad()
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if cp.get("checkpoint_version") != VERSION:
        raise RuntimeError("Checkpoint incompatible: se requiere version 3B.1.")
    required = {"model_state_dict", "optimizer_state_dict", "epoch_completed", "global_step",
                "best_validation_loss", "early_stopping_counter", "model_config", "training_config",
                "initialization_policy", "special_token_ids", "tokenizer_sha256", "train_dataset_sha256",
                "validation_dataset_sha256", "model_code_sha256", "training_code_sha256", "seed",
                "pytorch_version", "device_type", "history", "torch_rng_state", "python_random_state",
                "dataloader_generator_states", "cuda_rng_states", "mps_rng_state", "smoke_only"}
    if required - cp.keys():
        raise RuntimeError(f"Checkpoint incompleto: faltan {sorted(required - cp.keys())}.")
    if cp.get("smoke_only") and not allow_smoke:
        raise RuntimeError("Checkpoint de smoke test prohibido en produccion.")
    pairs = {"tokenizer_sha256": "datos/tokenizer/tokenizer.json", "train_dataset_sha256": "datos/dataset/train.pt",
             "validation_dataset_sha256": "datos/dataset/validation.pt", "model_code_sha256": "modelo.py"}
    for key, path in pairs.items():
        if cp.get(key) != hashes[path]:
            raise RuntimeError(f"Checkpoint incompatible: {key}.")

    if cp.get("training_code_sha256") != sha256(Path(__file__)):
        raise RuntimeError("Codigo de entrenamiento distinto; resume no autorizado silenciosamente.")
    if cp["initialization_policy"] != "B" or cp["special_token_ids"] != {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3}:
        raise RuntimeError("Inicializacion o tokens especiales incompatibles.")
    if cp["seed"] != cp["training_config"]["seed"] or cp["pytorch_version"] != str(torch.__version__):
        raise RuntimeError("Seed inconsistente o version de PyTorch distinta.")
    model = construir_modelo(cp["model_config"]).to(device)
    model.load_state_dict(cp["model_state_dict"], strict=True)
    optimizer = crear_optimizador(model, cp["training_config"])
    optimizer.load_state_dict(cp["optimizer_state_dict"])
    loaders = crear_loaders(cp["training_config"])
    for split, loader in zip(("train", "validation"), loaders):
        loader.generator.set_state(cp["dataloader_generator_states"][split])
    torch.set_rng_state(cp["torch_rng_state"])
    random.setstate(cp["python_random_state"])
    if cp["device_type"] == device.type == "cuda":
        torch.cuda.set_rng_state_all(cp["cuda_rng_states"])
    elif cp["device_type"] == device.type == "mps":
        torch.mps.set_rng_state(cp["mps_rng_state"])
    elif cp["device_type"] != device.type:
        print("Advertencia: cambio de device; no se garantiza reproduccion numerica exacta.")
    return model, optimizer, loaders, cp


def actualizar_early_stopping(value, best, counter, min_delta):
    if not math.isfinite(value):
        raise RuntimeError("Validation loss no finita para early stopping.")
    return (value, 0, True) if value <= best - min_delta else (best, counter + 1, False)


def benchmark_batches(tc, device):
    rows = []
    for batch in (8, 16, 32):
        candidate = dict(tc, batch_size=batch)
        fijar_semilla(42)
        model = construir_modelo(configuracion_modelo()).to(device)
        optimizer = crear_optimizador(model, candidate)
        loader = crear_dataloader("train", batch_size=batch, shuffle=True, seed=42)
        iterator = iter(loader)
        elapsed, tokens, targets = [], [], []
        try:
            for step in range(7):
                x, y = next(iterator)
                sincronizar(device)
                start = time.perf_counter()
                result = paso_entrenamiento(model, optimizer, x, y, device, candidate["gradient_clip_norm"])
                sincronizar(device)
                if step >= 2:
                    elapsed.append(time.perf_counter() - start)
                    tokens.append(result["input_tokens"])
                    targets.append(result["targets"])
            median = statistics.median(elapsed)
            rows.append({"batch_size": batch, "warmup_steps": 2, "measured_steps": 5,
                         "median_step_seconds": median, "input_tokens_per_second": statistics.median(tokens) / median,
                         "targets_per_second": statistics.median(targets) / median,
                         "stable": True, "OOM": False, "process_memory_bytes": None, "model_discarded": True})
        except (torch.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower() and not isinstance(error, torch.OutOfMemoryError):
                raise
            rows.append({"batch_size": batch, "stable": False, "OOM": True, "error": str(error)})
        finally:
            del model, optimizer, iterator, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
    r = {row["batch_size"]: row for row in rows}
    if r[16]["stable"]:
        if r[32]["stable"] and r[32]["input_tokens_per_second"] >= 1.1 * r[16]["input_tokens_per_second"]:
            selected, reason = 32, "32 mejora throughput al menos 10% frente a 16."
        elif r[8]["stable"] and r[8]["input_tokens_per_second"] > 1.2 * r[16]["input_tokens_per_second"]:
            selected, reason = 8, "16 degrada throughput mas de 20% frente a 8."
        else:
            selected, reason = 16, "16 estable; 32 no alcanza mejora de throughput del 10%."
    elif r[8]["stable"]:
        selected, reason = 8, "16 no es operativo; fallback a 8."
    else:
        raise RuntimeError("Ningun batch baseline operativo.")
    return {"candidates": rows, "selected_batch_size": selected, "selection_reason": reason,
            "selection_uses_loss": False, "optimizer_steps": 21, "process_memory_measured": False}


def estados_iguales(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(estados_iguales(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(estados_iguales(x, y) for x, y in zip(a, b))
    return a == b


def smoke_tests(tc, device):
    hashes, mc = verificar_integridad(), configuracion_modelo()
    overfit_tc = dict(tc, batch_size=min(4, tc["batch_size"]))
    fijar_semilla(42)
    model = construir_modelo(mc).to(device)
    optimizer = crear_optimizador(model, overfit_tc)
    x, y = next(iter(crear_dataloader("train", batch_size=overfit_tc["batch_size"], shuffle=True, seed=42)))
    model.eval()
    with torch.no_grad():
        initial = model(x.to(device), y.to(device))[1].item()
    for _ in range(40):
        paso_entrenamiento(model, optimizer, x, y, device, tc["gradient_clip_norm"])
    model.eval()
    with torch.no_grad():
        final = model(x.to(device), y.to(device))[1].item()
    reduction = (initial - final) / initial
    if not math.isfinite(final) or reduction < .10:
        raise RuntimeError(f"Overfit insuficiente: {initial} -> {final}")
    overfit = {"batch_size": overfit_tc["batch_size"], "optimizer_steps": 40, "initial_eval_loss": initial,
               "final_eval_loss": final, "relative_reduction": reduction, "passed": True, "model_discarded": True}
    del model, optimizer
    fijar_semilla(42)
    model = construir_modelo(mc).to(device)
    optimizer = crear_optimizador(model, tc)
    loaders = crear_loaders(tc)
    ce_sum, targets, norm = 0.0, 0, 0.0
    iterator = iter(loaders[0])
    for _ in range(10):
        x, y = next(iterator)
        step = paso_entrenamiento(model, optimizer, x, y, device, tc["gradient_clip_norm"])
        ce_sum += step["ce_sum"]
        targets += step["targets"]
        norm += step["gradient_norm_before_clipping"]
    validation = evaluar(model, loaders[1], device)
    # Synthetic boundary for roundtrip: no Train epoch is claimed completed.
    history = [{"smoke_only": True, "train_loss": ce_sum / targets, "validation_loss": validation["loss"]}]
    checkpoint = crear_checkpoint(model, optimizer, 0, 10, validation["loss"], 2,
                                 mc, tc, history, loaders, device, hashes, smoke_only=True)
    model.eval()
    with torch.no_grad():
        reference_logits = model(x.to(device))[0].cpu()
    with tempfile.TemporaryDirectory(prefix="minigpt_3B_smoke_") as directory:
        path = Path(directory) / "temporary_checkpoint.pt"
        guardar_checkpoint(path, checkpoint)
        expected_torch = torch.rand(8)
        expected_python = [random.random() for _ in range(4)]
        expected_next_batch = next(iter(loaders[0]))
        names = ("LONGITUD_CONTEXTO", "DIMENSION_EMBEDDING", "NUM_CABEZAS", "NUM_CAPAS", "DIMENSION_FEED_FORWARD", "DROPOUT")
        saved = {name: getattr(config, name) for name in names}
        try:
            for name, value in zip(names, (64, 32, 2, 1, 64, .5)):
                setattr(config, name, value)
            loaded, restored_optimizer, restored_loaders, restored = cargar_checkpoint(path, device, allow_smoke=True)
        finally:
            for name, value in saved.items():
                setattr(config, name, value)
        assert estados_iguales(model.state_dict(), loaded.state_dict())
        assert estados_iguales(optimizer.state_dict(), restored_optimizer.state_dict())
        assert estados_iguales(torch.rand(8), expected_torch)
        assert [random.random() for _ in range(4)] == expected_python
        assert estados_iguales(next(iter(restored_loaders[0])), expected_next_batch)
        loaded.eval()
        with torch.no_grad():
            loaded_logits = loaded(x.to(device))[0].cpu()
        assert torch.equal(reference_logits, loaded_logits)
        assert restored["epoch_completed"] == 0 and restored["global_step"] == 10
        assert restored["early_stopping_counter"] == 2 and restored["history"] == history
        assert restored["best_validation_loss"] == validation["loss"]
        rejected = []
        for field in ("tokenizer_sha256", "train_dataset_sha256", "validation_dataset_sha256"):
            altered = dict(checkpoint, **{field: "invalid"})
            guardar_checkpoint(path, altered)
            try:
                cargar_checkpoint(path, device, allow_smoke=True)
            except RuntimeError as error:
                assert field in str(error)
                rejected.append(field)
            else:
                raise AssertionError("Checkpoint incompatible aceptado.")
        guardar_checkpoint(path, checkpoint)
        try:
            cargar_checkpoint(path, device)
        except RuntimeError as error:
            assert "smoke test" in str(error)
        else:
            raise AssertionError("Smoke checkpoint aceptado en produccion.")
    resume = {"model_parameters_exact": True, "eval_logits_exact": True, "optimizer_state_exact": True,
              "torch_RNG_exact": True, "python_RNG_exact": True, "next_epoch_shuffle_exact": True,
              "history_preserved": True, "global_step": 10, "epoch_completed": 0, "next_epoch": 1,
              "early_stopping_counter": 2, "stored_config_overrides_future_globals": True,
              "incompatible_hashes_rejected": rejected, "smoke_checkpoint_rejected_in_production": True,
              "temporary_checkpoint_deleted": not path.exists(), "synthetic_epoch_boundary": True}
    result = {"optimizer_steps": 10, "train_loss": ce_sum / targets, "train_perplexity": perplexity(ce_sum / targets),
              "train_targets": targets, "gradient_norm_mean_before_clipping": norm / 10,
              "all_gradients_finite": True, "validation": validation, "train_epoch_completed": False,
              "no_production_outputs": True, "models_discarded": True}
    del model, optimizer, loaded, restored_optimizer, checkpoint, iterator
    return {"one_batch_overfit_test": overfit, "end_to_end_smoke_test": result, "resume_test": resume}


def entrenar(args):
    hashes = verificar_integridad()
    device = elegir_device(args.device)
    tc = configuracion_entrenamiento(args)
    torch.set_num_threads(tc["cpu_threads"])
    fijar_semilla(args.seed)
    if args.smoke_test:
        if args.resume:
            raise ValueError("--smoke-test y --resume incompatibles.")
        return smoke_tests(tc, device)
    best_path, last_path = BASE / "modelos/checkpoint_best.pt", BASE / "modelos/checkpoint_last.pt"
    history_path = BASE / "resultados/historial_entrenamiento.json"
    if args.resume:
        model, optimizer, loaders, checkpoint = cargar_checkpoint(args.resume, device)
        restored_tc = checkpoint["training_config"]
        if any(tc[k] != restored_tc[k] for k in tc if k not in ("max_epochs", "log_interval")):
            raise ValueError("Resume requiere los mismos parametros: paselos por CLI.")
        mc, history = checkpoint["model_config"], checkpoint["history"]
        start_epoch, global_step = checkpoint["epoch_completed"] + 1, checkpoint["global_step"]
        best, counter = checkpoint["best_validation_loss"], checkpoint["early_stopping_counter"]
    else:
        existing = [p for p in (best_path, last_path, history_path) if p.exists()]
        if existing and not args.overwrite:
            raise FileExistsError("Ya existen artefactos oficiales: use --resume o --overwrite explicito.")
        if existing:
            print("Advertencia: --overwrite reemplazara los artefactos oficiales anteriores.")
        mc = configuracion_modelo()
        model = construir_modelo(mc).to(device)
        optimizer = crear_optimizador(model, tc)
        loaders = crear_loaders(tc)
        history, start_epoch, global_step, best, counter = [], 1, 0, float("inf"), 0
    if counter >= tc["early_stopping_patience"]:
        print("Checkpoint ya alcanzo early stopping; no se ejecutan mas pasos.")
        return history
    print(f"Device={device}; FP32; batch={tc['batch_size']}; batches/epoca={len(loaders[0])}; seed={tc['seed']}")
    for epoch in range(start_epoch, tc["max_epochs"] + 1):
        started = time.perf_counter()
        ce_sum, targets, tokens, grad_norm, batches = 0.0, 0, 0, 0.0, 0
        for x, y in loaders[0]:

            step = paso_entrenamiento(model, optimizer, x, y, device, tc["gradient_clip_norm"])
            ce_sum += step["ce_sum"]
            targets += step["targets"]
            tokens += step["input_tokens"]
            grad_norm += step["gradient_norm_before_clipping"]
            global_step += 1
            batches += 1
            if args.verbose or (tc["log_interval"] > 0 and batches % tc["log_interval"] == 0):
                print(f"Epoca {epoch}: batch {batches}/{len(loaders[0])}, Train loss={ce_sum/targets:.4f}")
        sincronizar(device)
        train_seconds = time.perf_counter() - started
        if targets != 1598592 or batches != len(loaders[0]):
            raise RuntimeError("Epoca de Train incompleta.")
        validation = evaluar(model, loaders[1], device)
        best, counter, improved = actualizar_early_stopping(validation["loss"], best, counter, tc["early_stopping_min_delta"])
        train_loss = ce_sum / targets
        row = {"epoch": epoch, "train_loss": train_loss, "train_perplexity": perplexity(train_loss),
               "validation_loss": validation["loss"], "validation_perplexity": validation["perplexity"],
               "learning_rate": optimizer.param_groups[0]["lr"], "global_step": global_step,
               "train_targets": targets, "validation_targets": validation["targets"], "batches": batches,
               "validation_batches": validation["batches"], "duration_seconds": time.perf_counter() - started,
               "train_seconds": train_seconds, "input_tokens_per_second": tokens / train_seconds,
               "targets_per_second": targets / train_seconds, "gradient_norm_mean": grad_norm / batches,
               "best": improved, "early_stopping_counter": counter}
        history.append(row)
        checkpoint = crear_checkpoint(model, optimizer, epoch, global_step, best, counter, mc, tc,
                                     history, loaders, device, hashes)
        if improved:
            guardar_checkpoint(best_path, checkpoint)
        guardar_checkpoint(last_path, checkpoint)
        guardar_json_atomico(history_path, history)
        train_ppl = str(round(row["train_perplexity"], 2)) if row["train_perplexity"] is not None else "overflow"
        val_ppl = str(round(row["validation_perplexity"], 2)) if row["validation_perplexity"] is not None else "overflow"
        print(f"Epoca {epoch}/{tc['max_epochs']} | Train {train_loss:.4f} PPL={train_ppl} | "
              f"Val {validation['loss']:.4f} PPL={val_ppl} | lr={row['learning_rate']:.2g} | "
              f"grad={row['gradient_norm_mean']:.3f} | batches={batches} targets={targets}/{validation['targets']} | "
              f"{row['duration_seconds']:.1f}s {row['input_tokens_per_second']:.0f} tokens/s | best={improved}")
        if counter >= tc["early_stopping_patience"]:
            print("Early stopping basado en Validation.")
            break
    return history


def crear_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.EPOCAS)
    parser.add_argument("--batch-size", type=int, default=config.TAMANIO_BATCH)
    parser.add_argument("--lr", type=float, default=config.TASA_APRENDIZAJE)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=config.SEMILLA)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--betas", type=float, nargs=2, default=config.BETAS_ADAMW)
    parser.add_argument("--eps", type=float, default=config.EPS_ADAMW)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--clip-norm", type=float, default=config.GRADIENT_CLIP_NORM)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=config.EARLY_STOPPING_MIN_DELTA)
    parser.add_argument("--cpu-threads", type=int, default=config.CPU_THREADS)
    parser.add_argument("--log-interval", type=int, default=config.MOSTRAR_CADA)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main():
    parser = crear_parser()
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20 or args.batch_size < 1 or args.cpu_threads < 1 or args.patience < 1:
        parser.error("epochs en [1,20]; batch-size, cpu-threads y patience positivos.")
    if any(not math.isfinite(v) for v in (args.lr, args.eps, args.clip_norm, args.min_delta, args.weight_decay)):
        parser.error("Hiperparametros deben ser finitos.")
    if args.lr <= 0 or args.eps <= 0 or args.clip_norm <= 0 or args.min_delta < 0 or args.weight_decay < 0:
        parser.error("Hiperparametros fuera de rango.")
    if not all(0 <= beta < 1 for beta in args.betas):
        parser.error("betas en [0,1).")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    result = entrenar(args)
    if args.smoke_test:
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


