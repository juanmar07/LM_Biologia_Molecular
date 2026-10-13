"""3C: evaluacion del best seleccionado por Validation; sin entrenamiento.
Test se deserializa exclusivamente despues de superar Validation Gate.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import types
import torch
import torch.nn.functional as F
from dataset import DatasetSecuencias, crear_dataloader
from tokenizador import TokenizadorEducativo

BASE = Path(__file__).resolve().parent
BEST = BASE / "modelos/checkpoint_best.pt"
REPORT = BASE / "resultados/reporte_evaluacion_3C.json"
EXPECTED = {
 "datos/tokenizer/tokenizer.json": "97953e99660fdebcf071b8327cd79f9cfa62e2ba38eae85d69f9f0e718d55b7e",
 "datos/dataset/train.pt": "563dc91fca06407c124cc25d50d8966c0ce2457fb47099d0927b25b99a203dcc",
 "datos/dataset/validation.pt": "8fe0658708d461799ab5aaf9207aa81601bf734eab28707e1657a839be72afd9",
 "datos/dataset/test.pt": "6d4193e076f18d83bff4c0b45098a25a8bad381f3b6eae0f29cb2f49e44e4938",
 "datos/dataset/reporte_dataset.json": "a060e3abb4d4842c33543518977faf2bdb21484e5b8bde0f717d66172afe69f6",
 "resultados/reporte_entrenamiento_3B.json": "0be6240aab092f90ef0420cf0a64485147e9692f9739347ad7e45b4a78ff2e41",
 "modelo.py": "39229ef8813572cd0de38119f2b765e9dfa357e1016d9691e19615c3ed7850a3",
 "resultados/reporte_inicializacion_3A1.json": "e6d2f4e23d8bbf9c1c408536d14f73c04e40281e327e136cddd1a51a5da694b2",
}
EXPECTED_DOCUMENTS = [
 "13_T\u00c9CNICAS+MOLECULARES+APLICADAS.pdf",
 "CAP\u00cdTULO 2_ Las bases qu\u00edmicas de la vida.pdf",
 "articulo expresion genica 2018.pdf",
]
HISTORICAL_LOSS = 4.725705095024419
CHECKPOINT_FIELDS = {
 "checkpoint_version", "model_state_dict", "optimizer_state_dict", "epoch_completed", "global_step",
 "best_validation_loss", "early_stopping_counter", "model_config", "training_config", "initialization_policy",
 "special_token_ids", "tokenizer_sha256", "train_dataset_sha256", "validation_dataset_sha256", "model_code_sha256",
 "training_code_sha256", "seed", "pytorch_version", "device_type", "history", "torch_rng_state",
 "python_random_state", "dataloader_generator_states", "cuda_rng_states", "mps_rng_state", "smoke_only",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def inventory():
    return {p.relative_to(BASE).as_posix(): sha256(p) for p in BASE.rglob("*") if p.is_file()}


def weights_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(json.dumps(list(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def choose_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    require(requested != "cuda" or torch.cuda.is_available(), "CUDA no disponible.")
    require(requested != "mps" or torch.backends.mps.is_available(), "MPS no disponible.")
    return torch.device(requested)


def integrity_and_checkpoint(path, before):
    require(path.resolve() == BEST.resolve(), "3C solo autoriza checkpoint_best.pt.")
    for name, expected in EXPECTED.items():
        require(before.get(name) == expected, f"Artefacto congelado incompatible: {name}")
    history = json.loads((BASE / "resultados/historial_entrenamiento.json").read_text(encoding="utf-8"))
    require(len(history) == 20 and [r["epoch"] for r in history] == list(range(1, 21)), "Historial no contiene las 20 epocas oficiales.")
    best = min(history, key=lambda row: row["validation_loss"])
    require(best["epoch"] == 19 and abs(best["validation_loss"] - HISTORICAL_LOSS) <= 1e-10, "Mejor epoca historica incompatible.")
    require(history[-1]["validation_loss"] > best["validation_loss"] and not history[-1]["best"], "Epoca 20 no debe mejorar best.")
    # Reproduce the original min_delta policy over the complete history.
    selected, current, counter = None, float("inf"), 0
    for row in history:
        improved = row["validation_loss"] <= current - .001
        require(row["best"] == improved, "Indicador best incompatible con politica previa.")
        if improved:
            selected, current, counter = row["epoch"], row["validation_loss"], 0
        else:
            counter += 1
        require(row["early_stopping_counter"] == counter, "Contador historico incompatible.")
    require(selected == 19, "Politica original no selecciona epoca 19.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(CHECKPOINT_FIELDS <= checkpoint.keys(), "Checkpoint 3B incompleto.")
    require(checkpoint["checkpoint_version"] == "3B.1" and not checkpoint["smoke_only"], "Checkpoint no oficial o version distinta.")
    require(checkpoint["epoch_completed"] == 19 and checkpoint["global_step"] == 14839, "Estado del best incompatible.")
    require(checkpoint["early_stopping_counter"] == 0, "Best no reinicia contador.")
    require(abs(checkpoint["best_validation_loss"] - HISTORICAL_LOSS) <= 1e-10, "Best loss incompatible.")
    require(checkpoint["history"] == history[:19], "Historial dentro del checkpoint no coincide.")
    initialization = json.loads((BASE / "resultados/reporte_inicializacion_3A1.json").read_text(encoding="utf-8"))
    expected_config = dict(initialization["frozen_architecture"], initialization_policy="B")
    require(checkpoint["model_config"] == expected_config, "Arquitectura o inicializacion no congeladas.")
    require(checkpoint["initialization_policy"] == "B" and checkpoint["seed"] == 42, "Policy/seed incompatibles.")
    require(checkpoint["special_token_ids"] == {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3}, "IDs especiales distintos.")
    for key, name in {
        "tokenizer_sha256": "datos/tokenizer/tokenizer.json", "train_dataset_sha256": "datos/dataset/train.pt",
        "validation_dataset_sha256": "datos/dataset/validation.pt", "model_code_sha256": "modelo.py",
        "training_code_sha256": "entrenar.py",
    }.items():
        require(checkpoint[key] == before[name], f"Checkpoint incompatible: {key}")
    report_3B = json.loads((BASE / "resultados/reporte_entrenamiento_3B.json").read_text(encoding="utf-8"))
    report_2D = json.loads((BASE / "datos/dataset/reporte_dataset.json").read_text(encoding="utf-8"))
    require(report_2D["sha256_dataset_code"] == before["dataset.py"], "Dataset code distinto del congelado 2D.")
    require(report_3B["production_training_configuration"] == checkpoint["training_config"], "Training config distinta de 3B.")
    require(report_3B["scope_verification"]["training_code_sha256"] == before["entrenar.py"], "Training code distinto de metadata 3B.")
    require(report_3B["verified_hashes"]["modelo.py"]["actual"] == before["modelo.py"], "Model code distinto de metadata 3B.")
    require(report_3B["test_frozen"]["used_for_early_stopping"] is False, "Test intervino en seleccion previa.")
    # Discard optimizer tensors loaded as part of the required file; no optimizer is instantiated.
    del checkpoint["optimizer_state_dict"]
    verified = {name: {"sha256": before[name], "verified": True} for name in (
        *EXPECTED, "entrenar.py", "dataset.py", "tokenizador.py", "config.py",
        "modelos/checkpoint_best.pt", "modelos/checkpoint_last.pt", "resultados/historial_entrenamiento.json",
        "resultados/reporte_entrenamiento_3B.json")}
    return checkpoint, history, verified


def reconstruct(checkpoint, device):
    mc = checkpoint["model_config"]
    namespace = types.ModuleType("minigpt_final_evaluation")
    namespace.__file__ = str(BASE / "modelo.py")
    exec(compile((BASE / "modelo.py").read_text(encoding="utf-8"), namespace.__file__, "exec"), namespace.__dict__)
    for constant, key in {"LONGITUD_CONTEXTO": "context_length", "DIMENSION_EMBEDDING": "d_model",
                          "NUM_CABEZAS": "n_heads", "NUM_CAPAS": "n_layers",
                          "DIMENSION_FEED_FORWARD": "d_ff", "DROPOUT": "dropout"}.items():
        setattr(namespace, constant, mc[key])
    torch.manual_seed(checkpoint["seed"])
    model = namespace.MiniGPT(mc["vocab_size"])
    state = checkpoint["model_state_dict"]
    require(torch.equal(state["embedding_tokens.weight"], state["salida_vocabulario.weight"]), "Aliases guardados de weight tying difieren.")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    require(sum(p.numel() for p in model.parameters() if p.requires_grad) == 1379456, "Numero de parametros distinto.")
    require(model.embedding_tokens.weight is model.salida_vocabulario.weight, "Weight tying perdido.")
    require(model.embedding_tokens.weight.data_ptr() == model.salida_vocabulario.weight.data_ptr(), "Storage no compartido.")
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Pesos no finitos.")
    return model


def verify_dataset(ds, split):
    windows, targets, documents = {"validation": (875, 111644, 4), "test": (812, 103858, 3)}[split]
    require(len(ds) == windows and len(ds.documents) == documents, f"Estructura de {split} incompatible.")
    require(ds.data["stride"] == 128 and ds.context == 128 and ds.data["evaluation_unique_targets"], "Protocolo 2D incompatible.")
    require(ds.data["ignore_index"] == -100 and ds.data["vocab_size"] == 6000, "Metadata de targets incompatible.")
    if split == "test":
        require(sorted(d["source"] for d in ds.documents) == sorted(EXPECTED_DOCUMENTS), "Documentos Test inesperados.")
    coverage = [torch.zeros(len(d["tokens"]), dtype=torch.int32) for d in ds.documents]
    eos = [0] * documents
    for row, (doc, start, prefix) in enumerate(ds.windows.tolist()):
        x, y = ds[row]
        valid = y != -100
        require(x.dtype == y.dtype == torch.long and x.shape == y.shape == (128,), "XY incompatibles.")
        require(bool(((x >= 0) & (x < 6000)).all()), "Input ID invalido.")
        require(bool(((y[valid] >= 0) & (y[valid] < 6000) & (y[valid] != 2) & (y[valid] != 0)).all()), "BOS/PAD o ID invalido como target.")
        length = min(128, len(ds.documents[doc]["tokens"]) - start - 1)
        require(bool((y[:prefix] == -100).all()) and bool((y[length:] == -100).all()), "Mascara de targets incorrecta.")
        require(torch.equal(y[prefix:length], ds.documents[doc]["tokens"][start+1+prefix:start+1+length].long()), "Next-token shift incorrecto.")
        coverage[doc][start+1+prefix:start+1+length] += 1
        eos[doc] += int((y[valid] == 3).sum())
    require(all(int(c[0]) == 0 and bool((c[1:] == 1).all()) for c in coverage), "Targets repetidos u omitidos.")
    require(eos == [1] * documents, "EOS no contribuye exactamente una vez por documento.")
    require(sum(int(c.sum()) for c in coverage) == targets, "Conteo de targets incompatible.")
    return {"unique_targets": targets, "windows": windows, "documents": documents,
            "every_target_once": True, "BOS_never_target": True, "EOS_targets_per_document": eos}


def validation_gate(model, device, tolerance):
    loader = crear_dataloader("validation", batch_size=16, shuffle=False, seed=42)
    structural = verify_dataset(loader.dataset, "validation")
    start, ce_sum, targets, windows = time.perf_counter(), 0., 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            summed = F.cross_entropy(logits.reshape(-1, 6000), y.reshape(-1), ignore_index=-100, reduction="sum")
            require(torch.isfinite(summed).item(), "Validation CE no finita.")
            ce_sum += summed.item()
            targets += int((y != -100).sum().item())
            windows += x.size(0)
    loss = ce_sum / targets
    difference = abs(loss - HISTORICAL_LOSS)
    require(windows == 875 and targets == 111644, "Validation Gate incompleto.")
    require(difference <= tolerance, f"Validation Gate falla: delta={difference} > {tolerance}; Test no fue abierto.")
    elapsed = time.perf_counter() - start
    return {"passed": True, "loss": loss, "perplexity": math.exp(loss), "cross_entropy_sum": ce_sum,

            "targets": targets, "windows": windows, "batches": len(loader), "historical_loss": HISTORICAL_LOSS,
            "absolute_difference": difference, "tolerance_set_before_Test": tolerance,
            "seconds": elapsed, "targets_per_second": targets / elapsed, "structural_integrity": structural}


def selection_plan(ds):
    # Fixed rule, independent of predictions/loss: middle window per document,
    # plus quarter window of doc0 and three-quarter window of doc1 (five total).
    selected = {}
    for doc in range(3):
        rows = (ds.windows[:, 0] == doc).nonzero().flatten().tolist()
        selected[rows[len(rows)//2]] = doc
        if doc < 2:
            selected[rows[len(rows)//4 if doc == 0 else 3*len(rows)//4]] = doc
    return selected


def token_description(tokenizer, token):
    return {"id": token, "text": tokenizer.decode([token], omitir_especiales=False),
            "bytes_hex": tokenizer.pieces[token-4].hex() if token >= 4 else None}


def test_pass(model, loader, device, tokenizer, collect_examples):
    ds = loader.dataset
    plan = selection_plan(ds) if collect_examples else {}
    rows_by_doc = [{"source": d["source"], "windows": 0, "targets": 0, "CE_rows": [],
                    "top1_correct": 0, "top5_correct": 0} for d in ds.documents]
    all_rows, examples, cursor = [], [], 0
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x.to(device))
            require(torch.isfinite(logits).all().item(), "Logits Test no finitos.")
            # Per-target CE in FP32, accumulated in CPU FP64 for grouping-independent totals.
            per_target = F.cross_entropy(logits.reshape(-1, 6000), y.to(device).reshape(-1),
                                         ignore_index=-100, reduction="none").reshape(y.shape).cpu().double()
            predictions = logits.topk(5, dim=-1).indices.cpu()
            require(torch.isfinite(per_target).all().item(), "Test NLL no finita.")
            for local in range(x.size(0)):
                row = cursor + local
                doc = int(ds.windows[row, 0])
                valid = y[local] != -100
                ce = per_target[local, valid].sum().item()
                count = int(valid.sum())
                top1 = int((predictions[local, valid, 0] == y[local, valid]).sum())
                top5 = int((predictions[local, valid] == y[local, valid].unsqueeze(1)).any(-1).sum())
                r = rows_by_doc[doc]
                r["windows"] += 1; r["targets"] += count; r["CE_rows"].append(ce)
                r["top1_correct"] += top1; r["top5_correct"] += top5
                all_rows.append(ce)
                if row in plan:
                    positions = valid.nonzero().flatten()
                    position = int(positions[len(positions)//2])
                    prob = torch.softmax(logits[local, position], dim=-1)
                    probabilities, indices = prob.topk(5)
                    brief = tokenizer.decode(x[local, max(0, position-11):position+1].tolist())[-80:]
                    examples.append({"document": ds.documents[doc]["source"], "window_index": row,
                        "position_in_window": position, "absolute_target_position": int(ds.windows[row,1])+position+1,
                        "context_at_most_12_tokens_80_chars": brief,
                        "target": token_description(tokenizer, int(y[local,position])),
                        "top5": [dict(token_description(tokenizer, int(t)), probability=float(p))
                                 for t,p in zip(indices.cpu().tolist(), probabilities.cpu().tolist())]})
            cursor += x.size(0)
    duration = time.perf_counter() - start
    for r in rows_by_doc:
        r["cross_entropy_sum"] = math.fsum(r.pop("CE_rows"))
        r["loss"] = r["cross_entropy_sum"] / r["targets"]
        r["perplexity"] = math.exp(r["loss"])
        r["top1_accuracy"] = r["top1_correct"] / r["targets"]
        r["top5_accuracy"] = r["top5_correct"] / r["targets"]
    targets = sum(r["targets"] for r in rows_by_doc)
    ce_total = math.fsum(all_rows)
    difference = abs(math.fsum(r["cross_entropy_sum"] for r in rows_by_doc) - ce_total)
    require(cursor == 812 and targets == 103858, "Test incompleto.")
    require(difference <= 1e-7, "Desglose por documento no recompone CE global.")
    loss = ce_total / targets
    return {"windows": cursor, "targets": targets, "cross_entropy_sum": ce_total,
            "loss": loss, "mean_negative_log_likelihood": loss, "perplexity": math.exp(loss),
            "bits_per_token": loss / math.log(2),
            "top1_correct": sum(r["top1_correct"] for r in rows_by_doc),
            "top5_correct": sum(r["top5_correct"] for r in rows_by_doc),
            "top1_accuracy": sum(r["top1_correct"] for r in rows_by_doc)/targets,
            "top5_accuracy": sum(r["top5_correct"] for r in rows_by_doc)/targets,
            "per_document_metrics": rows_by_doc, "document_CE_recomposition_difference": difference,
            "examples": examples, "seconds": duration, "targets_per_second": targets/duration}


def write_report(report):
    descriptor, temporary = tempfile.mkstemp(prefix="reporte_evaluacion_3C.", suffix=".tmp", dir=REPORT.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        os.replace(temporary, REPORT)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BEST)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    args = parser.parse_args()
    require(not REPORT.exists(), "Ya existe reporte oficial 3C; consulte el resultado congelado, no lo sobrescriba.")
    before = inventory()
    checkpoint, history, verified = integrity_and_checkpoint(args.checkpoint, before)
    device = choose_device(args.device)
    tolerance = 1e-5 if device.type == "cpu" else 1e-4
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    model = reconstruct(checkpoint, device)
    weights_before = weights_hash(model)
    gate = validation_gate(model, device, tolerance)
    print("Validation Gate: "+json.dumps(gate, ensure_ascii=True), flush=True)
    if args.split == "validation":
        require(weights_hash(model) == weights_before, "Pesos cambiaron.")
        require(inventory() == before, "Archivo existente cambio.")
        return
    frozen = {"passed_validation_gate": True, "checkpoint": str(BEST.resolve()),
        "checkpoint_sha256": before["modelos/checkpoint_best.pt"],
        "model_config": checkpoint["model_config"], "training_config": checkpoint["training_config"],
        "batch_size": 16, "shuffle": False,
        "num_workers": 0, "drop_last": False, "precision": "FP32", "ignore_index": -100,
        "mode": "eval() and torch.no_grad()", "aggregation": "per-target CE, FP64 CPU sums / valid targets",
        "document_CE_tolerance": 1e-7, "repeat_test_tolerance": 0.,
        "repeat_protocol": "one official Test pass plus one identical numerical-verification pass",
        "qualitative_selection_rule": "middle window each document plus doc0 quarter/doc1 three-quarter; median valid position; <=12 input tokens and <=80 context characters",
        "timezone": "America/Bogota",
        "local_freeze_timestamp": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-5))).isoformat(),
        "no_decisions_change_after_Test": True, "evaluation_code_sha256": before["evaluar.py"],
        "critical_artifact_hashes": verified}
    require(inventory() == before, "Artefacto cambio antes de revelar Test.")
    print("Configuracion congelada antes de deserializar Test.", flush=True)
    # First and only Test deserialization, strictly AFTER the successful gate.
    test_loader = crear_dataloader("test", batch_size=16, shuffle=False, seed=42)
    structural = verify_dataset(test_loader.dataset, "test")
    tokenizer = TokenizadorEducativo().cargar()
    official = test_pass(model, test_loader, device, tokenizer, True)
    repeat = test_pass(model, test_loader, device, tokenizer, False)
    fields = ("windows", "targets", "cross_entropy_sum", "loss", "perplexity", "top1_correct",
              "top5_correct", "top1_accuracy", "top5_accuracy", "per_document_metrics")
    require(all(official[k] == repeat[k] for k in fields), "Repeticion identica no reproduce Test.")
    weights_after = weights_hash(model)
    require(weights_after == weights_before and all(p.grad is None for p in model.parameters()), "Pesos o gradientes cambiaron.")
    require(inventory() == before, "Archivo existente cambio durante evaluacion.")
    loss, val = official["loss"], gate["loss"]
    uniform = math.log(6000)
    report = {
        "version": "3C.1", "evaluation_type": "single final holdout evaluation with identical verification repeat",
        "checkpoint_path": str(BEST.resolve()), "checkpoint_sha256": before["modelos/checkpoint_best.pt"],
        "checkpoint_version": checkpoint["checkpoint_version"], "checkpoint_epoch": 19,
        "checkpoint_global_step": 14839, "checkpoint_best_validation_loss": checkpoint["best_validation_loss"],
        "verified_artifact_hashes": verified, "model_config": checkpoint["model_config"],
        "model_parameters": {"total": 1379456, "trainable": 1379456, "weight_tying": True, "all_finite": True},
        "history_audit": {"epochs": 20, "minimum_validation_epoch": 19, "minimum_validation_loss": HISTORICAL_LOSS,
                         "epoch_20_validation_loss": history[-1]["validation_loss"], "original_best_policy_reproduced": True},
        "checkpoint_audit": {"required_fields_present": True, "embedded_history_matches": True,
                           "selected_using_Validation_only": True, "last_checkpoint_deserialized": False,
                           "optimizer_state_loaded_as_file_data_only": True},
        "validation_gate": gate, "test_frozen_configuration": frozen,
        "test_documents": 3, "test_windows": official["windows"], "test_targets": official["targets"],
        "test_cross_entropy_sum": official["cross_entropy_sum"], "test_loss": loss,
        "test_mean_negative_log_likelihood": official["mean_negative_log_likelihood"],
        "test_perplexity": official["perplexity"], "test_bits_per_token": official["bits_per_token"],
        "test_top1_accuracy": official["top1_accuracy"], "test_top5_accuracy": official["top5_accuracy"],
        "test_top1_correct": official["top1_correct"], "test_top5_correct": official["top5_correct"],
        "per_document_metrics": official["per_document_metrics"], "test_structural_integrity": structural,
        "document_CE_recomposition_difference": official["document_CE_recomposition_difference"],
        "validation_test_comparison": {"validation_loss": val, "validation_perplexity": gate["perplexity"],
            "test_loss": loss, "test_perplexity": official["perplexity"], "test_minus_validation_loss": loss-val,
            "relative_loss_difference_percent": 100*(loss-val)/val,
            "relative_perplexity_difference_percent": 100*(official["perplexity"]/gate["perplexity"]-1),
            "description": "Test mejor (menor NLL)" if loss < val else ("Test peor (mayor NLL)" if loss > val else "Test igual"),
            "arbitrary_acceptance_threshold": None, "train_epoch19_loss_reference": history[18]["train_loss"],
            "train_not_directly_equivalent": "Train usa dropout y targets solapados por stride64; Validation/Test usan eval y targets unicos."},
        "uniform_baseline_comparison": {"uniform_loss": uniform, "uniform_perplexity": 6000.,

            "absolute_NLL_reduction": uniform-loss, "relative_NLL_reduction_percent": 100*(uniform-loss)/uniform,
            "not_a_comparison_against_other_trained_models": True},
        "qualitative_next_token_examples": official["examples"],
        "runtime": {"device": str(device), "pytorch_version": str(torch.__version__), "cpu_threads": 2,
            "checkpoint_pytorch_version": checkpoint["pytorch_version"], "batch_size": 16,
            "validation_gate_seconds": gate["seconds"], "test_seconds": official["seconds"],
            "test_targets_per_second": official["targets_per_second"], "verification_repeat_seconds": repeat["seconds"]},
        "reproducibility": {"official_passes": 1, "verification_passes": 1, "same_checkpoint_and_protocol": True,
            "exact_metric_identity": True, "max_loss_difference": abs(loss-repeat["loss"]), "model_weights_sha256_before": weights_before,
            "model_weights_sha256_after": weights_after, "universal_GPU_determinism_claimed": False},
        "weights_modified": False, "optimizer_created": False, "backward_calls": 0, "test_used_for_selection": False,
        "generation_performed": False, "files_created": ["evaluar.py", "resultados/reporte_evaluacion_3C.json"],
        "files_modified": [], "scope_verification": {"all_preexisting_files_unchanged": True,
            "best_checkpoint_byte_identical": True, "last_checkpoint_byte_identical": True,
            "evaluation_start_file_count": len(before), "preexisting_artifact_hashes": before},
        "warnings": ["Test metrics are descriptive, never a basis for selecting/retraining this model.",
            "Perplexity does not establish scientific correctness, factuality or generation quality.",
            "Three held-out documents provide limited scope; per-document differences are descriptive.",
            "Single byte-level BPE pieces can decode to incomplete words or replacement characters; token IDs and raw bytes are included.",
            "Test CE uses FP64 accumulation of FP32 per-target losses, mathematically equivalent to target-weighted summed CE.",
            "Only this runtime/device was checked for exact reproducibility."],
        "verdict": "EVALUACI\u00d3N FINAL V\u00c1LIDA \u2014 LISTO PARA ADAPTAR GENERACI\u00d3N",
    }
    write_report(report)
    after = inventory()
    require(all(after.get(name) == digest for name, digest in before.items()), "Archivo existente cambio.")
    require(set(after)-set(before) == {"resultados/reporte_evaluacion_3C.json"}, "Salida no autorizada.")
    print("FINAL_JSON="+json.dumps({k:report[k] for k in (
        "test_loss", "test_perplexity", "test_bits_per_token", "test_top1_accuracy", "test_top5_accuracy",
        "test_cross_entropy_sum", "test_targets", "test_windows", "per_document_metrics",
        "validation_test_comparison", "uniform_baseline_comparison", "qualitative_next_token_examples",
        "runtime", "reproducibility", "verdict")}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()


