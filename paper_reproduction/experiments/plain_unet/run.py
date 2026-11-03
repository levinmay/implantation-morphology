"""Train and evaluate the conventional U-Net on the five paper folds."""
import os
os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
import sys
sys.dont_write_bytecode = True
import fcntl
import json
import math
from pathlib import Path
import random
import time
import traceback
import numpy as np
import pandas as pd
from PIL import Image
import torch
from .cohort import digest, save_json, save_torch, status, prepare, partition, dice
from .unet import PlainUNet
from .model import preprocess, loss_function, native_prediction, augment

OUTPUT = None
EPOCHS = 250
BATCH = 4


def new_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return PlainUNet().cuda()


def load_cohort(frame):
    samples = {}
    for record in frame.to_dict("records"):
        with Image.open(record["image"]) as image, Image.open(record["mask"]) as mask:
            samples[record["unit_id"]] = preprocess(image, mask)
    return samples


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)


def learning_rates(optimizer, step, total, warmup):
    factor = (step + 1) / warmup if step < warmup else max(0, 1 - (step - warmup) / (total - warmup)) ** 0.9
    for group in optimizer.param_groups:
        group["lr"] = 1e-3 * factor


def batch_samples(samples, records, indices, generator=None):
    values = []
    for index in indices:
        sample = samples[records[int(index)]["unit_id"]]
        values.append(augment(*sample[:3], generator) if generator is not None else sample[:3])
    return [torch.stack([value[k] for value in values]).cuda() for k in range(3)]


def evaluate(model, frame, samples, folder=None):
    model.eval()
    if folder is not None:
        folder.mkdir(exist_ok=True)
    rows = []
    with torch.no_grad():
        records = frame.to_dict("records")
        for start in range(0, len(records), BATCH):
            subset = records[start:start + BATCH]
            images = torch.stack([samples[r["unit_id"]][0] for r in subset]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            for j, record in enumerate(subset):
                sample = samples[record["unit_id"]]
                prediction = native_prediction(logits[j:j + 1], sample[3], sample[4])
                reference = np.asarray(Image.open(record["mask"]).convert("L")) > 0
                if prediction.shape != reference.shape:
                    raise ValueError("Prediction grid differs")
                row = {k: record[k] for k in ("unit_id", "component_id", "evaluation_fold", "four_group")}
                row.update(method="plain_unet", dice=dice(prediction, reference))
                rows.append(row)
                if folder is not None:
                    Image.fromarray(prediction.astype(np.uint8) * 255).save(folder / (record["unit_id"] + ".png"))
    return pd.DataFrame(rows)


def smoke(frame, samples):
    model = new_model(42)
    records = partition(frame, 0)["train"].iloc[:2].to_dict("records")
    x, y, valid = batch_samples(samples, records, range(len(records)))
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before_logits = model(x).detach().clone()
        before = float(loss_function(before_logits, y, valid))
    save_torch(OUTPUT / "private/smoke_initial.pt", model.state_dict())
    optimizer = optimizer_for(model)
    for group in optimizer.param_groups:
        group["lr"] = 1e-4
    # Fixed train-only examples test real backpropagation without augmentation/dropout noise.
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_function(logits, y, valid)
        loss.backward()
        for module in (model.encoders, model.decoders, model.output):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):
                raise ValueError("Invalid encoder/decoder smoke gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        after = float(loss_function(model(x), y, valid))
    if not after < before:
        raise ValueError("Smoke fitting did not reduce train-only loss")
    model.load_state_dict(torch.load(OUTPUT / "private/smoke_initial.pt", map_location="cpu", weights_only=True))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        restored = model(x)
    if not torch.equal(restored, before_logits):
        raise ValueError("Checkpoint restoration changed inference")
    output = evaluate(model, partition(frame, 0)["train"].iloc[:2], samples)
    save_json(OUTPUT / "aggregate/SMOKE.json", {"status": "PASS", "training_loss_before": before,
        "training_loss_after": after, "native_grid_checked": len(output), "encoder_decoder_gradients": True,
        "checkpoint_round_trip": True, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3})
    del optimizer, model
    torch.cuda.empty_cache()


def qualitative(frame, folder):
    destination = folder / "qualitative"
    destination.mkdir(exist_ok=True)
    # First case in each group is chosen independently of prediction quality.
    for group, subset in frame.groupby("four_group"):
        record = subset.sort_values("unit_id").iloc[0]
        panels = []
        for path in (record.image, record["mask"], str(folder / "masks" / (record.unit_id + ".png"))):
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((384, 384))
                canvas = Image.new("RGB", (384, 384))
                canvas.paste(im, (0, 0))
                panels.append(canvas)
        canvas = Image.new("RGB", (1152, 384))
        for index, panel in enumerate(panels):
            canvas.paste(panel, (384 * index, 0))
        canvas.save(destination / (group + "_image_reference_prediction.png"))


def train_fold(frame, samples, fold):
    folder = OUTPUT / "private" / f"fold_{fold}"
    folder.mkdir(exist_ok=True)
    if (folder / "COMPLETE.json").exists():
        checks = json.loads((folder / "COMPLETE.json").read_text())
        if any(digest(folder / name) != value for name, value in checks.items()):
            raise ValueError("Completed fold changed")
        return
    model = new_model(42 + fold)
    optimizer = optimizer_for(model)
    subsets = partition(frame, fold)
    records = subsets["train"].to_dict("records")
    batches = math.ceil(len(records) / BATCH)
    first, best, history = 1, -1, []
    if (folder / "latest.pt").exists():
        state = torch.load(folder / "latest.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first, best, history = state["epoch"] + 1, state["best"], state["history"]
    for epoch in range(first, EPOCHS + 1):
        started = time.monotonic()
        model.train()
        torch.manual_seed(42 + fold * 1000 + epoch)
        torch.cuda.manual_seed_all(42 + fold * 1000 + epoch)
        generator = np.random.default_rng(42 + fold * 1000 + epoch)
        order = generator.permutation(len(records))
        losses = []
        for number, start in enumerate(range(0, len(order), BATCH)):
            learning_rates(optimizer, (epoch - 1) * batches + number, EPOCHS * batches, 5 * batches)
            x, y, valid = batch_samples(samples, records, order[start:start + BATCH], generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_function(model(x), y, valid)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.monotonic() - started}
        if epoch == 1 or epoch % 5 == 0:
            row["validation_dice"] = float(evaluate(model, subsets["val"], samples).dice.mean())
            if row["validation_dice"] > best:
                best = row["validation_dice"]
                save_torch(folder / "best_model.pt", model.state_dict())
                save_json(folder / "BEST.json", {"epoch": epoch, "validation_dice": best})
        history.append(row)
        pd.DataFrame(history).to_csv(folder / "training.csv", index=False)
        save_torch(folder / "latest.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "epoch": epoch, "best": best, "history": history})
        status(OUTPUT, "training", outer_fold=fold, total_epochs=EPOCHS, **row)
    model.load_state_dict(torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True))
    status(OUTPUT, "evaluating_outer_test", outer_fold=fold)
    metrics = evaluate(model, subsets["test"], samples, folder / "masks")
    metrics.to_csv(folder / "metrics.csv", index=False)
    qualitative(subsets["test"], folder)
    paths = [folder / name for name in ("best_model.pt", "BEST.json", "training.csv", "metrics.csv")]
    paths += list((folder / "masks").glob("*.png")) + list((folder / "qualitative").glob("*.png"))
    save_json(folder / "COMPLETE.json", {str(p.relative_to(folder)): digest(p) for p in paths})
    del model, optimizer
    torch.cuda.empty_cache()


def aggregate(frame):
    metrics = pd.concat([pd.read_csv(OUTPUT / f"private/fold_{fold}/metrics.csv") for fold in range(5)], ignore_index=True)
    if len(metrics) != 577 or metrics.unit_id.nunique() != 577 or set(metrics.unit_id) != set(frame.unit_id):
        raise ValueError("Incomplete out-of-fold coverage")
    baseline = []
    for record in frame.to_dict("records"):
        path = Path(record["nnunet_prediction"])
        prediction = np.asarray(Image.open(path).convert("L")) > 0
        reference = np.asarray(Image.open(record["mask"]).convert("L")) > 0
        if prediction.shape != reference.shape:
            raise ValueError("nnU-Net native grid differs")
        row = {k: record[k] for k in ("unit_id", "component_id", "evaluation_fold", "four_group")}
        row.update(method="nnunet_original_raw", dice=dice(prediction, reference))
        baseline.append(row)
    metrics = pd.concat([metrics, pd.DataFrame(baseline)], ignore_index=True)
    if not np.isfinite(metrics.dice).all() or not metrics.dice.between(0, 1).all():
        raise ValueError("Invalid Dice values")
    metrics.to_csv(OUTPUT / "private/METRICS.csv", index=False)
    summary = []
    for method, subset in metrics.groupby("method"):
        for group in ["overall", *sorted(subset.four_group.unique())]:
            selected = subset if group == "overall" else subset.loc[subset.four_group == group]
            summary.append({"method": method, "group": group, "observations": len(selected), "mean_dice": float(selected.dice.mean()), "median_dice": float(selected.dice.median())})
    pd.DataFrame(summary).to_csv(OUTPUT / "aggregate/RESULTS.csv", index=False)
    paired = metrics.pivot(index=["unit_id", "component_id"], columns="method", values="dice").reset_index()
    if paired.isna().any().any():
        raise ValueError("Unpaired predictions")
    codes, groups = pd.factorize(paired.component_id)
    differences = (paired.plain_unet - paired.nnunet_original_raw).to_numpy()
    generator = np.random.default_rng(20261007)
    draws = [np.average(differences, weights=generator.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))[codes]) for _ in range(5000)]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    pd.DataFrame([{"comparison": "plain_unet_minus_nnunet", "dice_difference": differences.mean(), "ci_low": lo, "ci_high": hi}]).to_csv(OUTPUT / "aggregate/PAIRED_COMPARISONS.csv", index=False)
    save_json(OUTPUT / "aggregate/VALIDATION.json", {"status": "PASS", "units": 577, "outer_folds": 5,
        "groups": len(groups), "test_used_for_selection": False, "threshold": 0.5, "external_validation": False})


def main():
    import argparse
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "release-dir", "split-dir", "nnunet-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("Conventional U-Net training requires a CUDA GPU with bfloat16 support")
    OUTPUT = args.output_dir.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name in ("private", "aggregate", "tmp"):
        (OUTPUT / name).mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(OUTPUT / "tmp")
    lock = (OUTPUT / "RUN.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT / "SUCCESS").exists():
        return
    try:
        config = {"architecture": "Plain U-Net", "pretraining": "none; Kaiming initialization",
            "channels": [64, 128, 256, 512, 1024], "convolutions": "two padded 3x3 convolutions and ReLU per level",
            "downsampling": "2x2 max pooling", "upsampling": "2x2 transposed convolutions with concatenated skips",
            "normalization_layers": "none", "epochs": EPOCHS, "batch_size": BATCH, "seed_by_fold": [42, 43, 44, 45, 46],
            "input": "RGB, ImageNet normalization, longest side 512, zero padding in normalized space",
            "loss": "valid-region BCE + per-image soft Dice", "augmentation": "random 90-degree rotations and horizontal/vertical flips",
            "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 0.01,
            "schedule": "5-epoch linear warmup then polynomial decay power 0.9", "mixed_precision": "bfloat16",
            "selection": "maximum validation native-grid mean Dice at epoch 1 and every 5 epochs",
            "prediction": "native-grid interpolated logits > 0; no mask cleanup; no test-time augmentation",
            "sources": {name: digest(Path(__file__).parent / name) for name in ("run.py", "model.py", "cohort.py", "test_model.py", "unet.py", "test_unet.py")}}
        path = OUTPUT / "CONFIG.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Config/source changed; use a new run version")
        save_json(path, config)
        status(OUTPUT, "preflight")
        frame, hashes = prepare(args)
        for record in frame.to_dict("records"):
            p = Path(record["nnunet_prediction"])
            hashes[str(p)] = digest(p)
        save_json(OUTPUT / "private/ALL_INPUT_HASHES.json", hashes)
        torch.set_num_threads(2)
        status(OUTPUT, "preprocessing")
        samples = load_cohort(frame)
        status(OUTPUT, "smoke_test")
        smoke(frame, samples)
        for fold in range(5):
            train_fold(frame, samples, fold)
        status(OUTPUT, "aggregating")
        aggregate(frame)
        if any(digest(path) != expected for path, expected in hashes.items()):
            raise ValueError("Original inputs changed")
        (OUTPUT / "SUCCESS").write_text("Validated five-fold completion\n")
        status(OUTPUT, "complete")
    except BaseException as error:
        (OUTPUT / "FAILED").write_text(traceback.format_exc())
        status(OUTPUT, "failed", error_type=type(error).__name__, message=str(error))
        raise


if __name__ == "__main__":
    main()
