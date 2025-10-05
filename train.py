# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data_loader import get_dataset
from model import AFFCCL

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": float(p_ma),
        "recall_macro": float(r_ma),
        "f1_macro": float(f1_ma),
        "precision_micro": float(p_mi),
        "recall_micro": float(r_mi),
        "f1_micro": float(f1_mi),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(y_true)),
    }

def grid_values(start: float, stop: float, step: float) -> List[float]:
    k = int(round((stop - start) / step))
    return [round(start + i * step, 2) for i in range(k + 1)]

def tune_gamma_eta_fast(
    mu_tr: np.ndarray, y_tr: np.ndarray, seed: int, device: str,
    gmin: float, gmax: float, gstep: float,
    emin: float, emax: float, estep: float,
) -> Tuple[float, float, Dict[str, Any]]:
    # hold-out validation inside training split
    tr_idx, val_idx = train_test_split(
        np.arange(len(y_tr)), test_size=0.2, random_state=seed, stratify=y_tr,
    )
    Xtr, Ytr = mu_tr[tr_idx], y_tr[tr_idx]
    Xval, Yval = mu_tr[val_idx], y_tr[val_idx]

    gamma_grid = grid_values(gmin, gmax, gstep)
    eta_grid   = grid_values(emin, emax, estep)

    # prepare once
    base = AFFCCL(random_state=seed, device=device)
    base.prepare(Xtr, Ytr)

    best = {"acc": -1.0, "f1": -1.0, "gamma": None, "eta": None}
    for g in gamma_grid:
        for e in eta_grid:
            base.fit_from_prepared(gamma=g, eta=e)
            yhat = base.predict(Xval)
            acc = accuracy_score(Yval, yhat)
            f1  = precision_recall_fscore_support(Yval, yhat, average="macro", zero_division=0)[2]
            better = (
                (acc > best["acc"] + 1e-12)
                or (abs(acc - best["acc"]) <= 1e-12 and f1 > best["f1"] + 1e-12)
                or (abs(acc - best["acc"]) <= 1e-12 and abs(f1 - best["f1"]) <= 1e-12 and (g < (best["gamma"] or 9)))
                or (abs(acc - best["acc"]) <= 1e-12 and abs(f1 - best["f1"]) <= 1e-12 and g == best["gamma"] and (e < (best["eta"] or 9)))
            )
            if better:
                best.update({"acc": float(acc), "f1": float(f1), "gamma": g, "eta": e})
    return float(best["gamma"]), float(best["eta"]), best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    # paper ranges
    ap.add_argument("--gamma-min", type=float, default=0.40)
    ap.add_argument("--gamma-max", type=float, default=0.95)
    ap.add_argument("--gamma-step", type=float, default=0.01)
    ap.add_argument("--eta-min", type=float, default=0.80)
    ap.add_argument("--eta-max", type=float, default=0.99)
    ap.add_argument("--eta-step", type=float, default=0.01)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--outdir", type=str, default="outputs/affccl")
    args = ap.parse_args()

    X_raw, y, feat, mu = get_dataset(args.dataset)
    outdir = Path(args.outdir) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for r in range(args.repeat):
        seed_r = args.random_state + r
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=seed_r, stratify=y,
        )
        Xtr, Ytr = mu[tr_idx], y[tr_idx]
        Xte, Yte = mu[te_idx], y[te_idx]

        # γ/η grid (fast)
        g_best, e_best, tune_info = tune_gamma_eta_fast(
            Xtr, Ytr, seed_r, device=args.device,
            gmin=args.gamma_min, gmax=args.gamma_max, gstep=args.gamma_step,
            emin=args.eta_min,   emax=args.eta_max,   estep=args.eta_step,
        )

        # fit on training split with best γ/η
        clf = AFFCCL(gamma=g_best, eta=e_best, random_state=seed_r, device=args.device)
        clf.fit(Xtr, Ytr)
        yhat = clf.predict(Xte)
        m = metrics(Yte, yhat)
        m.update({
            "run": r + 1, "seed": seed_r,
            "best_gamma": g_best, "best_eta": e_best,
            "val_acc": tune_info["acc"], "val_f1": tune_info["f1"],
        })
        runs.append(m)

        # save predictions and model
        pred_path = outdir / f"run_{r+1:02d}_test_preds.csv"
        pred_path.write_text("idx,y_true,y_pred\n", encoding="utf-8")
        with pred_path.open("a", encoding="utf-8") as f:
            for i, (yt, yp) in enumerate(zip(Yte, yhat)):
                f.write(f"{int(te_idx[i])},{int(yt)},{int(yp)}\n")
        clf.save(str(outdir / f"run_{r+1:02d}_model.pkl"))

        # ONE-LINE summary (English)
        print(
            f"RUN {r+1}/{args.repeat} | seed={seed_r} | "
            f"best_gamma={g_best:.2f} | best_eta={e_best:.2f} | "
            f"val_acc={tune_info['acc']:.4f} | val_f1={tune_info['f1']:.4f} | "
            f"test_acc={m['accuracy']:.4f} | test_f1={m['f1_macro']:.4f}"
        )

    # aggregate
    keys = ["accuracy","precision_macro","recall_macro","f1_macro","precision_micro","recall_micro","f1_micro"]
    agg = {}
    for k in keys:
        vals = np.array([ri[k] for ri in runs], dtype=float)
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std())}

    (outdir / "runs_metrics.json").write_text(
        json.dumps({"runs": runs, "aggregate": agg}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("AGGREGATE | " + " | ".join(f"{k}={agg[k]['mean']:.4f}±{agg[k]['std']:.4f}" for k in keys))

if __name__ == "__main__":
    main()
