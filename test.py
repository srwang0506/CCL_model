# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from data_loader import get_dataset
from model import AFFCCL

def metrics(y_true: np.ndarray, y_pred: np.ndarray):
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
    }

def grid_values(start: float, stop: float, step: float):
    k = int(round((stop - start) / step))
    return [round(start + i * step, 2) for i in range(k + 1)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--model", type=str, default=None, help="path to saved model.pkl; if absent, fit here")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    # optional grid on full data (80/20 split just for tuning here)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--gamma-min", type=float, default=0.40)
    ap.add_argument("--gamma-max", type=float, default=0.95)
    ap.add_argument("--gamma-step", type=float, default=0.01)
    ap.add_argument("--eta-min", type=float, default=0.80)
    ap.add_argument("--eta-max", type=float, default=0.99)
    ap.add_argument("--eta-step", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.60)
    ap.add_argument("--eta", type=float, default=0.85)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    X_raw, y, feat, mu = get_dataset(args.dataset)

    if args.model and Path(args.model).exists():
        clf = AFFCCL.load(args.model)
        yhat = clf.predict(mu)
        m = metrics(y, yhat)
        print(f"EVAL | loaded_model | test_acc={m['accuracy']:.4f} | test_f1={m['f1_macro']:.4f}")
    else:
        if args.grid:
            tr_idx, te_idx = train_test_split(
                np.arange(len(y)), test_size=0.2, random_state=args.random_state, stratify=y,
            )
            Xtr, Ytr = mu[tr_idx], y[tr_idx]
            Xte, Yte = mu[te_idx], y[te_idx]

            clf = AFFCCL(random_state=args.random_state, device=args.device)
            clf.prepare(Xtr, Ytr)
            best = {"acc": -1.0, "f1": -1.0, "g": None, "e": None}
            for g in grid_values(args.gamma_min, args.gamma_max, args.gamma_step):
                for e in grid_values(args.eta_min, args.eta_max, args.eta_step):
                    clf.fit_from_prepared(gamma=g, eta=e)
                    yhat_val = clf.predict(Xte)  # here using held-out as validation just for quick grid in test
                    acc = accuracy_score(Yte, yhat_val)
                    f1 = precision_recall_fscore_support(Yte, yhat_val, average="macro", zero_division=0)[2]
                    if (acc > best["acc"] + 1e-12) or (abs(acc - best["acc"]) <= 1e-12 and f1 > best["f1"] + 1e-12):
                        best.update({"acc": float(acc), "f1": float(f1), "g": g, "e": e})
            print(f"GRID | best_gamma={best['g']:.2f} | best_eta={best['e']:.2f} | val_acc={best['acc']:.4f} | val_f1={best['f1']:.4f}")
            # final fit on full data with best thresholds
            clf = AFFCCL(gamma=best["g"], eta=best["e"], random_state=args.random_state, device=args.device)
            clf.fit(mu, y)
            yhat = clf.predict(mu)
            m = metrics(y, yhat)
            print(f"EVAL | full_fit | gamma={best['g']:.2f} | eta={best['e']:.2f} | acc={m['accuracy']:.4f} | f1_macro={m['f1_macro']:.4f}")
        else:
            clf = AFFCCL(gamma=args.gamma, eta=args.eta, random_state=args.random_state, device=args.device)
            clf.fit(mu, y)
            yhat = clf.predict(mu)
            m = metrics(y, yhat)
            print(f"EVAL | gamma={args.gamma:.2f} | eta={args.eta:.2f} | acc={m['accuracy']:.4f} | f1_macro={m['f1_macro']:.4f}")

    # save metrics if requested
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
