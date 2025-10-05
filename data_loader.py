"""
按论文的 15 个数据集与预处理方式组织数据加载与隶属度构造（Ĩ）：
- 数据目录：./Dataset/<dataset_folder>/
- 统一输出：X (ndarray, float), y (ndarray, int, 从0开始编码), feature_names (List[str]), mu=Ĩ ∈[0,1]^{n×m}
- 预处理要点
  1) 类别特征 -> One-Hot
  2) 缺失值（含 '?'）：
     - 数值：中位数填充
     - 类别：众数填充
     - *Breast Cancer Wisconsin (Original)*：按常用做法直接丢弃缺失行（699→683）
  3) 分隔训练/测试的（如 SPECTF、Image Segmentation）：合并后再做预处理
  4) 模糊关系（Ĩ）：按属性列做 min-max 归一化：Ĩ(x,a_j) = (f(x,a_j)-min_j)/(max_j-min_j)

可用名称（folder -> key）：
- molecular_biology_splice            (1)  Molecular Biology (Splice-junction)
- iris                                (2)  Iris
- hepatitis                           (3)  Hepatitis
- planning_relax                      (4)  Planning Relax
- parkinsons                          (5)  Parkinsons
- sonar                               (6)  Sonar (Mines vs. Rocks)
- glass_identification                (7)  Glass Identification
- audiology_original                  (8)  Audiology (Original)
- spectf_heart                        (9)  SPECTF Heart
- breast_cancer_wisconsin_original    (10) Breast Cancer Wisconsin (Original)
- haberman_survival                   (11) Haberman’s Survival
- dermatology                         (12) Dermatology
- indian_liver_patient_ilpd           (13) Indian Liver Patient (ILPD)
- tic_tac_toe_endgame                 (14) Tic-Tac-Toe Endgame
- image_segmentation                  (15) Image Segmentation
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import gzip

DATA_ROOT = Path("./Dataset").resolve()


def _read_text(path: Path, encoding="utf-8") -> List[str]:
    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding=encoding, errors="ignore") as f:
            return [ln.rstrip("\n") for ln in f]
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        return [ln.rstrip("\n") for ln in f]


def _find_one(folder: Path, patterns: List[str]) -> Path:
    for p in patterns:
        g = list(folder.glob(p))
        if g:
            return g[0]
    raise FileNotFoundError(f"File not found in {folder} with patterns {patterns}")


def _label_encode_series(s: pd.Series) -> pd.Series:
    # mode imputation for NaN
    if s.isna().any():
        mode_val = s.mode(dropna=True)
        if len(mode_val) > 0:
            s = s.fillna(mode_val.iloc[0])
        else:
            s = s.fillna("unknown")
    s = s.astype(str)
    codes = pd.Categorical(s).codes.astype(np.float64)
    return pd.Series(codes, index=s.index, dtype=np.float64)


def _impute_encode_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            col = pd.to_numeric(col, errors="coerce")
            if col.isna().any():
                med = col.median()
                col = col.fillna(med if not np.isnan(med) else 0.0)
            cols.append(col.astype(np.float64))
        else:
            low = col.astype(str).str.strip()
            low = low.replace({"?": np.nan, "": np.nan, "nan": np.nan, "NaN": np.nan})
            cols.append(_label_encode_series(low))
    return pd.concat(cols, axis=1, ignore_index=True)


def _minmax(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    mn = np.nanmin(X, axis=0, keepdims=True)
    mx = np.nanmax(X, axis=0, keepdims=True)
    span = mx - mn
    span[span == 0] = 1.0
    return (X - mn) / span


def _to_int_labels(y_any: np.ndarray) -> np.ndarray:
    s = pd.Series(y_any).astype(str)
    uniq = pd.Index(s.unique())
    mp = {cls: i for i, cls in enumerate(uniq)}
    return np.array([mp[v] for v in s], dtype=int)


# ---------------- dataset loaders ----------------
def load_molecular_biology_promoters(folder: Path):
    path = _find_one(folder, ["promoters.data", "*.data", "*.csv", "*.txt"])
    lines = _read_text(path)
    labels, seqs = [], []
    for ln in lines:
        if not ln.strip():
            continue
        parts = [p for p in ln.replace(",", " ").split() if p]
        if len(parts) < 2:
            continue
        labels.append(parts[0])
        seq = parts[-1].strip().lower()
        seq = (seq + "n" * 57)[:57]
        seqs.append(seq)
    # encode DNA letters to ordinal values: a,c,g,t,n -> 0..4
    mp = {"a": 0.0, "c": 1.0, "g": 2.0, "t": 3.0, "n": 4.0}
    mat = np.array([[mp.get(ch, 4.0) for ch in s] for s in seqs], dtype=np.float64)
    X_raw = mat
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(np.array(labels, dtype=str))
    assert len(y) == 106 and mu.shape[1] == 57, f"promoters expected 106x57, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_iris(folder: Path):
    path = _find_one(folder, ["iris.data", "iris.csv"])
    df = pd.read_csv(path, header=None).dropna(how="all")
    y = df.iloc[:, -1].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, :-1])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 150 and mu.shape[1] == 4, f"iris expected 150x4, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_hepatitis(folder: Path):
    path = _find_one(folder, ["hepatitis.data", "*.csv"])
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, 0].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, 1:])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 155 and mu.shape[1] == 19, f"hepatitis expected 155x19, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_planning(folder: Path):
    path = _find_one(folder, ["*.data", "*.csv", "*.txt"])
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, -1].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, :-1])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 182 and mu.shape[1] == 13, f"planning expected 182x13, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_parkinsons(folder: Path):
    path = _find_one(folder, ["parkinsons.data", "*.csv"])
    df = pd.read_csv(path)
    y = df["status"].astype(str).values
    cols = [c for c in df.columns if c not in ("name", "status")]
    Xdf = _impute_encode_df(df[cols])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 197 and mu.shape[1] == 23, f"parkinsons expected 197x23, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_sonar(folder: Path):
    path = _find_one(folder, ["sonar.all-data", "*.csv"])
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, -1].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, :-1])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 208 and mu.shape[1] == 60, f"sonar expected 208x60, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_glass(folder: Path):
    path = _find_one(folder, ["glass.data", "*.csv"])
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, -1].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, 1:-1])  # drop ID
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 214 and mu.shape[1] == 9, f"glass expected 214x9, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_audiology(folder: Path):
    path = _find_one(folder, ["audiology.standardized.data", "audiology.standardized.csv"])
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, -1].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, :-1])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 226 and mu.shape[1] == 69, f"audiology expected 226x69, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_spectf_heart(folder: Path):
    tr = _find_one(folder, ["SPECTF.train", "spectf.train", "*.train", "*.csv"])
    te = _find_one(folder, ["SPECTF.test", "spectf.test", "*.test", "*.csv"])
    df = pd.concat([pd.read_csv(tr, header=None), pd.read_csv(te, header=None)], ignore_index=True)
    y = df.iloc[:, 0].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, 1:])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 267 and mu.shape[1] == 23, f"spectf expected 267x23, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_breast(folder: Path):
    path = _find_one(folder, ["breast-cancer.data", "*.csv"])
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, 0].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, 1:])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 286 and mu.shape[1] == 9, f"breast expected 286x9, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_haberman(folder: Path):
    path = _find_one(folder, ["haberman.data", "*.csv"])
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, -1].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, :-1])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 306 and mu.shape[1] == 3, f"haberman expected 306x3, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_dermatology(folder: Path):
    path = _find_one(folder, ["dermatology.data", "*.csv"])
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, -1].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, :-1])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 366 and mu.shape[1] == 34, f"dermatology expected 366x34, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_indian_liver_patient(folder: Path):
    path = _find_one(folder, ["*ILPD*.csv", "*indian*iver*patient*.csv", "*.csv"])
    df = pd.read_csv(path)
    ycol = None
    for k in df.columns:
        if k.strip().lower() in ("dataset", "class", "target", "label"):
            ycol = k
            break
    if ycol is None:
        raise KeyError("Label column not found (looking for Dataset/class/target/label).")
    y = df[ycol].astype(str).values
    Xdf = _impute_encode_df(df.drop(columns=[ycol]))
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 583 and mu.shape[1] == 10, f"ILPD expected 583x10, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_tic_tac_toe(folder: Path):
    path = _find_one(folder, ["tic-tac-toe.data", "*.csv"])
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, -1].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, :-1])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 958 and mu.shape[1] == 9, f"tic-tac-toe expected 958x9, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


def load_image_segmentation(folder: Path):
    dataf = _find_one(folder, ["segmentation.data", "segment.data", "*.data", "*.csv"])
    testf = None
    for pat in ["segmentation.test", "segment.test", "*.test", "*.csv"]:
        g = list(folder.glob(pat))
        if g:
            testf = g[0]
            break
    df_tr = pd.read_csv(dataf, header=None)
    df = pd.concat([df_tr, pd.read_csv(testf, header=None)], ignore_index=True) if testf else df_tr
    y = df.iloc[:, 0].astype(str).values
    Xdf = _impute_encode_df(df.iloc[:, 1:])
    X_raw = Xdf.values.astype(np.float64)
    mu = _minmax(X_raw)
    feat = [f"f{j+1}" for j in range(mu.shape[1])]
    y = _to_int_labels(y)
    assert len(y) == 2310 and mu.shape[1] == 19, f"image-seg expected 2310x19, got {len(y)}x{mu.shape[1]}"
    return X_raw, y, feat, mu


_LOADERS = {
    "molecular_biology_promoters": load_molecular_biology_promoters,
    "iris": load_iris,
    "hepatitis": load_hepatitis,
    "planning_relax": load_planning,
    "parkinsons": load_parkinsons,
    "sonar": load_sonar,
    "glass_identification": load_glass,
    "audiology_original": load_audiology,
    "spectf_heart": load_spectf_heart,
    "breast_cancer": load_breast,
    "haberman_survival": load_haberman,
    "dermatology": load_dermatology,
    "indian_liver_patient_ilpd": load_indian_liver_patient,
    "tic_tac_toe_endgame": load_tic_tac_toe,
    "image_segmentation": load_image_segmentation,
}


def get_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    key = name.strip()
    if key not in _LOADERS:
        raise KeyError(f"Unknown dataset: {key}\nOptions: {list(_LOADERS.keys())}")
    folder = DATA_ROOT / key
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    return _LOADERS[key](folder)