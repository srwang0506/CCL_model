# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

_EPS = 1e-12


# ---------- Galois operators ----------
def L_operator(mu: torch.Tensor, X_idx: torch.Tensor) -> torch.Tensor:
    m = mu.shape[1]
    if X_idx.numel() == 0:
        return torch.ones(m, dtype=mu.dtype, device=mu.device)
    return mu.index_select(0, X_idx).min(dim=0).values


def H_operator(mu: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    mask = (mu >= (B.unsqueeze(0) - _EPS)).all(dim=1)
    return torch.nonzero(mask, as_tuple=False).flatten()


# ---------- statistics: delta, fluct, weights, CCD ----------
def delta_ck(mu: torch.Tensor, y_np: np.ndarray, cls: int) -> torch.Tensor:
    idx = np.where(y_np == cls)[0]
    if idx.size == 0:
        return torch.zeros(mu.shape[1], dtype=mu.dtype, device=mu.device)
    idx_t = torch.from_numpy(idx).to(mu.device)
    return mu.index_select(0, idx_t).mean(dim=0)


def fluct_ck(mu: torch.Tensor, y_np: np.ndarray, cls: int, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
    idx = np.where(y_np == cls)[0]
    if idx.size == 0:
        return torch.zeros(mu.shape[1], dtype=mu.dtype, device=mu.device)
    idx_t = torch.from_numpy(idx).to(mu.device)
    block = mu.index_select(0, idx_t)
    if delta is None:
        delta = block.mean(dim=0)
    return (block - delta.unsqueeze(0)).abs().sum(dim=0)


def weights_w(mu: torch.Tensor, y_np: np.ndarray, cls: int) -> torch.Tensor:
    d = delta_ck(mu, y_np, cls)
    Fl = fluct_ck(mu, y_np, cls, d)
    m = mu.shape[1]
    if float(Fl.sum().item()) == 0.0:
        return torch.zeros(m, dtype=mu.dtype, device=mu.device)
    zero = (Fl <= _EPS)
    if torch.any(zero):
        w = torch.zeros(m, dtype=mu.dtype, device=mu.device)
        w[zero] = 1.0 / float(zero.sum().item())
        return w
    inv = (m - 1) / Fl  # (m-1)/Fl normalized later
    return inv / inv.sum()


def CCD_value(B: torch.Tensor, delta_k: torch.Tensor, w_k: torch.Tensor) -> float:
    return float(1.0 - torch.sum((delta_k - B).abs() * w_k).item())


# ---------- Concept ----------
@dataclass
class Concept:
    cls: int
    extent: np.ndarray            # numpy int array (CPU)
    intent: torch.Tensor          # torch double tensor (device)
    ccd: float
    size: int


class AFFCCL:
    def __init__(
        self,
        gamma: float = 0.60,
        eta: float = 0.85,
        random_state: int = 42,
        gaussian_ccd_mu: float = 0.0,
        gaussian_ccd_sigma: float = 0.0,
        device: str = "auto",
    ):
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.random_state = int(random_state)
        self.g_mu = float(gaussian_ccd_mu)
        self.g_sigma = float(gaussian_ccd_sigma)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # learned / prepared state
        self.classes_: np.ndarray = np.array([], dtype=int)
        self.m_: int = 0
        self.delta_: Dict[int, torch.Tensor] = {}
        self.w_: Dict[int, torch.Tensor] = {}
        self.Q_gamma_: Dict[int, List[Concept]] = {}
        self.QF_: Dict[int, List[Concept]] = {}

        # cached for fast grid
        self._mu_prepared: Optional[torch.Tensor] = None
        self._y_prepared: Optional[np.ndarray] = None
        self._ext_attr_cached: Optional[List[np.ndarray]] = None
        self._ext_obj_cached: Optional[Dict[int, List[np.ndarray]]] = None

        self._rng = np.random.RandomState(self.random_state)

    # --------- helpers ----------
    def _ccd_noise(self, base: float) -> float:
        if self.g_sigma <= 0.0:
            return base
        return float(base + self._rng.normal(self.g_mu, self.g_sigma))

    @staticmethod
    def _extent_key(X: np.ndarray) -> Tuple[int, ...]:
        return tuple(np.unique(np.sort(X)).tolist())

    # --------- extents enumeration ----------
    def _object_concepts_extents(self, mu: torch.Tensor, y_np: np.ndarray, cls: int) -> List[np.ndarray]:
        idx = np.where(y_np == cls)[0]
        store: Dict[Tuple[int, ...], None] = {}
        for i in idx:
            B = L_operator(mu, torch.tensor([i], dtype=torch.long, device=mu.device))
            X = H_operator(mu, B).detach().to("cpu").numpy().astype(int)
            if X.size == 0:
                continue
            store[self._extent_key(X)] = None
        return [np.fromiter(k, dtype=int) for k in store.keys()]

    def _attribute_concepts_extents(self, mu: torch.Tensor) -> List[np.ndarray]:
        """
        Strictly follow Def. 2.7/2.8:
        For each attribute a_j and for each object x (i.e., each row),
        let v = I(x, a_j); define (x,a)^* = { y | I(y, a_j) > v }.
        Then the attribute-induced fuzzy concepts are ((x,a)^*, L((x,a)^*)),
        and we union all j, x; finally drop empty extent and (U, ∅).
        """
        n, m = mu.shape
        store: Dict[Tuple[int, ...], None] = {}
        for j in range(m):
            col = mu[:, j]                    # shape [n]
            # 逐样本阈值 v = I(x,a_j)
            for i in range(n):
                v = col[i]
                mask = col > v                # (x,a)^* = { y | I(y,a) > I(x,a) }
                if not torch.any(mask):
                    continue
                X = torch.nonzero(mask, as_tuple=False).flatten()
                if X.numel() == n:
                    continue
                X_np = X.detach().to("cpu").numpy().astype(int)
                if X_np.size == 0:
                    continue
                store[self._extent_key(X_np)] = None
        return [np.fromiter(k, dtype=int) for k in store.keys()]

    # --------- build Q^γ for a class ----------
    def _build_Q_gamma_for_class(
        self, cls: int, mu: torch.Tensor,
        ext_obj: List[np.ndarray], ext_attr: List[np.ndarray]
    ) -> List[Concept]:
        merged: Dict[Tuple[int, ...], None] = {}
        for X in ext_obj:
            merged[self._extent_key(X)] = None
        for X in ext_attr:
            merged[self._extent_key(X)] = None

        d_k = self.delta_[cls]
        w_k = self.w_[cls]
        out: List[Concept] = []
        for key in merged.keys():
            X_np = np.fromiter(key, dtype=int)
            B = L_operator(mu, torch.from_numpy(X_np).to(mu.device))
            ccd = self._ccd_noise(CCD_value(B, d_k, w_k))
            if ccd >= self.gamma:
                out.append(Concept(cls=cls, extent=X_np, intent=B.detach().clone(), ccd=float(ccd), size=int(X_np.size)))
        return out

    # --------- θ & pseudo-concepts ----------
    @staticmethod
    def _theta(c1: Concept, c2: Concept) -> float:
        X1, X2 = c1.extent, c2.extent
        n1, n2 = max(1, X1.size), max(1, X2.size)
        inter = np.intersect1d(X1, X2).size
        if inter == 0:
            return 0.0
        s1m2 = np.setdiff1d(X1, X2).size
        s2m1 = np.setdiff1d(X2, X1).size
        d1, d2 = c1.ccd, c2.ccd
        p3 = (s1m2 / n1) * d1
        p4 = (s2m1 / n2) * d2
        p5 = inter * (d1 / n1 + d2 / n2)
        num = p5 * inter
        den = p5 * inter + p3 * s1m2 + p4 * s2m1
        return 0.0 if den <= _EPS else float(num / den)

    @staticmethod
    def _pseudo_from_cluster(cluster: List[Concept], device: torch.device, dtype: torch.dtype) -> Tuple[np.ndarray, torch.Tensor]:
        cluster = sorted(cluster, key=lambda c: c.ccd, reverse=True)
        Xp = cluster[0].extent
        for c in cluster[1:]:
            Xp = np.union1d(Xp, c.extent)
        p = len(cluster)
        if p == 1:
            Bp = cluster[0].intent.to(device=device, dtype=dtype)
            return Xp.astype(int), Bp
        m = cluster[0].intent.shape[0]
        Bp = torch.zeros(m, dtype=dtype, device=device)
        coeff12 = math.ldexp(1.0, -(p - 1))  # 2^{-(p-1)}
        Bp = Bp + coeff12 * cluster[0].intent.to(device=device, dtype=dtype)
        Bp = Bp + coeff12 * cluster[1].intent.to(device=device, dtype=dtype)
        for i in range(3, p + 1):
            coeff_i = math.ldexp(1.0, -(p - i + 1))
            Bp = Bp + coeff_i * cluster[i - 1].intent.to(device=device, dtype=dtype)
        return Xp.astype(int), Bp

    def _cluster_Q_gamma(self, cls: int, Qg: List[Concept]) -> List[Concept]:
        pool = sorted(Qg, key=lambda c: c.ccd, reverse=True)
        pseudo: List[Concept] = []
        while pool:
            core = pool[0]
            cluster = [core]
            rest: List[Concept] = []
            for c in pool[1:]:
                if np.intersect1d(core.extent, c.extent).size == 0:
                    rest.append(c)
                    continue
                if self._theta(core, c) >= self.eta:
                    cluster.append(c)
                else:
                    rest.append(c)
            Xp, Bp = self._pseudo_from_cluster(cluster, device=self.device, dtype=torch.float64)
            ccd_p = CCD_value(Bp, self.delta_[cls], self.w_[cls])
            pseudo.append(Concept(cls=cls, extent=Xp, intent=Bp.detach().clone(), ccd=float(ccd_p), size=int(Xp.size)))
            pool = rest
        return pseudo

    # --------- public: standard fit / predict ----------
    def fit(self, mu_np: np.ndarray, y_np: np.ndarray) -> "AFFCCL":
        mu = torch.as_tensor(mu_np, dtype=torch.float64, device=self.device)
        self.classes_ = np.unique(y_np).astype(int)
        self.m_ = mu.shape[1]

        # per-class stats
        self.delta_.clear()
        self.w_.clear()
        for cls in self.classes_:
            self.delta_[int(cls)] = delta_ck(mu, y_np, int(cls))
            self.w_[int(cls)] = weights_w(mu, y_np, int(cls))

        # enumerate extents
        ext_attr = self._attribute_concepts_extents(mu)
        self.Q_gamma_.clear()
        for cls in self.classes_:
            ext_obj = self._object_concepts_extents(mu, y_np, int(cls))
            self.Q_gamma_[int(cls)] = self._build_Q_gamma_for_class(int(cls), mu, ext_obj, ext_attr)

        # clustering
        self.QF_.clear()
        for cls in self.classes_:
            self.QF_[int(cls)] = self._cluster_Q_gamma(int(cls), self.Q_gamma_[int(cls)])
        return self

    def predict(self, mu_np: np.ndarray) -> np.ndarray:
        mu = torch.as_tensor(mu_np, dtype=torch.float64, device=self.device)
        n, m = mu.shape
        yhat = np.empty(n, dtype=int)
        intents_per_class: Dict[int, List[torch.Tensor]] = {
            int(cls): [c.intent.to(device=self.device, dtype=torch.float64) for c in self.QF_.get(int(cls), [])]
            for cls in self.classes_
        }
        for i in range(n):
            Bt = mu[i, :]
            best_upper = (-1e18, None)
            best_lower = (-1e18, None)
            for cls in self.classes_:
                intents = intents_per_class[int(cls)]
                uppers = [B for B in intents if torch.all(B <= Bt + _EPS)]
                Uvec = (torch.stack(uppers).max(dim=0).values if uppers else torch.zeros(m, dtype=torch.float64, device=self.device))
                SAh = float(1.0 - torch.norm(Bt - Uvec, p=2).item())

                lowers = [B for B in intents if torch.all(Bt <= B + _EPS)]
                Lvec = (torch.stack(lowers).min(dim=0).values if lowers else torch.ones(m, dtype=torch.float64, device=self.device))
                SAl = float(1.0 - torch.norm(Bt - Lvec, p=2).item())

                if SAh > best_upper[0] + _EPS or (abs(SAh - best_upper[0]) <= _EPS and (best_upper[1] is None or int(cls) < best_upper[1])):
                    best_upper = (SAh, int(cls))
                if SAl > best_lower[0] + _EPS or (abs(SAl - best_lower[0]) <= _EPS and (best_lower[1] is None or int(cls) < best_lower[1])):
                    best_lower = (SAl, int(cls))
            yhat[i] = int(best_lower[1]) if best_lower[0] >= best_upper[0] - _EPS else int(best_upper[1])
        return yhat

    # --------- fast grid: prepare -> fit_from_prepared ----------
    def prepare(self, mu_np: np.ndarray, y_np: np.ndarray) -> None:
        """Precompute parts independent of (gamma, eta) for fast grid."""
        mu = torch.as_tensor(mu_np, dtype=torch.float64, device=self.device)
        self._mu_prepared = mu
        self._y_prepared = np.asarray(y_np, dtype=int)
        self.classes_ = np.unique(self._y_prepared).astype(int)
        self.m_ = mu.shape[1]

        # per-class stats
        self.delta_.clear()
        self.w_.clear()
        for cls in self.classes_:
            self.delta_[int(cls)] = delta_ck(mu, self._y_prepared, int(cls))
            self.w_[int(cls)] = weights_w(mu, self._y_prepared, int(cls))

        # extents candidates (attr + per-class object)
        self._ext_attr_cached = self._attribute_concepts_extents(mu)
        self._ext_obj_cached = {}
        for cls in self.classes_:
            self._ext_obj_cached[int(cls)] = self._object_concepts_extents(mu, self._y_prepared, int(cls))

    def fit_from_prepared(self, gamma: Optional[float] = None, eta: Optional[float] = None) -> "AFFCCL":
        """Use cached extents / stats to build Q^γ and cluster with given (gamma, eta)."""
        if self._mu_prepared is None or self._ext_attr_cached is None or self._ext_obj_cached is None:
            raise RuntimeError("Call prepare() before fit_from_prepared().")
        if gamma is not None:
            self.gamma = float(gamma)
        if eta is not None:
            self.eta = float(eta)

        mu = self._mu_prepared
        self.Q_gamma_.clear()
        for cls in self.classes_:
            self.Q_gamma_[int(cls)] = self._build_Q_gamma_for_class(
                int(cls), mu, self._ext_obj_cached[int(cls)], self._ext_attr_cached
            )
        self.QF_.clear()
        for cls in self.classes_:
            self.QF_[int(cls)] = self._cluster_Q_gamma(int(cls), self.Q_gamma_[int(cls)])
        return self

    # --------- persistence ----------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "AFFCCL":
        with open(path, "rb") as f:
            return pickle.load(f)
