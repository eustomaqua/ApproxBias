# coding: utf-8
"""
Metric-parametrized HFM under L_p ground metrics.

For an L_p ground distance d_p(u, v)=||u-v||_p, a linear projection
    phi_w(u) = w^T u
is 1-Lipschitz whenever ||w||_q <= 1, where q is the dual exponent of p:
    1/p + 1/q = 1.
This module implements dual-norm normalisation and uses it in projection-based
nearest cross-group distance computation:

- StratES: exact early-stopping search under a valid 1-Lipschitz projection.
- StratRA: fixed-window candidate-set approximation.
- Seeded StratES: StratRA first gives per-point upper bounds; StratES then
  refines exactly, usually with fewer scanned candidates.
- HFM helpers for single and multiple sensitive attributes.
- Rank stability and sign consistency helpers for ground-metric robustness.
"""


from typing import Dict, Iterable, Mapping, Optional, Tuple, Union, Any
import math
import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree, distance

# def njit(*args, **kwargs):
#     if args and callable(args[0]):
#         return args[0]
#     return lambda f: f
#
# prange = range

from hfm.utils.verifiers import INF64, EPS64
from hfm.utils.decorators import fantasy_timer
import pdb
ArrayLike = Union[np.ndarray, Iterable[float]]
PType = Union[int, float, str]


# -----------------------------------------------------------------------------
# Dual exponent and dual-norm normalisation


@njit
def _as_float_p(p: PType) -> float:
    """Convert common p specifications into float, including 'inf'."""
    if isinstance(p, str):
        if p.lower() in {"inf", "infty", "infinity", "linf", "l_inf",
                         "l-infinity"}:
            return float("inf")
        p = float(p)
    else:
        p = float(p)
    if p < 1.0:
        raise ValueError("L_p distance requires p >= 1.")
    return p  # float(p)


@njit
def dual_exponent(p: PType) -> float:
    """
    Return the dual exponent q of p, where 1/p + 1/q = 1.

    Cases:
        p = 1      -> q = inf
        p = 2      -> q = 2
        p = inf    -> q = 1
        1 < p < inf -> q = p/(p-1)
    """
    p = _as_float_p(p)
    # if p < 1.0:
    #     raise ValueError("L_p norm requires p >= 1.")
    if p == 1.0:
        return float("inf")
    if math.isinf(p):
        return 1.0
    return p / (p - 1.0)


@njit
def ellp_norm_vec(x: ArrayLike, p: PType) -> float:
    # def vector_norm
    """Compute ||x||_p for p in [1, inf)."""
    p = _as_float_p(p)
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    if x.size == 0:
        return 0.0
    # if p < 1.0:
    #     raise ValueError("L_p norm requires p >= 1.")
    if p == 1.0:
        return float(np.sum(ax))
    if p == 2.0:
        return float(np.sqrt(np.dot(x, x)))
        # return float(np.linalg.norm(x))
    if math.isinf(p):
        return float(np.max(ax))  # if x.size else 0.0
    return float(np.sum(ax ** p) ** (1.0 / p))


# @fantasy_timer
# def dual_normalized_vec(w, p, eps=EPS64):
#     out = w.copy()
#     L = ellp_norm_vec(w, dual_exponent(p))
#     if L <= eps:
#         return out
#     for i in range(out.shape[0]):
#         out[i] /= L
#     return out


# @fantasy_timer
# @njit
def dual_normalize(w: ArrayLike, p: PType, *,
                   target_norm: float = 1.0, eps: float = EPS64,
                   ) -> np.ndarray:
    """
    Normalise a projection vector w so that ||w||_q = target_norm <= 1,
    where q is the dual exponent of the L_p ground metric.

    This guarantees that phi_w(x)=w^T x is target_norm-Lipschitz w.r.t. L_p,
    and 1-Lipschitz when target_norm <= 1.

    Examples
    --------
    L1 ground metric:  q=inf, divide by max(abs(w)).
    L2 ground metric:  q=2,   divide by sqrt(sum(w^2)).
    Linf ground metric:q=1,   divide by sum(abs(w)).
    Lp ground metric:  q=p/(p-1).
    """
    # if target_norm <= 0 or target_norm > 1.0 + eps:
    if not (0.0 < target_norm <= 1.0 + eps):
        raise ValueError(
            "target_norm should be in (0, 1] to keep projections 1-Lipschitz.")
    w = np.asarray(w, dtype=np.float64).copy()
    q = dual_exponent(p)
    nrm = ellp_norm_vec(w, q)
    if nrm <= eps:
        raise ValueError("Cannot normalise a near-zero projection vector.")
    return (target_norm / nrm) * w


def random_dual_unit_vectors(dim: int, n_vectors: int = 1, p: PType = 2.0, *,
                             rng: Optional[np.random.Generator] = None,
                             # seed: Optional[int] = None,
                             distribution: str = "normal",
                             target_norm: float = 1.0,
                             ) -> np.ndarray:
    """
    Generate random projection vectors satisfying ||w||_q = target_norm,
    where q is the dual exponent of the L_p ground metric.

    For p=2, these are ordinary Euclidean unit vectors. For other p values,
    they are not necessarily orthogonal; orthogonality is an L2 notion and is
    not required for the 1-Lipschitz lower-bound property.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if n_vectors <= 0:
        raise ValueError("n_vectors must be positive.")
    if rng is None:
        rng = np.random.default_rng()
    # rng = np.random.default_rng(seed)

    if distribution == "normal":
        W = rng.normal(size=(n_vectors, dim))
    elif distribution == "uniform":
        W = rng.uniform(-1.0, 1.0, size=(n_vectors, dim))
    else:
        raise ValueError("distribution must be 'normal' or 'uniform'.")

    for k in range(n_vectors):
        # In the extremely unlikely case of a zero vector, resample it.
        while ellp_norm_vec(W[k], dual_exponent(p)) <= EPS64:
            W[k] = rng.normal(size=dim)
        W[k] = dual_normalize(W[k], p, target_norm=target_norm)
    return W.astype(np.float64, copy=False)


# -----------------------------------------------------------------------------
# Numba-compatible Lp distance and projection search kernels


@njit(cache=True)
def _lp_distance_rows(X: np.ndarray, i: int, j: int, p: float) -> float:
    """Lp distance between rows X[i] and X[j], numba-compatible."""
    d = X.shape[1]
    if p == 1.0:
        s = 0.0
        for k in range(d):
            s += abs(X[i, k] - X[j, k])
        return s
    if p == 2.0:
        s = 0.0
        for k in range(d):
            t = X[i, k] - X[j, k]
            s += t * t
        return math.sqrt(s)
    if math.isinf(p):
        mx = 0.0
        for k in range(d):
            t = abs(X[i, k] - X[j, k])
            if t > mx:
                mx = t
        return mx
    s = 0.0
    for k in range(d):
        s += abs(X[i, k] - X[j, k]) ** p
    return s ** (1.0 / p)


@njit(cache=True)
def _has_cross_group(A: np.ndarray) -> bool:
    n = A.shape[0]
    if n <= 1:
        return False
    first = A[0]
    for i in range(1, n):
        if A[i] != first:
            return True
    return False


@njit(cache=True)
def _direct_dmin_single_sa(X: np.ndarray, A: np.ndarray, p: float) -> np.ndarray:
    """Exact O(n^2) nearest cross-group distances for validation."""
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        best = INF64
        ai = A[i]
        for j in range(n):
            if A[j] == ai:
                continue
            curr = _lp_distance_rows(X, i, j, p)
            if curr < best:
                best = curr
        out[i] = best
    return out


@njit(cache=True)
def _strat_es_dmin_single_projection(
    X: np.ndarray,
    A: np.ndarray,
    w: np.ndarray,
    p: float,
    seed_upper: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact nearest cross-group distances using one 1-Lipschitz projection.

    The scan expands from the anchor in projected order. It stops once the
    nearest unscanned point on either side has projected gap >= current best.
    This is exact because projected gap <= true Lp distance.

    Returns
    -------
    d_min : shape (n,)
        Exact nearest cross-group distances.
    scanned : shape (n,)
        Number of true distance computations performed per anchor.
    """
    n = X.shape[0]
    proj = X @ w
    order = np.argsort(proj)

    d_min = np.empty(n, dtype=np.float64)
    scanned = np.zeros(n, dtype=np.int64)

    for pos in prange(n):
        i = order[pos]
        ai = A[i]
        gi = proj[i]
        best = INF64
        if seed_upper is not None:
            if seed_upper[i] < best:
                best = seed_upper[i]

        left = pos - 1
        right = pos + 1

        while left >= 0 or right < n:
            gap_left = INF64
            gap_right = INF64
            if left >= 0:
                gap_left = gi - proj[order[left]]
            if right < n:
                gap_right = proj[order[right]] - gi

            # If the smallest possible projected gap is already no better than
            # current best, then no unscanned point can improve the answer.
            if gap_left >= best and gap_right >= best:
                break

            if gap_left <= gap_right:
                j = order[left]
                # Move pointer now; same-group points can still certify that
                # all further points on this side are even farther in projection.
                left -= 1
            else:
                j = order[right]
                right += 1

            if A[j] == ai:
                continue

            curr = _lp_distance_rows(X, i, j, p)
            scanned[i] += 1
            if curr < best:
                best = curr

        d_min[i] = best

    return d_min, scanned


@njit(cache=True)
def _strat_ra_dmin_one_projection(
    X: np.ndarray,
    A: np.ndarray,
    w: np.ndarray,
    p: float,
    m2: int,
) -> np.ndarray:
    """
    Fixed-window candidate distances for one projection.

    For each anchor, scan at most m2 cross-group candidates to the left and
    at most m2 cross-group candidates to the right in projected order.
    """
    n = X.shape[0]
    proj = X @ w
    order = np.argsort(proj)
    out = np.empty(n, dtype=np.float64)

    for pos in prange(n):
        i = order[pos]
        ai = A[i]
        best = INF64

        # Left side
        count = 0
        jpos = pos - 1
        while jpos >= 0 and count < m2:
            j = order[jpos]
            if A[j] != ai:
                curr = _lp_distance_rows(X, i, j, p)
                if curr < best:
                    best = curr
                count += 1
            jpos -= 1

        # Right side
        count = 0
        jpos = pos + 1
        while jpos < n and count < m2:
            j = order[jpos]
            if A[j] != ai:
                curr = _lp_distance_rows(X, i, j, p)
                if curr < best:
                    best = curr
                count += 1
            jpos += 1

        out[i] = best

    return out


# -----------------------------------------------------------------------------
# Public distance-computation APIs


def _validate_inputs(X: np.ndarray, A: np.ndarray, *, require_cross_group: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    A = np.asarray(A)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if A.ndim != 1:
        raise ValueError("A must be a 1D sensitive-attribute vector for single-SA functions.")
    if X.shape[0] != A.shape[0]:
        raise ValueError("X and A must have the same number of rows.")
    if require_cross_group and len(np.unique(A)) < 2:
        raise ValueError("At least two sensitive-attribute values are required for cross-group distances.")
    return X, A


def aggregate_dmin(d_min: np.ndarray) -> Tuple[float, float]:
    """Return (max nearest cross-group distance, average nearest cross-group distance)."""
    d_min = np.asarray(d_min, dtype=np.float64)
    if np.any(~np.isfinite(d_min)):
        raise ValueError("Some points have no finite cross-group candidate distance.")
    return float(np.max(d_min)), float(np.mean(d_min))


@fantasy_timer
def direct_single_sa(
    X: np.ndarray,
    A: np.ndarray,
    *,
    p: Union[int, float, str] = 2.0,
    return_dmin: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray]]:
    """Exact O(n^2) nearest cross-group distance computation for one SA."""
    p = _as_float_p(p)
    X, A = _validate_inputs(X, A)
    d_min = _direct_dmin_single_sa(X, A, p)
    d_max, d_avg = aggregate_dmin(d_min)
    if return_dmin:
        return d_max, d_avg, d_min
    return d_max, d_avg


@fantasy_timer
def strat_es_single_sa(
    X: np.ndarray,
    A: np.ndarray,
    *,
    p: Union[int, float, str] = 2.0,
    w: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    seed_upper: Optional[np.ndarray] = None,
    return_dmin: bool = False,
    return_scanned: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray], Tuple[float, float, np.ndarray, np.ndarray]]:
    """
    Exact StratES for one sensitive attribute under L_p distance.

    Parameters
    ----------
    X : array, shape (n, d)
        Points where distances are evaluated, e.g. concatenate non-sensitive
        features and true labels/predictions.
    A : array, shape (n,)
        One sensitive attribute.
    p : float or 'inf'
        L_p ground metric.
    w : optional array, shape (d,)
        Projection vector. If supplied, it is dual-normalised automatically.
    seed_upper : optional array, shape (n,)
        Per-point valid upper bounds. Usually from StratRA. Exactness is
        preserved because these are only used as initial best distances.
    """
    p = _as_float_p(p)
    X, A = _validate_inputs(X, A)
    n, d = X.shape

    if w is None:
        rng = np.random.default_rng(seed)
        w = random_dual_unit_vectors(d, 1, p, rng=rng)[0]
    else:
        w = dual_normalize(w, p)

    if seed_upper is not None:
        seed_upper = np.asarray(seed_upper, dtype=np.float64)
        if seed_upper.shape != (n,):
            raise ValueError("seed_upper must have shape (n_samples,).")
    d_min, scanned = _strat_es_dmin_single_projection(X, A, w.astype(np.float64), p, seed_upper)
    d_max, d_avg = aggregate_dmin(d_min)

    if return_dmin and return_scanned:
        return d_max, d_avg, d_min, scanned
    if return_dmin:
        return d_max, d_avg, d_min
    if return_scanned:
        return d_max, d_avg, scanned
    return d_max, d_avg


def strat_ra_dmin_single_sa(
    X: np.ndarray,
    A: np.ndarray,
    *,
    p: Union[int, float, str] = 2.0,
    m1: int = 25,
    m2: Optional[int] = None,
    n_e: int = 2,
    seed: Optional[int] = None,
    distribution: str = "normal",
) -> np.ndarray:
    """
    StratRA candidate-set approximation for one sensitive attribute.

    Returns per-point approximate nearest cross-group distances. These values
    are conservative upper approximations: approximate d_i >= exact d_i, as long
    as each candidate distance is computed exactly and missing candidates remain
    at +inf.
    """
    p = _as_float_p(p)
    X, A = _validate_inputs(X, A)
    n, d = X.shape
    if m1 <= 0:
        raise ValueError("m1 must be positive.")
    if n_e <= 0:
        raise ValueError("n_e must be positive.")
    if m2 is None:
        m2 = max(1, int(math.ceil(2.0 * math.log2(max(n, 2)))))
    if m2 <= 0:
        raise ValueError("m2 must be positive.")

    rng = np.random.default_rng(seed)
    best = np.full(n, INF64, dtype=np.float64)
    for _ in range(m1):
        W = random_dual_unit_vectors(d, n_e, p, rng=rng, distribution=distribution)
        for k in range(n_e):
            cand = _strat_ra_dmin_one_projection(X, A, W[k], p, int(m2))
            best = np.minimum(best, cand)
    return best


@fantasy_timer
def strat_ra_single_sa(
    X: np.ndarray,
    A: np.ndarray,
    *,
    p: Union[int, float, str] = 2.0,
    m1: int = 25,
    m2: Optional[int] = None,
    n_e: int = 2,
    seed: Optional[int] = None,
    return_dmin: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray]]:
    """Return StratRA approximate (max, avg) distances for one SA."""
    d_min = strat_ra_dmin_single_sa(X, A, p=p, m1=m1, m2=m2, n_e=n_e, seed=seed)
    d_max, d_avg = aggregate_dmin(d_min)
    if return_dmin:
        return d_max, d_avg, d_min
    return d_max, d_avg


def seeded_strat_es_single_sa(
    X: np.ndarray, A: np.ndarray, *,
    p: Union[int, float, str] = 2.0,
    m1: int = 10, m2: Optional[int] = None, n_e: int = 2,
    seed: Optional[int] = None,
    return_dmin: bool = False,
    return_scanned: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray], Tuple[float, float, np.ndarray, np.ndarray]]:
    """
    StratRA-seeded exact StratES.

    StratRA first generates valid per-point upper bounds. StratES then refines
    them exactly under a fresh dual-normalised 1-Lipschitz projection.
    """
    seed_upper = strat_ra_dmin_single_sa(X, A, p=p, m1=m1, m2=m2, n_e=n_e, seed=seed)
    # Use a different random stream for the final exact projection.
    final_seed = None if seed is None else seed + 1000003
    return strat_es_single_sa(
        X,
        A,
        p=p,
        seed=final_seed,
        seed_upper=seed_upper,
        return_dmin=return_dmin,
        return_scanned=return_scanned,
    )


def distance_single_sa(X: np.ndarray, A: np.ndarray, *,
                       p: Union[int, float, str] = 2.0,
                       strategy: str = "seeded_es",
                       m1: int = 10,
                       m2: Optional[int] = None,
                       n_e: int = 2,
                       seed: Optional[int] = None,
                       return_dmin: bool = False,
                       ):
    """
    Unified API for one-SA distance computation.

    strategy in {'direct', 'strat_es', 'strat_ra', 'seeded_es'}.
    """
    s = strategy.lower()
    if s in {"direct", "naive"}:
        return direct_single_sa(X, A, p=p, return_dmin=return_dmin)
    if s in {"es", "strat_es", "strates"}:
        return strat_es_single_sa(X, A, p=p, seed=seed, return_dmin=return_dmin)
    if s in {"ra", "strat_ra", "stratra"}:
        return strat_ra_single_sa(X, A, p=p, m1=m1, m2=m2, n_e=n_e, seed=seed, return_dmin=return_dmin)
    if s in {"seeded_es", "ra_seeded_es", "seeded_strat_es"}:
        return seeded_strat_es_single_sa(X, A, p=p, m1=m1, m2=m2, n_e=n_e, seed=seed, return_dmin=return_dmin)
    raise ValueError("Unknown strategy. Use 'direct', 'strat_es', 'strat_ra', or 'seeded_es'.")


def distance_multivar(
    X: np.ndarray,
    A: np.ndarray,
    *,
    p: Union[int, float, str] = 2.0,
    strategy: str = "seeded_es",
    m1: int = 10,
    m2: Optional[int] = None,
    n_e: int = 2,
    seed: Optional[int] = None,
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    """
    Multiple-SA version of the distance computation.

    Returns
    -------
    D_max : max over SAs of one-SA maximum distances.
    D_avg : average over SAs of one-SA average distances.
    details : dict with per-SA arrays.
    """
    X = np.asarray(X, dtype=np.float64)
    A = np.asarray(A)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if X.shape[0] != A.shape[0]:
        raise ValueError("X and A must have the same number of rows.")

    n_a = A.shape[1]
    per_max = np.empty(n_a, dtype=np.float64)
    per_avg = np.empty(n_a, dtype=np.float64)

    for r in range(n_a):
        # Offset seeds so columns do not reuse identical directions.
        col_seed = None if seed is None else int(seed + 7919 * r)
        per_max[r], per_avg[r] = distance_single_sa(
            X,
            A[:, r],
            p=p,
            strategy=strategy,
            m1=m1,
            m2=m2,
            n_e=n_e,
            seed=col_seed,
            return_dmin=False,
        )

    details = {"per_sa_max": per_max, "per_sa_avg": per_avg}
    return float(np.max(per_max)), float(np.mean(per_avg)), details


# -----------------------------------------------------------------------------
# HFM helpers


def _safe_log_ratio(num: float, den: float, eps: float = 0.0) -> float:
    if eps > 0.0:
        num = max(float(num), eps)
        den = max(float(den), eps)
    if num <= 0.0 or den <= 0.0:
        raise ValueError("HFM log-ratio requires positive numerator and denominator distances.")
    return float(math.log(num / den))


def hfm_scores(
    X_data: np.ndarray,
    X_model: np.ndarray,
    A: np.ndarray,
    *,
    p: Union[int, float, str] = 2.0,
    strategy: str = "seeded_es",
    m1: int = 10,
    m2: Optional[int] = None,
    n_e: int = 2,
    seed: Optional[int] = None,
    eps: float = 0.0,
) -> Dict[str, object]:
    """
    Compute maximum HFM and average HFM for a chosen L_p ground metric.

    X_data  should contain the points for data geometry, e.g. [x_non_sensitive, y].
    X_model should contain the points for model geometry, e.g. [x_non_sensitive, y_hat].
    A       contains one or multiple sensitive attributes.
    """
    D_data_max, D_data_avg, data_details = distance_multivar(
        X_data, A, p=p, strategy=strategy, m1=m1, m2=m2, n_e=n_e, seed=seed
    )
    model_seed = None if seed is None else seed + 104729
    D_model_max, D_model_avg, model_details = distance_multivar(
        X_model, A, p=p, strategy=strategy, m1=m1, m2=m2, n_e=n_e, seed=model_seed
    )
    return {
        "p": p,
        "strategy": strategy,
        "hfm_max": _safe_log_ratio(D_model_max, D_data_max, eps=eps),
        "hfm_avg": _safe_log_ratio(D_model_avg, D_data_avg, eps=eps),
        "D_data_max": D_data_max,
        "D_data_avg": D_data_avg,
        "D_model_max": D_model_max,
        "D_model_avg": D_model_avg,
        "data_details": data_details,
        "model_details": model_details,
    }


# -----------------------------------------------------------------------------
# Ground-metric robustness helpers: rank stability and sign consistency


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        # ranks are 1-based; average of positions i+1 through j
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: ArrayLike, y: ArrayLike) -> float:
    """Pearson correlation with a zero-variance guard."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    den = math.sqrt(float(np.dot(xm, xm) * np.dot(ym, ym)))
    if den <= 0.0:
        return float("nan")
    return float(np.dot(xm, ym) / den)


def spearman_rank_corr(x: ArrayLike, y: ArrayLike) -> float:
    """Spearman rank correlation using average ranks for ties."""
    return pearson_corr(_rankdata_average(np.asarray(x)), _rankdata_average(np.asarray(y)))


def sign_eps(x: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    """Three-valued sign: +1 if x>eps, -1 if x<-eps, otherwise 0."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.int64)
    out[x > eps] = 1
    out[x < -eps] = -1
    return out


def sign_agreement(x: ArrayLike, y: ArrayLike, eps: float = 1e-6) -> float:
    """Pairwise sign agreement rate between two HFM score arrays."""
    sx = sign_eps(x, eps)
    sy = sign_eps(y, eps)
    if sx.shape != sy.shape:
        raise ValueError("x and y must have the same shape.")
    return float(np.mean(sx == sy))


def all_metric_sign_consistency(scores_by_metric: Dict[object, ArrayLike], eps: float = 1e-6) -> float:
    """
    Fraction of evaluation units whose HFM sign agrees across all metrics.
    scores_by_metric maps metric labels, e.g. 'L1', 'L2', 'Linf', to arrays.
    """
    keys = list(scores_by_metric.keys())
    if not keys:
        raise ValueError("scores_by_metric cannot be empty.")
    signs = [sign_eps(scores_by_metric[k], eps) for k in keys]
    base = signs[0]
    for s in signs[1:]:
        if s.shape != base.shape:
            raise ValueError("All score arrays must have the same shape.")
    stacked = np.vstack(signs)
    return float(np.mean(np.all(stacked == stacked[0:1, :], axis=0)))


def robustness_against_baseline(
    scores_by_metric: Dict[object, ArrayLike],
    *,
    baseline: object = 2,
    eps: float = 1e-6,
) -> Dict[object, Dict[str, float]]:
    """
    Compute rank stability and sign consistency against a baseline metric.

    Returns a nested dict:
        metric -> {'spearman': ..., 'sign_agreement': ...}
    """
    if baseline not in scores_by_metric:
        raise KeyError(f"baseline {baseline!r} not found in scores_by_metric.")
    base = np.asarray(scores_by_metric[baseline], dtype=np.float64)
    out: Dict[object, Dict[str, float]] = {}
    for key, val in scores_by_metric.items():
        arr = np.asarray(val, dtype=np.float64)
        out[key] = {
            "spearman": spearman_rank_corr(arr, base),
            "sign_agreement": sign_agreement(arr, base, eps=eps),
        }
    return out


# -----------------------------------------------------------------------------
# Small smoke-test helper


def _smoke_test() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))
    A = rng.integers(0, 4, size=80)
    for p in [1, 2, 3, float("inf")]:
        W = random_dual_unit_vectors(X.shape[1], 3, p, rng=rng)
        q = dual_exponent(p)
        norms = [ellp_norm_vec(w, q) for w in W]
        print("p=", p, "q=", q, "dual norms=", norms)

        d0 = direct_single_sa(X, A, p=p)
        d1 = strat_es_single_sa(X, A, p=p, seed=42)
        print("direct", d0, "strat_es", d1, "abs diff", np.abs(np.array(d0) - np.array(d1)))


if __name__ == "__main__":
    _smoke_test()
