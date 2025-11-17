"""
Esse código contém a implementação de três métodos MTL:
    1) LogisticL21MTL  : penalização ℓ21 por linha (group-lasso por linha).
    2) LogisticDirtyMTL: decomposição W = P + Q com ℓ21(P) + ℓ1(Q).
    3) LogisticRMTFL   : decomposição W = L + S com nuclear(L) + ℓ21(S) e continuation em τ.

OBSERVAÇÕES PRÁTICAS
--------------------
• standardize=True padroniza cada X_t durante o treino. Ao final, trazemos os pesos ao espaço ORIGINAL,
  para que predict/decision_function recebam X bruto (sem padronizar novamente).
• rMTFL: o prox nuclear é caro (SVD). Se p ou T forem grandes, considere randomized SVD.
• Parâmetros:
    - L21   : lam                   (força de seleção de linhas compartilhadas).
    - Dirty : lam1 para ℓ21(P) e lam2 para ℓ1(Q).
    - rMTFL : tau para nuclear(L) e lam para ℓ21(S); tau_init e cont_factor controlam continuation.

Este arquivo é auto-contido e usa apenas NumPy + scikit-learn BaseEstimator para API.
"""

import numpy as np
from numpy.linalg import norm, svd
from sklearn.base import BaseEstimator, ClassifierMixin

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmóide com clipping para estabilidade numérica.
    Entrada:  z ∈ R^{n}
    Saída  :  σ(z) ∈ (0,1)^{n}
    """
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logloss_mean(y: np.ndarray, z: np.ndarray) -> float:
    """
    Log-loss BINÁRIA MÉDIA para rótulos y ∈ {0,1}^{n} e logits z ∈ R^{n}.
    Retorna: (1/n) * Σ [ -y log p - (1-y) log(1-p) ]  onde p = σ(z).
    """
    p = _sigmoid(z)
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _grad_logloss_mean(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Gradiente da LOG-LOSS MÉDIA em relação a w.
    Para X ∈ R^{n×(p+1)} (com viés embutido), y ∈ {0,1}^n, w ∈ R^{p+1}.
    Retorna: (X^T (σ(Xw) - y)) / n
    """
    r = _sigmoid(X @ w) - y
    return (X.T @ r) / X.shape[0]

def _augment_bias(X: np.ndarray) -> np.ndarray:
    """
    Acrescenta uma coluna de 1s (viés) ao final de X.
    Entrada:  X ∈ R^{n×p}
    Saída  :  X^+ ∈ R^{n×(p+1)}  tal que X^+ = [X | 1]
    """
    n = X.shape[0]
    return np.hstack([X, np.ones((n, 1), dtype=X.dtype)])


def _power_spectral_norm(X: np.ndarray, n_iter: int = 100, tol: float = 1e-6, seed: int | None = None) -> float:
    """
    Estima ||X||_2 (norma espectral) por método da potência.
    Usada para derivar a constante de Lipschitz da log-loss (média).
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(X.shape[1],))
    v /= norm(v) + 1e-12
    prev = 0.0
    for _ in range(n_iter):
        u = X @ v
        nu = norm(u)
        if nu == 0.0:
            return 0.0
        u /= nu
        v = X.T @ u
        nv = norm(v)
        if nv == 0.0:
            return 0.0
        v /= nv
        if abs(nv - prev) <= tol * max(1.0, nv):
            break
        prev = nv
    return nv


def _lipschitz_logistic_mean(X: np.ndarray, seed: int | None = None) -> float:
    """
    Constante de Lipschitz da LOG-LOSS MÉDIA:
        L = 0,25 * ||X||_2^2 / n
    (X já inclui coluna do viés; usamos o mesmo L para todos os parâmetros, inclusive viés.)
    """
    s = _power_spectral_norm(X, seed=seed)
    return 0.25 * (s * s) / max(1, X.shape[0])



def _group_l21_norm_rows_except_last(W: np.ndarray) -> float:
    """
    Norma ℓ21 por LINHAS, ignorando a última linha (viés):
        ||W[:-1,:]||_{2,1} = Σ_i ||W[i,:]||_2
    """
    if W.shape[0] <= 1:
        return 0.0
    return np.sum(norm(W[:-1, :], axis=1))


def _l1_norm_rows_except_last(W: np.ndarray) -> float:
    """
    Norma ℓ1 entrywise, ignorando a última linha (viés):
        ||W[:-1,:]||_1 = Σ_{i,j} |W[i,j]|
    """
    if W.shape[0] <= 1:
        return 0.0
    return np.sum(np.abs(W[:-1, :]))


def _nuclear_norm_top_rows(W: np.ndarray) -> float:
    """
    Norma nuclear (soma dos singulares) nas LINHAS 0..p-2, ignorando a última (viés).
    """
    if W.shape[0] <= 1:
        return 0.0
    s = svd(W[:-1, :], full_matrices=False, compute_uv=False)
    return float(np.sum(s))


def _prox_l21_rows_except_last(W: np.ndarray, tau: float) -> np.ndarray:
    """
    Prox da ℓ21 por linha, ignorando o viés:
        prox_{τ||.||_{2,1}}(W) = shrink por linha: w_i ← max(0, 1 - τ/||w_i||_2) * w_i
    Aplica em W[:-1,:] e copia a última linha sem alteração.
    """
    if tau <= 0.0:
        return W
    V = W.copy()
    if V.shape[0] <= 1:
        return V
    R = V[:-1, :]
    rn = norm(R, axis=1)
    scale = np.maximum(0.0, 1.0 - tau / (rn + 1e-12))
    V[:-1, :] = (R.T * scale).T
    return V


def _prox_l1_rows_except_last(W: np.ndarray, tau: float) -> np.ndarray:
    """
    Prox da ℓ1 entrywise, ignorando o viés:
        prox_{τ||.||_1}(W) = soft-threshold por entrada.
    Aplica em W[:-1,:] e copia a última linha sem alteração.
    """
    if tau <= 0.0:
        return W
    V = W.copy()
    if V.shape[0] <= 1:
        return V
    R = V[:-1, :]
    V[:-1, :] = np.sign(R) * np.maximum(np.abs(R) - tau, 0.0)
    return V


def _prox_nuclear_top_rows(Z: np.ndarray, tau: float) -> np.ndarray:
    """
    Prox da norma nuclear nas LINHAS 0..p-2:
        prox_{τ||.||_*}(Z_top) = U diag(max(σ - τ, 0)) V^T
    A última linha (viés) é copiada sem alteração.
    """
    if tau <= 0.0:
        return Z
    if Z.shape[0] <= 1:
        return Z
    Zt = Z[:-1, :]
    U, s, Vt = svd(Zt, full_matrices=False)
    s_shr = np.maximum(s - tau, 0.0)
    top = (U * s_shr) @ Vt
    return np.vstack([top, Z[-1:, :]])


class _BaseMTL(BaseEstimator, ClassifierMixin):
    """
    API base compatível com scikit-learn:
      - fit(X_tasks, y_tasks, task_names)
      - predict_proba(X, task_index)
      - predict(X, task_index)

    Os pesos são trazidos ao ESPAÇO ORIGINAL ao final do fit, para que
    decision_function/predict recebam X bruto.

    Atributos após fit:
      - coef_      : np.ndarray de shape (T, p)  (sem viés)
      - intercept_ : np.ndarray de shape (T,)    (viés por tarefa)
      - _task_order_: lista com a ordem das tarefas usada no treino
      - _stats_    : dicionário t -> (mu, sd) usados na padronização (ou None)
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-5, random_state: int = 0, standardize: bool = False):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.standardize = bool(standardize)
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self._task_order_: list[str] | None = None
        self._stats_: dict[str, tuple[np.ndarray, np.ndarray] | None] | None = None

    def _prep_task(self, X: np.ndarray):
        """
        Retorna X padronizado e estatísticas (mu, sd), ou (X, None) se standardize=False.
        """
        if not self.standardize:
            return X, None
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-12
        return (X - mu) / sd, (mu, sd)

    def _unstandardize_and_set(self, W: np.ndarray, stats_list: list[tuple[np.ndarray, np.ndarray] | None], task_names: list[str]):
        """
        Converte pesos do espaço padronizado para o espaço original:
            w_orig = w_std / sd
            b_orig = b_std - mu^T (w_std / sd)
        e salva coef_, intercept_, ordem e estatísticas.
        """
        coefs, intercepts = [], []
        for j, t in enumerate(task_names):
            w_std = W[:-1, j].ravel()
            b_std = W[-1, j]
            st = stats_list[j]
            if st is None:
                coefs.append(w_std)
                intercepts.append(b_std)
            else:
                mu, sd = st
                sd = sd.ravel()
                mu = mu.ravel()
                w = w_std / sd
                b = b_std - np.dot(mu, w)
                coefs.append(w)
                intercepts.append(b)
        self.coef_ = np.vstack(coefs)
        self.intercept_ = np.array(intercepts)
        self._task_order_ = list(task_names)
        self._stats_ = dict(zip(task_names, stats_list))

    def _task_idx(self, task_index: int | str) -> int:
        return task_index if isinstance(task_index, int) else self._task_order_.index(task_index)

    def decision_function(self, X: np.ndarray, task_index: int | str = 0) -> np.ndarray:
        j = self._task_idx(task_index)
        return X @ self.coef_[j].ravel() + self.intercept_[j]

    def predict_proba(self, X: np.ndarray, task_index: int | str = 0) -> np.ndarray:
        z = self.decision_function(X, task_index)
        p = _sigmoid(z)
        return np.vstack([1.0 - p, p]).T

    def predict(self, X: np.ndarray, task_index: int | str = 0) -> np.ndarray:
        return (self.predict_proba(X, task_index)[:, 1] >= 0.5).astype(int)

class LogisticL21MTL(_BaseMTL):
    """
    Problema:
        min_W  Σ_t mean_logloss_t(W[:,t])  +  lam * ||W[:-1,:]||_{2,1}

    Intuição:
        • ℓ21 por LINHA zera linhas inteiras ⇒ seleção de features compartilhada entre tarefas.
        • Útil quando tarefas são semelhantes e usam quase o mesmo conjunto de variáveis.
    """

    def __init__(self, lam: float = 1.0, max_iter: int = 1000, tol: float = 1e-5, random_state: int = 0, standardize: bool = False):
        super().__init__(max_iter, tol, random_state, standardize)
        self.lam = float(lam)

    def fit(self, X_tasks: dict[str, np.ndarray], y_tasks: dict[str, np.ndarray], task_names: list[str]):
        T = len(task_names); assert T >= 1
        Xs, ys, stats, Ls = [], [], [], []

        for t in task_names:
            Xt, st = self._prep_task(X_tasks[t])
            Xta = _augment_bias(Xt)
            yt = y_tasks[t].astype(float).ravel()
            assert Xta.shape[0] == yt.shape[0]
            Xs.append(Xta); ys.append(yt); stats.append(st)
            Ls.append(_lipschitz_logistic_mean(Xta, seed=self.random_state))

        p1 = Xs[0].shape[1]        
        W = np.zeros((p1, T))
        Y = W.copy()
        tacc = 1.0
        step = 1.0 / max(1e-12, max(Ls))

        def objective(Wm):
            loss = sum(_logloss_mean(ys[j], Xs[j] @ Wm[:, j]) for j in range(T))
            reg = _group_l21_norm_rows_except_last(Wm)
            return loss + self.lam * reg

        prev_obj = objective(W)
        for _ in range(self.max_iter):
            G = np.zeros_like(W)
            for j in range(T):
                G[:, j] = _grad_logloss_mean(Xs[j], ys[j], Y[:, j])

            W_new = _prox_l21_rows_except_last(Y - step * G, step * self.lam)

            tacc_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tacc * tacc))
            Y = W_new + ((tacc - 1.0) / tacc_new) * (W_new - W)

            obj = objective(W_new)
            if abs(obj - prev_obj) <= self.tol * max(1.0, prev_obj):
                W, tacc, prev_obj = W_new, tacc_new, obj
                break
            W, tacc, prev_obj = W_new, tacc_new, obj

        self._unstandardize_and_set(W, stats, task_names)
        return self


class LogisticDirtyMTL(_BaseMTL):
    """
    Problema:
        min_{P,Q}  Σ_t mean_logloss_t(P[:,t] + Q[:,t])
                    + lam1 * ||P[:-1,:]||_{2,1}
                    + lam2 * ||Q[:-1,:]||_1

    Intuição:
        • P captura estrutura COMPARTILHADA por linha (como L21).
        • Q permite EXCEÇÕES esparsas por tarefa (ajustes finos coluna a coluna).
        • Útil quando tarefas são parecidas, mas algumas precisam de correções locais.
    """

    def __init__(self, lam1: float = 1.0, lam2: float = 0.1, max_iter: int = 1000, tol: float = 1e-5, random_state: int = 0, standardize: bool = False):
        super().__init__(max_iter, tol, random_state, standardize)
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)

    def fit(self, X_tasks: dict[str, np.ndarray], y_tasks: dict[str, np.ndarray], task_names: list[str]):
        T = len(task_names); assert T >= 1
        Xs, ys, stats, Ls = [], [], [], []

        for t in task_names:
            Xt, st = self._prep_task(X_tasks[t])
            Xta = _augment_bias(Xt)
            yt = y_tasks[t].astype(float).ravel()
            assert Xta.shape[0] == yt.shape[0]
            Xs.append(Xta); ys.append(yt); stats.append(st)
            Ls.append(_lipschitz_logistic_mean(Xta, seed=self.random_state))

        p1 = Xs[0].shape[1]
        P = np.zeros((p1, T))
        Q = np.zeros((p1, T))
        YP = P.copy()
        YQ = Q.copy()
        tacc = 1.0
        step = 1.0 / max(1e-12, max(Ls))

        def objective(Pm, Qm):
            Wm = Pm + Qm
            loss = sum(_logloss_mean(ys[j], Xs[j] @ Wm[:, j]) for j in range(T))
            reg = self.lam1 * _group_l21_norm_rows_except_last(Pm) + self.lam2 * _l1_norm_rows_except_last(Qm)
            return loss + reg

        prev_obj = objective(P, Q)
        for _ in range(self.max_iter):
            Wm = YP + YQ

            G = np.zeros_like(Wm)
            for j in range(T):
                G[:, j] = _grad_logloss_mean(Xs[j], ys[j], Wm[:, j])

            P_new = _prox_l21_rows_except_last(YP - step * G, step * self.lam1)
            Q_new = _prox_l1_rows_except_last(YQ - step * G, step * self.lam2)

            tacc_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tacc * tacc))
            YP = P_new + ((tacc - 1.0) / tacc_new) * (P_new - P)
            YQ = Q_new + ((tacc - 1.0) / tacc_new) * (Q_new - Q)

            obj = objective(P_new, Q_new)
            if abs(obj - prev_obj) <= self.tol * max(1.0, prev_obj):
                P, Q, tacc, prev_obj = P_new, Q_new, tacc_new, obj
                break
            P, Q, tacc, prev_obj = P_new, Q_new, tacc_new, obj

        W = P + Q
        self._unstandardize_and_set(W, stats, task_names)
        return self


class LogisticRMTFL(_BaseMTL):
    """
    Problema:
        min_{L,S}  Σ_t mean_logloss_t(L[:,t] + S[:,t])
                    + tau * ||L[:-1,:]||_*    (norma nuclear nas linhas 0..p-2)
                    + lam * ||S[:-1,:]||_{2,1}

    Intuição:
        • L modela uma estrutura de BAIXO RANK compartilhada entre tarefas (subespaço comum).
        • S capta variações por LINHA compartilhadas, como no L21.
        • Útil quando existe um subespaço latente global e, além disso, seleção de linhas.

    Continuation em τ:
        • Começamos com τ_init > τ para facilitar a entrada no regime de baixo rank e
          decrescemos progressivamente até τ (fator cont_factor ∈ (0,1)).
        • Acelera e estabiliza a convergência do problema com penalização nuclear.
    """

    def __init__(self, tau: float = 1.0, lam: float = 0.1, max_iter: int = 1200, tol: float = 1e-5,
                 random_state: int = 0, standardize: bool = False, tau_init: float | None = None, cont_factor: float = 0.7):
        super().__init__(max_iter, tol, random_state, standardize)
        self.tau = float(tau)
        self.lam = float(lam)
        self.tau_init = float(tau_init) if tau_init is not None else max(10.0 * self.tau, self.tau + 1.0)
        self.cont_factor = float(cont_factor)

    def fit(self, X_tasks: dict[str, np.ndarray], y_tasks: dict[str, np.ndarray], task_names: list[str]):
        T = len(task_names); assert T >= 1
        Xs, ys, stats, Ls = [], [], [], []

        for t in task_names:
            Xt, st = self._prep_task(X_tasks[t])
            Xta = _augment_bias(Xt)
            yt = y_tasks[t].astype(float).ravel()
            assert Xta.shape[0] == yt.shape[0]
            Xs.append(Xta); ys.append(yt); stats.append(st)
            Ls.append(_lipschitz_logistic_mean(Xta, seed=self.random_state))

        p1 = Xs[0].shape[1]
        Lm = np.zeros((p1, T))
        Sm = np.zeros((p1, T))
        YL = Lm.copy()
        YS = Sm.copy()
        tacc = 1.0
        step = 1.0 / max(1e-12, max(Ls))

        tau_eff = self.tau_init

        def objective(Lcur, Scur, tau_now):
            Wm = Lcur + Scur
            loss = sum(_logloss_mean(ys[j], Xs[j] @ Wm[:, j]) for j in range(T))
            reg = tau_now * _nuclear_norm_top_rows(Lcur) + self.lam * _group_l21_norm_rows_except_last(Scur)
            return loss + reg

        prev_obj = objective(Lm, Sm, tau_eff)

        for _ in range(self.max_iter):
            if tau_eff > self.tau:
                tau_eff = max(self.tau, self.cont_factor * tau_eff)

            Wm = YL + YS

            G = np.zeros_like(Wm)
            for j in range(T):
                G[:, j] = _grad_logloss_mean(Xs[j], ys[j], Wm[:, j])

            ZL = YL - step * G
            L_new = _prox_nuclear_top_rows(ZL, step * tau_eff)

            ZS = YS - step * G
            S_new = _prox_l21_rows_except_last(ZS, step * self.lam)

            tacc_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tacc * tacc))
            YL = L_new + ((tacc - 1.0) / tacc_new) * (L_new - Lm)
            YS = S_new + ((tacc - 1.0) / tacc_new) * (S_new - Sm)

            obj = objective(L_new, S_new, tau_eff)
            if abs(obj - prev_obj) <= self.tol * max(1.0, prev_obj):
                Lm, Sm, tacc, prev_obj = L_new, S_new, tacc_new, obj
                break
            Lm, Sm, tacc, prev_obj = L_new, S_new, tacc_new, obj

        W = Lm + Sm
        self._unstandardize_and_set(W, stats, task_names)
        return self



def mtl_logloss_mean(model: _BaseMTL, X_tasks: dict[str, np.ndarray], y_tasks: dict[str, np.ndarray]) -> float:
    """
    Soma das log-loss MÉDIAS por tarefa usando os pesos finais no ESPAÇO ORIGINAL.
    Útil para verificar se mudanças de hiperparâmetros estão de fato reduzindo o objetivo suave.
    """
    total = 0.0
    for j, t in enumerate(model._task_order_):
        X = X_tasks[t]
        y = y_tasks[t].astype(float).ravel()
        z = X @ model.coef_[j].ravel() + model.intercept_[j]
        total += _logloss_mean(y, z)
    return total