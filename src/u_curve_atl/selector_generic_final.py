
from __future__ import annotations
import numpy as np
import pandas as pd
import optuna
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable
from inspect import signature
from collections import deque 

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import get_scorer
from sklearn.utils.multiclass import type_of_target

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _is_classification(y) -> bool:
    t = type_of_target(y)
    return t in ("binary", "multiclass", "multiclass-indicator")

def _supports_sample_weight(est: BaseEstimator) -> bool:
    try:
        return "sample_weight" in signature(est.fit).parameters
    except Exception:
        return False

def _choose_scoring(scoring, y, estimator):
    if scoring is None or scoring == "auto":
        if _is_classification(y):
            if hasattr(estimator, "predict_proba"):
                return get_scorer("neg_log_loss")
            if hasattr(estimator, "decision_function"):
                return get_scorer("roc_auc") if type_of_target(y) == "binary" else get_scorer("roc_auc_ovr")
            return get_scorer("accuracy")
        else:
            return get_scorer("neg_mean_squared_error")
    if isinstance(scoring, str):
        return get_scorer(scoring)
    return scoring

def _as_builtin(obj):
    if isinstance(obj, dict):
        return { _as_builtin(k): _as_builtin(v) for k, v in obj.items() }
    if isinstance(obj, list):
        return [ _as_builtin(v) for v in obj ]
    if isinstance(obj, tuple):
        return tuple(_as_builtin(v) for v in obj)
    import numpy as np, pandas as pd
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):return obj.isoformat()
    return obj

def _resample_for_weights(
    rng: np.random.Generator,
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    weights: List[float],
    n_target: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    X_out = [X_list[0]]
    y_out = [y_list[0]]
    for Xs, ys, w in zip(X_list[1:], y_list[1:], weights[1:]):
        k = max(0, int(round(float(w) * n_target)))
        if k == 0 or len(ys) == 0:
            continue
        idx = rng.choice(len(ys), size=k, replace=(k > len(ys)))
        X_out.append(Xs[idx])
        y_out.append(ys[idx])
    return X_out, y_out

def _half_source(rng: np.random.Generator, Xs: np.ndarray, ys: np.ndarray, frac: float = 0.5):
    if len(ys) == 0 or frac >= 1.0:
        return Xs, ys
    k = max(1, int(round(frac * len(ys))))
    idx = rng.choice(len(ys), size=k, replace=False)
    return Xs[idx], ys[idx]

@dataclass
class _Combo:
    tasks: List[str]         
    val_loss: float
    params: Dict             
    weights: Dict[str, float] 

class UcurveATLSelectorGeneric:
    """
    Seleção por branch-and-bound em largura com poda heurística (Curva-U).
    Hiperparâmetros do estimador são otimizados UMA vez no alvo e fixos depois.  
    Apenas o peso da NOVA fonte é otimizado a cada expansão.                      
    """
    def __init__(
        self,
        estimator: BaseEstimator,
        param_space: Callable,
        scoring: Union[str, Callable, None] = "auto",
        depth: int = 3,
        n_trials: int = 80,
        weight_min: float = 0.0,
        weight_max: float = 1.0,
        preprocess: Optional[object] = None,
        test_size: float = 0.50,     
        val_size: float = 0.25,     
        random_state: int = 42,
        improve_delta: float = 0.01, 
    ):
        self.estimator = estimator
        self.param_space = param_space
        self.scoring = scoring
        self.depth = int(depth)
        self.n_trials = int(n_trials)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.preprocess = preprocess
        self.test_size = float(test_size)
        self.val_size = float(val_size)
        self.random_state = int(random_state)
        self.improve_delta = float(improve_delta)

        self.history_ = pd.DataFrame()
        self.best_: Optional[_Combo] = None
        self.target_: Optional[str] = None
        self.splits_: Dict[str, np.ndarray] = {}
        self._base_params_: Optional[Dict] = None  

    def _make_target_splits(self, X_target, y_target):
        strat_all = y_target if _is_classification(y_target) else None
        Xt_trv, Xt_test, yt_trv, yt_test = train_test_split(
            X_target, y_target,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=strat_all
        )
        rel_val = self.val_size / (1.0 - self.test_size)
        strat_trv = yt_trv if _is_classification(yt_trv) else None
        Xt_train, Xt_val, yt_train, yt_val = train_test_split(
            Xt_trv, yt_trv,
            test_size=rel_val,
            random_state=self.random_state + 1,
            shuffle=True,
            stratify=strat_trv
        )
        self.splits_ = {
            "X_train": Xt_train, "y_train": yt_train,
            "X_val": Xt_val, "y_val": yt_val,
            "X_test": Xt_test, "y_test": yt_test
        }

    def _prep_fit_transform(
        self,
        X_train_blocks: List[np.ndarray],
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        if self.preprocess is None:
            return X_train_blocks, X_val, X_test
        prep = clone(self.preprocess)
        Xtr_comb_for_prep = np.concatenate(X_train_blocks, axis=0)
        prep.fit(Xtr_comb_for_prep)
        X_train_blocks_s = [prep.transform(Xb) for Xb in X_train_blocks]
        X_val_s  = prep.transform(X_val)
        X_test_s = prep.transform(X_test)
        return X_train_blocks_s, X_val_s, X_test_s

    def _build_train_blocks(
        self,
        X_tasks: Dict[str, np.ndarray],
        y_tasks: Dict[str, np.ndarray],
        target: str,
        sources: List[str],
        weights_by_source: Dict[str, float],
        rng: np.random.Generator
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
        Xt_tr = self.splits_["X_train"]; yt_tr = self.splits_["y_train"]
        X_blocks = [Xt_tr]; y_blocks = [yt_tr]
        w_vec = None

        if _supports_sample_weight(self.estimator):
            n_t = len(yt_tr)
            w = [np.ones(n_t, dtype=float)]
            for s in sources:
                Xs, ys = X_tasks[s], y_tasks[s]
                Xs, ys = _half_source(rng, Xs, ys, 0.5)  
                X_blocks.append(Xs)
                y_blocks.append(ys)
                w.append(np.full(len(ys), float(weights_by_source.get(s, 0.0)), dtype=float))
            w_vec = np.concatenate(w, axis=0)
        else:
            X_src, y_src = [], []
            for s in sources:
                Xs, ys = _half_source(rng, X_tasks[s], y_tasks[s], 0.5)  
                X_src.append(Xs); y_src.append(ys)
            w_list = [1.0] + [float(weights_by_source.get(s, 0.0)) for s in sources]
            X_blocks, y_blocks = _resample_for_weights(
                rng, [Xt_tr] + X_src, [yt_tr] + y_src, w_list, n_target=len(yt_tr)
            )
            w_vec = None

        return X_blocks, y_blocks, w_vec

    def _tune_params_on_target(self, base_est: BaseEstimator) -> Tuple[float, Dict]:
        scorer = _choose_scoring(self.scoring, self.splits_["y_val"], base_est)

        def objective(trial: optuna.Trial):
            params = self.param_space(trial)
            clf = clone(self.estimator).set_params(**params)
            Xtr = self.splits_["X_train"]; ytr = self.splits_["y_train"]
            Xval = self.splits_["X_val"];   yval = self.splits_["y_val"]
            if self.preprocess is None:
                Xtr_s, Xval_s = Xtr, Xval
            else:
                prep = clone(self.preprocess)
                prep.fit(Xtr)
                Xtr_s = prep.transform(Xtr)
                Xval_s = prep.transform(Xval)
            clf.fit(Xtr_s, ytr)
            return -float(scorer(clf, Xval_s, yval))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_params = _as_builtin(self.param_space(study.best_trial))

        clf = clone(self.estimator).set_params(**best_params)
        scorer = _choose_scoring(self.scoring, self.splits_["y_val"], clf)
        Xtr = self.splits_["X_train"]; ytr = self.splits_["y_train"]
        Xval = self.splits_["X_val"];   yval = self.splits_["y_val"]
        if self.preprocess is None:
            Xtr_s, Xval_s = Xtr, Xval
        else:
            prep = clone(self.preprocess)
            prep.fit(Xtr)
            Xtr_s = prep.transform(Xtr)
            Xval_s = prep.transform(Xval)
        clf.fit(Xtr_s, ytr)
        val_loss = -float(scorer(clf, Xval_s, yval))
        return float(val_loss), best_params

    def _optimize_new_source_weight(
        self,
        combo_tasks: List[str],
        X_tasks: Dict[str, np.ndarray],
        y_tasks: Dict[str, np.ndarray],
        params_fixed: Dict,
        fixed_weights: Dict[str, float],
        new_source: Optional[str]
    ) -> Tuple[float, Dict[str, float]]:
        target = combo_tasks[0]
        sources = [t for t in combo_tasks[1:]]
        base_est = clone(self.estimator).set_params(**params_fixed)
        scorer = _choose_scoring(self.scoring, self.splits_["y_val"], base_est)

        def objective(trial: optuna.Trial):
            rng = np.random.default_rng(self.random_state + int(trial.number))
            weights = dict(fixed_weights)
            if new_source is not None:
                w = trial.suggest_float(f"w_{new_source}", self.weight_min, self.weight_max)
                weights[new_source] = float(w)
            X_blocks, y_blocks, sw = self._build_train_blocks(
                X_tasks, y_tasks, target, sources, weights, rng
            )
            X_blocks_s, X_val_s, _ = self._prep_fit_transform(
                X_blocks, self.splits_["X_val"], self.splits_["X_test"]
            )
            Xtr = np.concatenate(X_blocks_s, axis=0)
            ytr = np.concatenate(y_blocks, axis=0)
            clf = clone(self.estimator).set_params(**params_fixed)
            fit_kwargs = {}
            if sw is not None and _supports_sample_weight(clf):
                fit_kwargs["sample_weight"] = sw
            clf.fit(Xtr, ytr, **fit_kwargs)
            loss = -float(scorer(clf, X_val_s, self.splits_["y_val"]))
            return loss

        if new_source is None:
            rng = np.random.default_rng(self.random_state + 12345)
            X_blocks, y_blocks, sw = self._build_train_blocks(
                X_tasks, y_tasks, target, sources, fixed_weights, rng
            )
            X_blocks_s, X_val_s, _ = self._prep_fit_transform(
                X_blocks, self.splits_["X_val"], self.splits_["X_test"]
            )
            Xtr = np.concatenate(X_blocks_s, axis=0)
            ytr = np.concatenate(y_blocks, axis=0)
            clf = clone(self.estimator).set_params(**params_fixed)
            fit_kwargs = {}
            if sw is not None and _supports_sample_weight(clf):
                fit_kwargs["sample_weight"] = sw
            clf.fit(Xtr, ytr, **fit_kwargs)
            val_loss = -float(scorer(clf, X_val_s, self.splits_["y_val"]))
            return float(val_loss), dict(fixed_weights)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        best_w = dict(fixed_weights)
        if new_source is not None:
            best_w[new_source] = float(study.best_trial.params.get(f"w_{new_source}", self.weight_min))

        rng = np.random.default_rng(self.random_state + 999)
        X_blocks, y_blocks, sw = self._build_train_blocks(
            X_tasks, y_tasks, target, sources, best_w, rng
        )
        X_blocks_s, X_val_s, _ = self._prep_fit_transform(
            X_blocks, self.splits_["X_val"], self.splits_["X_test"]
        )
        Xtr = np.concatenate(X_blocks_s, axis=0)
        ytr = np.concatenate(y_blocks, axis=0)
        clf = clone(self.estimator).set_params(**params_fixed)
        fit_kwargs = {}
        if sw is not None and _supports_sample_weight(clf):
            fit_kwargs["sample_weight"] = sw
        clf.fit(Xtr, ytr, **fit_kwargs)
        val_loss = -float(scorer(clf, X_val_s, self.splits_["y_val"]))
        return float(val_loss), _as_builtin(best_w)

    def fit(self, X_tasks: Dict[str, np.ndarray], y_tasks: Dict[str, np.ndarray], target_task_name: str):
        self.target_ = target_task_name
        tname = target_task_name
        self._make_target_splits(X_tasks[tname], y_tasks[tname])

        base_loss, base_params = self._tune_params_on_target(clone(self.estimator))  
        self._base_params_ = dict(base_params)

        self.best_ = _Combo(tasks=[tname], val_loss=float(base_loss), params=dict(base_params), weights={})
        history_rows = []

        all_sources = [k for k in X_tasks.keys() if k != tname]
        frontier = deque()
        frontier.append((tuple(), 0, float(base_loss), {}))  

        while frontier:
            S_tuple, d, L_parent, w_fixed = frontier.popleft()
            if d == self.depth:
                continue
            S_set = set(S_tuple)
            for s in all_sources:
                if s in S_set:
                    continue
                S_child = tuple(list(S_tuple) + [s])

                combo = [tname] + list(S_child)
                L_child, w_child = self._optimize_new_source_weight(
                    combo, X_tasks, y_tasks, params_fixed=self._base_params_,
                    fixed_weights=w_fixed, new_source=s
                )

                improved = (L_child < self.best_.val_loss)
                pass_prune = (L_child < L_parent * (1.0 - self.improve_delta)) 

                history_rows.append({
                    "depth": d + 1,
                    "prev_combo": "/".join([tname] + list(S_tuple)),
                    "new_source": s,
                    "new_combo": "/".join(combo),
                    "prev_val_loss": float(L_parent),
                    "new_val_loss": float(L_child),
                    "improvement_vs_parent": float(L_parent - L_child),
                    "threshold": float(L_parent * (1.0 - self.improve_delta)),
                    "kept_for_expansion": bool(pass_prune),
                    "updated_incumbent": bool(improved),
                    "weights": _as_builtin(w_child),
                    "params": _as_builtin(self._base_params_),
                })

                if improved:
                    self.best_ = _Combo(
                        tasks=[tname] + list(S_child),
                        val_loss=float(L_child),
                        params=_as_builtin(self._base_params_),
                        weights=_as_builtin(w_child)
                    )

                if pass_prune:
                    frontier.append((S_child, d + 1, float(L_child), dict(w_child)))

        self.history_ = pd.DataFrame(history_rows)
        return self

    def evaluate_final_model(
        self,
        X_tasks: Dict[str, np.ndarray],
        y_tasks: Dict[str, np.ndarray],
        extra_metrics: Optional[Dict[str, Union[str, Callable]]] = None
    ) -> Dict:
        tname = self.target_
        Xt_tr, yt_tr = self.splits_["X_train"], self.splits_["y_train"]
        Xt_val, yt_val = self.splits_["X_val"], self.splits_["y_val"]
        Xt_te,  yt_te  = self.splits_["X_test"], self.splits_["y_test"]

        rng = np.random.default_rng(self.random_state + 123)
        sources = [t for t in self.best_.tasks[1:]]
        X_blocks, y_blocks, sw = self._build_train_blocks(
            X_tasks, y_tasks, tname, sources, {k: float(v) for k, v in self.best_.weights.items()}, rng
        )
        X_blocks_s, X_val_s, X_te_s = self._prep_fit_transform(X_blocks, Xt_val, Xt_te)
        Xtr = np.concatenate(X_blocks_s, axis=0)
        ytr = np.concatenate(y_blocks, axis=0)

        clf = clone(self.estimator).set_params(**self.best_.params)
        fit_kwargs = {}
        if sw is not None and _supports_sample_weight(clf):
            fit_kwargs["sample_weight"] = sw
        clf.fit(Xtr, ytr, **fit_kwargs)

        scorer_main = _choose_scoring(self.scoring, yt_te, clf)
        main_score = float(scorer_main(clf, X_te_s, yt_te))
        out = {
            "alvo": tname,
            "tarefas_selecionadas": list(self.best_.tasks),
            "params": _as_builtin(self.best_.params),
            "pesos_fontes": _as_builtin(self.best_.weights),
            "main_scorer": str(self.scoring) if isinstance(self.scoring, str) else "auto",
            "main_score": main_score,
            "n_test": int(len(yt_te)),
        }

        if extra_metrics:
            from sklearn.metrics import get_scorer
            extras = {}
            for name, sc in extra_metrics.items():
                s = get_scorer(sc) if isinstance(sc, str) else sc
                extras[name] = float(s(clf, X_te_s, yt_te))
            out["extra_metrics"] = _as_builtin(extras)
        return out

    def results_summary(self) -> Dict:
        df = self.history_
        base_loss = df["prev_val_loss"].iloc[0] if not df.empty else self.best_.val_loss
        best_loss = self.best_.val_loss
        gain = 100.0 * (base_loss - best_loss) / (abs(base_loss) if base_loss != 0 else 1.0)
        return _as_builtin({
            "alvo": self.target_,
            "tarefas_selecionadas": list(self.best_.tasks),
            "baseline_val_loss": float(base_loss),
            "melhor_val_loss": float(best_loss),
            "ganho_relativo_%": float(gain),
            "avaliacoes": int(len(df)),
        })

    def top_moves_table(self, k: int = 10) -> pd.DataFrame:
        if self.history_.empty:
            return pd.DataFrame(columns=[
                "depth","prev_combo","new_source","new_combo",
                "prev_val_loss","new_val_loss","improvement_vs_parent","kept_for_expansion"
            ])
        cols = ["depth","prev_combo","new_source","new_combo","prev_val_loss","new_val_loss",
                "improvement_vs_parent","kept_for_expansion"]
        return self.history_[cols].sort_values(by="improvement_vs_parent", ascending=False).head(k).reset_index(drop=True)
