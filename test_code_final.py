import argparse
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import optuna

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target

from u_curve_atl.selector_generic_final import UcurveATLSelectorGeneric
from u_curve_atl.comparators import (
    LogisticL21MTL, LogisticDirtyMTL, LogisticRMTFL
)

def load_info_yaml(base_path: Path):
    with open(base_path / "info.yaml", "r") as f:
        data = yaml.safe_load(f)
    datasets = data["datasets"]
    assert isinstance(datasets, list) and len(datasets) > 0
    return datasets

def load_task_csv(task_dir: Path):
    df = pd.read_csv(task_dir / "data.csv", header=None, index_col=None)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y_raw = df.iloc[:, -1]
    try:
        y = pd.to_numeric(y_raw, errors="raise").to_numpy()
    except Exception:
        y, _ = pd.factorize(y_raw)
        y = y.astype(int)
    return X, y

def build_tasks_dict(base_path: Path, tasks):
    X_tasks, y_tasks = {}, {}
    for t in tasks:
        X, y = load_task_csv(base_path / t)
        X_tasks[t] = X
        y_tasks[t] = y
    return X_tasks, y_tasks

def is_classification(y) -> bool:
    t = type_of_target(y)
    return t in ("binary", "multiclass", "multiclass-indicator")

def is_binary(y) -> bool:
    return is_classification(y) and np.unique(y).size == 2

def binarize01(y: np.ndarray) -> np.ndarray:
    u = np.unique(y)
    assert u.size == 2, "esperado binário"
    return (y == u.max()).astype(int)

def make_estimator_and_space(name: str):
    name = name.lower()
    if name == "logreg":
        est = LogisticRegression(max_iter=1200)
        def space(trial):
            return {
                "C": trial.suggest_float("C", 1e-5, 1e5, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l2", None]),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "max_iter": trial.suggest_int("max_iter", 800, 1200),
            }
        scoring = "auto"
        extras = {"roc_auc": "roc_auc", "average_precision": "average_precision", "neg_brier": "neg_brier_score", "neg_log_loss": "neg_log_loss"}
        preprocess = StandardScaler()
        return est, space, scoring, extras, preprocess

    if name == "rf_clf":
        est = RandomForestClassifier()
        def space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        scoring = "auto"
        extras = {"roc_auc": "roc_auc", "average_precision": "average_precision", "neg_log_loss": "neg_log_loss"}
        preprocess = None
        return est, space, scoring, extras, preprocess

    if name == "gb_clf":
        est = GradientBoostingClassifier()
        def space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }
        scoring = "auto"
        extras = {"roc_auc": "roc_auc", "average_precision": "average_precision", "neg_log_loss": "neg_log_loss"}
        preprocess = None
        return est, space, scoring, extras, preprocess

    if name == "rf_reg":
        est = RandomForestRegressor()
        def space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        scoring = "auto"
        extras = {"r2": "r2", "neg_mean_squared_error": "neg_mean_squared_error"}
        preprocess = None
        return est, space, scoring, extras, preprocess

    if name == "gb_reg":
        est = GradientBoostingRegressor()
        def space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }
        scoring = "auto"
        extras = {"r2": "r2", "neg_mean_squared_error": "neg_mean_squared_error"}
        preprocess = None
        return est, space, scoring, extras, preprocess

    raise ValueError(f"estimator desconhecido: {name}")

def _prepare_mtl_data(selector, X_tasks, y_tasks, target, preprocess):
    Xt_tr = selector.splits_["X_train"]; yt_tr = selector.splits_["y_train"]
    Xt_val = selector.splits_["X_val"];   yt_val = selector.splits_["y_val"]
    Xt_te = selector.splits_["X_test"];   yt_te = selector.splits_["y_test"]

    prep = clone(preprocess) if preprocess is not None else StandardScaler()
    prep.fit(Xt_tr)
    Xt_tr_s = prep.transform(Xt_tr)
    Xt_val_s = prep.transform(Xt_val)
    Xt_te_s = prep.transform(Xt_te)

    def _bin01(y):
        u = np.unique(y)
        if u.size == 2:
            return (y == u.max()).astype(int)
        return y

    order = [target] + [k for k in X_tasks.keys() if k != target]
    X_train_tasks = {target: Xt_tr_s}
    y_train_tasks = {target: _bin01(yt_tr)}
    for t in order[1:]:
        X_train_tasks[t] = prep.transform(X_tasks[t])
        y_train_tasks[t] = _bin01(y_tasks[t])

    return {
        "prep": prep,
        "X_train_tasks": X_train_tasks,
        "y_train_tasks": y_train_tasks,
        "task_order": order,
        "X_val": Xt_val_s,
        "y_val": _bin01(yt_val),
        "X_test": Xt_te_s,
        "y_test": _bin01(yt_te),
    }


def _tune_mtl_model(mtl_data, model_ctor, param_space, n_trials, seed):
    sampler = optuna.samplers.TPESampler(seed=seed)

    def objective(trial: optuna.Trial):
        params = param_space(trial)
        trial.set_user_attr("params_dict", params)
        mdl = model_ctor(**params)
        mdl.fit(mtl_data["X_train_tasks"], mtl_data["y_train_tasks"], mtl_data["task_order"])
        proba = mdl.predict_proba(mtl_data["X_val"], task_index=0)[:, 1]
        return log_loss(mtl_data["y_val"], proba, labels=[0, 1])

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_trial.user_attrs["params_dict"])


def _evaluate_mtl_model(mtl_data, model_ctor, params):
    mdl = model_ctor(**params)
    mdl.fit(mtl_data["X_train_tasks"], mtl_data["y_train_tasks"], mtl_data["task_order"])
    proba = mdl.predict_proba(mtl_data["X_test"], task_index=0)[:, 1]
    ytest = mtl_data["y_test"]
    return {
        "logloss": log_loss(ytest, proba, labels=[0, 1]),
        "auc": roc_auc_score(ytest, proba),
        "ap": average_precision_score(ytest, proba),
        "brier": brier_score_loss(ytest, proba),
    }


def _mtl_space_l21(trial: optuna.Trial):
    return {
        "lam": trial.suggest_float("lam", 1e-3, 1e3, log=True),
        "max_iter": trial.suggest_int("max_iter", 400, 1500),
        "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
        "standardize": False,
    }


def _mtl_space_dirty(trial: optuna.Trial):
    return {
        "lam1": trial.suggest_float("lam1", 1e-3, 1e3, log=True),
        "lam2": trial.suggest_float("lam2", 1e-4, 1e1, log=True),
        "max_iter": trial.suggest_int("max_iter", 400, 1500),
        "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
        "standardize": False,
    }


def _mtl_space_rmtfl(trial: optuna.Trial):
    tau = trial.suggest_float("tau", 1e-3, 1e2, log=True)
    tau_factor = trial.suggest_float("tau_init_factor", 2.0, 15.0)
    return {
        "tau": tau,
        "lam": trial.suggest_float("lam", 1e-3, 5.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 600, 1800),
        "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
        "tau_init": tau * tau_factor,
        "cont_factor": trial.suggest_float("cont_factor", 0.4, 0.9),
        "standardize": False,
    }

def _run_one_setting(tasks, X_tasks, y_tasks, est, param_space, scoring, extras, preprocess,
                     depth, n_trials, weight_min, weight_max, test_size, val_size, seed, improve_delta):
    rows = []
    for target in tasks:
        selector = UcurveATLSelectorGeneric(
            estimator=est, param_space=param_space, scoring=scoring,
            depth=depth, n_trials=n_trials,
            weight_min=weight_min, weight_max=weight_max,
            preprocess=preprocess, test_size=test_size, val_size=val_size,
            random_state=seed, improve_delta=improve_delta
        )
        selector.fit(X_tasks, y_tasks, target_task_name=target)
        out = selector.evaluate_final_model(X_tasks, y_tasks, extra_metrics=extras)
        extras_o = out.get("extra_metrics", {})
        main_scorer = str(out["main_scorer"])
        main_score  = float(out["main_score"])
        auc   = float(extras_o.get("roc_auc", np.nan)) if "roc_auc" in extras_o else np.nan
        ap    = float(extras_o.get("average_precision", np.nan)) if "average_precision" in extras_o else np.nan
        brier = (-float(extras_o["neg_brier"])) if "neg_brier" in extras_o else np.nan
        if main_scorer == "neg_log_loss":
            logloss = -main_score
        elif "neg_log_loss" in extras_o:
            logloss = -float(extras_o["neg_log_loss"])
        else:
            logloss = np.nan
        rows.append({"target": target, "auc": auc, "ap": ap, "brier": brier, "logloss": logloss})
    df = pd.DataFrame(rows)
    macro = {
        "auc": float(np.nanmean(df["auc"])),
        "ap": float(np.nanmean(df["ap"])),
        "brier": float(np.nanmean(df["brier"])),
        "logloss": float(np.nanmean(df["logloss"])),
        "n_targets": int(len(df))
    }
    return df, macro

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--estimator", type=str, default="logreg",
                        choices=["logreg", "rf_clf", "gb_clf", "rf_reg", "gb_reg"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=120)  
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_min", type=float, default=0.0)
    parser.add_argument("--weight_max", type=float, default=1.0)
    parser.add_argument("--test_size", type=float, default=0.50) 
    parser.add_argument("--val_size", type=float, default=0.25)  
    parser.add_argument("--scoring", type=str, default="auto")   
    parser.add_argument("--improve_delta", type=float, default=0.01)
    parser.add_argument("--compare_mtl", action="store_true")
    parser.add_argument("--mtl_folds", type=int, default=5)
    parser.add_argument("--mtl_repeats", type=int, default=2)
    parser.add_argument("--abl_depths", type=str, default="", help="ex: 2,3,4,5")
    parser.add_argument("--abl_eps", type=str, default="", help="ex: 0.005,0.01,0.02,0.05")
    args = parser.parse_args()

    base_path = Path(args.path)
    tasks = load_info_yaml(base_path)
    X_tasks, y_tasks = build_tasks_dict(base_path, tasks)

    est, param_space, default_scoring, extra_metrics, preprocess = make_estimator_and_space(args.estimator)
    scoring = args.scoring or default_scoring

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/depth_eps_results", exist_ok=True) 

    def _parse_list(s, typ=float):
        return [] if not s else [typ(x.strip()) for x in s.split(",") if x.strip()]

    depth_list = _parse_list(args.abl_depths, int)
    eps_list   = _parse_list(args.abl_eps, float)

    if depth_list:
        rows = []
        for d in depth_list:
            _, macro = _run_one_setting(
                tasks, X_tasks, y_tasks, est, param_space, scoring, extra_metrics, preprocess,
                depth=d, n_trials=args.n_trials,
                weight_min=args.weight_min, weight_max=args.weight_max,
                test_size=args.test_size, val_size=args.val_size,
                seed=args.seed, improve_delta=args.improve_delta
            )
            rows.append({"depth": d, **macro})
        dfD = pd.DataFrame(rows).sort_values("depth")
        dfD.to_csv("results/ablation/ablation_depth.csv", index=False)
        plt.figure(); plt.plot(dfD["depth"], dfD["auc"], marker="o"); plt.xlabel("depth"); plt.ylabel("AUC"); plt.tight_layout()
        plt.savefig("results/ablation/abl_depth_auc.png", dpi=200); plt.close()
        plt.figure(); plt.plot(dfD["depth"], dfD["logloss"], marker="o"); plt.xlabel("depth"); plt.ylabel("log-loss"); plt.tight_layout()
        plt.savefig("results/ablation/abl_depth_logloss.png", dpi=200); plt.close()

    if eps_list:
        rows = []
        for e in eps_list:
            _, macro = _run_one_setting(
                tasks, X_tasks, y_tasks, est, param_space, scoring, extra_metrics, preprocess,
                depth=args.depth, n_trials=args.n_trials,
                weight_min=args.weight_min, weight_max=args.weight_max,
                test_size=args.test_size, val_size=args.val_size,
                seed=args.seed, improve_delta=e
            )
            rows.append({"epsilon": e, **macro})
        dfE = pd.DataFrame(rows).sort_values("epsilon")
        dfE.to_csv("results/ablation/ablation_eps.csv", index=False)
        plt.figure(); plt.plot(dfE["epsilon"], dfE["auc"], marker="o"); plt.xlabel("epsilon"); plt.ylabel("AUC"); plt.tight_layout()
        plt.savefig("results/ablation/abl_eps_auc.png", dpi=200); plt.close()
        plt.figure(); plt.plot(dfE["epsilon"], dfE["logloss"], marker="o"); plt.xlabel("epsilon"); plt.ylabel("log-loss"); plt.tight_layout()
        plt.savefig("results/ablation/abl_eps_logloss.png", dpi=200); plt.close()

    summaries = []
    all_hist = []
    comp_rows = []

    for target in tasks:
        print(f"\n================ TARGET: {target} ================\n")
        selector = UcurveATLSelectorGeneric(
            estimator=est,
            param_space=param_space,
            scoring=scoring,
            depth=args.depth,
            n_trials=args.n_trials,
            weight_min=args.weight_min,
            weight_max=args.weight_max,
            preprocess=preprocess,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.seed,
            improve_delta=args.improve_delta
        )
        selector.fit(X_tasks, y_tasks, target_task_name=target)
        summary_val = selector.results_summary()

        test_out = selector.evaluate_final_model(
            X_tasks, y_tasks, extra_metrics=extra_metrics
        )

        extras = test_out.get("extra_metrics", {})
        main_scorer = str(test_out["main_scorer"])
        main_score = float(test_out["main_score"])
        test_auc = float(extras.get("roc_auc", np.nan)) if "roc_auc" in extras else np.nan
        test_ap = float(extras.get("average_precision", np.nan)) if "average_precision" in extras else np.nan
        test_brier = (-float(extras["neg_brier"])) if "neg_brier" in extras else np.nan
        if main_scorer == "neg_log_loss":
            test_log_loss = -main_score
        elif "neg_log_loss" in extras:
            test_log_loss = -float(extras["neg_log_loss"])
        else:
            test_log_loss = np.nan
        test_mse = -float(extras["neg_mean_squared_error"]) if "neg_mean_squared_error" in extras else np.nan
        test_r2 = float(extras["r2"]) if "r2" in extras else np.nan

        summaries.append({
            "target": target,
            "best_tasks": "|".join(selector.best_.tasks if selector.best_ else [target]),
            "baseline_val": summary_val["baseline_val_loss"],
            "best_val": summary_val["melhor_val_loss"],
            "gain_val_pct": summary_val["ganho_relativo_%"],
            "n_val_evals": summary_val["avaliacoes"],
            "test_main_scorer": main_scorer,
            "test_main_score": main_score,
            "test_log_loss": test_log_loss,
            "test_auc": test_auc,
            "test_ap": test_ap,
            "test_brier": test_brier,
            "test_mse": test_mse,
            "test_r2": test_r2,
            "n_test": int(test_out["n_test"]),
            "best_params_json": json.dumps(selector.best_.params if selector.best_ else {})
        })

        hist = selector.history_.copy()
        if not hist.empty:
            hist.insert(0, "target", target)
            all_hist.append(hist)

        print("Melhor combinação:", selector.best_.tasks if selector.best_ else [target])
        print(f"Val baseline: {summary_val['baseline_val_loss']:.6f}")
        print(f"Val melhor:   {summary_val['melhor_val_loss']:.6f}")
        print(f"Ganho (%):    {summary_val['ganho_relativo_%']:.4f}")
        if not np.isnan(test_auc): print(f"Teste AUC:    {test_auc:.6f}")
        if not np.isnan(test_log_loss): print(f"Teste LogLoss:{test_log_loss:.6f}")
        if not np.isnan(test_r2): print(f"Teste R2:     {test_r2:.6f}")

        yt = y_tasks[target]
        if args.compare_mtl and is_binary(yt):
            mtl_data = _prepare_mtl_data(selector, X_tasks, y_tasks, target, preprocess)
            mtl_trials = max(10, args.n_trials)

            params_l21 = _tune_mtl_model(mtl_data, LogisticL21MTL, _mtl_space_l21, mtl_trials, args.seed + 101)
            params_l21["random_state"] = args.seed
            res_l21 = _evaluate_mtl_model(mtl_data, LogisticL21MTL, params_l21)

            params_dirty = _tune_mtl_model(mtl_data, LogisticDirtyMTL, _mtl_space_dirty, mtl_trials, args.seed + 202)
            params_dirty["random_state"] = args.seed
            res_dirty = _evaluate_mtl_model(mtl_data, LogisticDirtyMTL, params_dirty)

            params_rmtfl = _tune_mtl_model(mtl_data, LogisticRMTFL, _mtl_space_rmtfl, mtl_trials, args.seed + 303)
            params_rmtfl["random_state"] = args.seed
            res_rmtfl = _evaluate_mtl_model(mtl_data, LogisticRMTFL, params_rmtfl)

            comp_rows.extend([
                {"target": target, "method": "ATL_selector", "logloss": test_log_loss, "auc": test_auc, "ap": test_ap, "brier": test_brier},
                {"target": target, "method": "Logistic_L21", "logloss": res_l21["logloss"], "auc": res_l21["auc"], "ap": res_l21["ap"], "brier": res_l21["brier"]},
                {"target": target, "method": "Logistic_Dirty", "logloss": res_dirty["logloss"], "auc": res_dirty["auc"], "ap": res_dirty["ap"], "brier": res_dirty["brier"]},
                {"target": target, "method": "Logistic_rMTFL", "logloss": res_rmtfl["logloss"], "auc": res_rmtfl["auc"], "ap": res_rmtfl["ap"], "brier": res_rmtfl["brier"]},
            ])

            methods = ["ATL_selector", "Logistic_L21", "Logistic_Dirty", "Logistic_rMTFL"]
            aucs = [test_auc, res_l21["auc"], res_dirty["auc"], res_rmtfl["auc"]]
            lls  = [test_log_loss, res_l21["logloss"], res_dirty["logloss"], res_rmtfl["logloss"]]

            plt.figure(figsize=(6,4))
            plt.bar(methods, aucs)
            plt.ylabel("AUC"); plt.title(f"AUC teste - {target}")
            plt.xticks(rotation=15); plt.tight_layout()
            plt.savefig(f"results/compare_auc_{target}.png", dpi=200); plt.close()

            plt.figure(figsize=(6,4))
            plt.bar(methods, lls)
            plt.ylabel("LogLoss"); plt.title(f"LogLoss teste - {target}")
            plt.xticks(rotation=15); plt.tight_layout()
            plt.savefig(f"results/compare_logloss_{target}.png", dpi=200); plt.close()

    if summaries:
        pd.DataFrame(summaries).to_csv("summary_generic_atl.csv", index=False)
        print("\n[LOG] Resumo salvo em 'summary_generic_atl.csv'.")
    if all_hist:
        pd.concat(all_hist, ignore_index=True).to_csv("history_generic_atl.csv", index=False)
        print("[LOG] Histórico salvo em 'history_generic_atl.csv'.")
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv("comparisons_mtl.csv", index=False)
        print("[LOG] Comparações salvas em 'comparisons_mtl.csv'.")

if __name__ == "__main__":
    main()
