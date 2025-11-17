"""Roda apenas os métodos MTL comparativos"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from u_curve_atl.selector_generic_final import UcurveATLSelectorGeneric
from u_curve_atl.comparators import (
    LogisticDirtyMTL,
    LogisticL21MTL,
    LogisticRMTFL,
)

from test_code_final import (
    build_tasks_dict,
    is_binary,
    load_info_yaml,
    _prepare_mtl_data,
    _tune_mtl_model,
    _evaluate_mtl_model,
    _mtl_space_dirty,
    _mtl_space_l21,
    _mtl_space_rmtfl,
)


def _dummy_space(trial) -> Dict:
    return {}


def _run_for_target(
    base_name: str,
    estimator: str,
    run_depth: int,
    selector: UcurveATLSelectorGeneric,
    X_tasks: Dict[str, np.ndarray],
    y_tasks: Dict[str, np.ndarray],
    target: str,
    n_trials: int,
    seed: int,
) -> List[Dict]:
    selector._make_target_splits(X_tasks[target], y_tasks[target])
    mtl_data = _prepare_mtl_data(selector, X_tasks, y_tasks, target, selector.preprocess)

    rows = []
    trials = max(10, n_trials)

    params_l21 = _tune_mtl_model(mtl_data, LogisticL21MTL, _mtl_space_l21, trials, seed + 101)
    params_l21["random_state"] = seed
    res_l21 = _evaluate_mtl_model(mtl_data, LogisticL21MTL, params_l21)
    rows.append({
        "base_name": base_name,
        "estimator": estimator,
        "run_depth": run_depth,
        "target": target,
        "method": "Logistic_L21",
        **res_l21,
        "params_json": json.dumps(params_l21),
    })

    params_dirty = _tune_mtl_model(mtl_data, LogisticDirtyMTL, _mtl_space_dirty, trials, seed + 202)
    params_dirty["random_state"] = seed
    res_dirty = _evaluate_mtl_model(mtl_data, LogisticDirtyMTL, params_dirty)
    rows.append({
        "base_name": base_name,
        "estimator": estimator,
        "run_depth": run_depth,
        "target": target,
        "method": "Logistic_Dirty",
        **res_dirty,
        "params_json": json.dumps(params_dirty),
    })

    params_rmtfl = _tune_mtl_model(mtl_data, LogisticRMTFL, _mtl_space_rmtfl, trials, seed + 303)
    params_rmtfl["random_state"] = seed
    res_rmtfl = _evaluate_mtl_model(mtl_data, LogisticRMTFL, params_rmtfl)
    rows.append({
        "base_name": base_name,
        "estimator": estimator,
        "run_depth": run_depth,
        "target": target,
        "method": "Logistic_rMTFL",
        **res_rmtfl,
        "params_json": json.dumps(params_rmtfl),
    })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Executa somente os comparadores MTL com tuning.")
    parser.add_argument("--path", type=str, required=True, help="Diretório da base (onde está info.yaml).")
    parser.add_argument("--base_name", type=str, required=True, help="Nome da base para salvar nas tabelas finais.")
    parser.add_argument("--estimator", type=str, default="logreg", help="Nome do estimador principal (apenas metadado).")
    parser.add_argument("--run_depth", type=int, default=0, help="Profundidade identificadora do run (metadado).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_trials", type=int, default=80, help="Número de trials Optuna para cada comparador.")
    parser.add_argument("--test_size", type=float, default=0.50)
    parser.add_argument("--val_size", type=float, default=0.25)
    parser.add_argument("--output", type=str, default="comparisons_mtl_only.csv")
    args = parser.parse_args()

    base_path = Path(args.path).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {base_path}")

    tasks = load_info_yaml(base_path)
    X_tasks, y_tasks = build_tasks_dict(base_path, tasks)

    selector = UcurveATLSelectorGeneric(
        estimator=LogisticRegression(),
        param_space=_dummy_space,
        scoring="neg_log_loss",
        depth=1,
        n_trials=1,
        weight_min=0.0,
        weight_max=1.0,
        preprocess=None,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        improve_delta=0.01,
    )

    all_rows: List[Dict] = []
    for target in tasks:
        if not is_binary(y_tasks[target]):
            print(f"[SKIP] {target}: não é binária, pulando comparadores.")
            continue
        rows = _run_for_target(
            args.base_name,
            args.estimator,
            args.run_depth,
            selector,
            X_tasks,
            y_tasks,
            target,
            args.n_trials,
            args.seed,
        )
        all_rows.extend(rows)
        print(f"[OK] Comparadores recalculados para {target}.")

    if not all_rows:
        print("[WARN] Nenhum alvo elegível. Nenhum arquivo gerado.")
        return

    df = pd.DataFrame(all_rows)
    out_path = Path(args.output).expanduser().resolve()
    df.to_csv(out_path, index=False)
    print(f"[DONE] Comparações salvas em {out_path}")


if __name__ == "__main__":
    main()
