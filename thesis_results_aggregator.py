
"""
Thesis Results Aggregator
- Lê e agrega resultados das bases (9 no total).
- Para bases com grupos reais, calcula métricas de comparação com clusters previstos.

Arquivos esperados dentro de cada "dir":
  - comparisons_mtl.csv
  - history_generic_atl.csv
  - summary_generic_atl.csv
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score
    from sklearn.metrics import confusion_matrix, accuracy_score
    from scipy.optimize import linear_sum_assignment
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


REAL_BASES = {"spam_a", "spam_b", "landmine"}


def _split_run_path(path: Path, runs_dir: Path) -> Optional[Tuple[str, str, str]]:
    try:
        rel = path.relative_to(runs_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 3:
        return None
    base_name, estimator, depth_dir = parts[0], parts[1], parts[2]
    run_depth = depth_dir.replace("depth_", "")
    return base_name, estimator, run_depth


def _count_best_members(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    tokens = [t.strip() for t in str(value).split("|") if t and t.strip().lower() != "nan"]
    return len(tokens)


def _ensure_numeric(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def compute_runs_reports(df_summ: pd.DataFrame, outdir: Path) -> None:
    if df_summ.empty:
        return

    df = df_summ.copy()
    numeric_cols = [
        "baseline_val",
        "best_val",
        "gain_val_pct",
        "test_log_loss",
        "test_auc",
        "test_ap",
        "test_brier",
        "test_mse",
        "test_r2",
        "test_main_score",
    ]
    _ensure_numeric(df, numeric_cols)

    if {"baseline_val", "best_val"}.issubset(df.columns):
        df["val_improvement"] = df["baseline_val"] - df["best_val"]
    else:
        df["val_improvement"] = np.nan

    if "gain_val_pct" not in df.columns:
        df["gain_val_pct"] = np.nan

    if "best_tasks" in df.columns:
        df["best_combo_size"] = df["best_tasks"].apply(_count_best_members)
    else:
        df["best_combo_size"] = 0

    df["improved"] = np.where(df["val_improvement"].notna() & (df["val_improvement"] > 0), 1, 0)
    df["category"] = df["base_name"].map(lambda b: "real" if b in REAL_BASES else "synthetic")

    base_group = df.groupby("base_name", dropna=False)
    base_summary = pd.DataFrame({
        "category": base_group["category"].first(),
        "estimators": base_group["estimator"].apply(lambda s: ",".join(sorted(set(s)))),
        "depths": base_group["run_depth"].apply(lambda s: ",".join(sorted(str(v) for v in set(s)))),
        "n_targets": base_group["target"].nunique() if "target" in df.columns else base_group.size(),
        "sum_best_combo_size": base_group["best_combo_size"].sum(),
        "sum_val_improvement": base_group["val_improvement"].sum(),
        "sum_gain_val_pct": base_group["gain_val_pct"].sum(),
        "count_improved_targets": base_group["improved"].sum(),
    })

    if "baseline_val" in df.columns:
        base_summary["mean_baseline_val"] = base_group["baseline_val"].mean()
    if "best_val" in df.columns:
        base_summary["mean_best_val"] = base_group["best_val"].mean()
    if "gain_val_pct" in df.columns:
        base_summary["mean_gain_val_pct"] = base_group["gain_val_pct"].mean()
    if "val_improvement" in df.columns:
        base_summary["mean_val_improvement"] = base_group["val_improvement"].mean()
    for col in ["test_auc", "test_log_loss", "test_ap", "test_brier", "test_mse", "test_r2", "test_main_score"]:
        if col in df.columns:
            base_summary[f"mean_{col}"] = base_group[col].mean()

    base_summary.sort_index(inplace=True)
    base_summary.to_csv(outdir / "runs_new_summary_by_base.csv")

    category_group = df.groupby("category", dropna=False)
    cat_summary = pd.DataFrame({
        "n_bases": category_group["base_name"].nunique(),
        "n_targets": category_group["target"].count() if "target" in df.columns else category_group.size(),
        "sum_best_combo_size": category_group["best_combo_size"].sum(),
        "sum_val_improvement": category_group["val_improvement"].sum(),
        "sum_gain_val_pct": category_group["gain_val_pct"].sum(),
        "count_improved_targets": category_group["improved"].sum(),
    })

    if "baseline_val" in df.columns:
        cat_summary["mean_baseline_val"] = category_group["baseline_val"].mean()
    if "best_val" in df.columns:
        cat_summary["mean_best_val"] = category_group["best_val"].mean()
    if "gain_val_pct" in df.columns:
        cat_summary["mean_gain_val_pct"] = category_group["gain_val_pct"].mean()
    if "val_improvement" in df.columns:
        cat_summary["mean_val_improvement"] = category_group["val_improvement"].mean()
    for col in ["test_auc", "test_log_loss", "test_ap", "test_brier", "test_mse", "test_r2", "test_main_score"]:
        if col in df.columns:
            cat_summary[f"mean_{col}"] = category_group[col].mean()

    cat_summary["avg_best_combo_size"] = cat_summary.apply(
        lambda row: row["sum_best_combo_size"] / row["n_targets"] if row["n_targets"] else np.nan,
        axis=1,
    )

    cat_summary.sort_index(inplace=True)
    cat_summary.to_csv(outdir / "runs_new_summary_by_category.csv")

    overall = {
        "n_bases": df["base_name"].nunique(),
        "n_targets": len(df) if "target" in df.columns else len(df),
        "sum_best_combo_size": df["best_combo_size"].sum(),
        "sum_val_improvement": df["val_improvement"].sum(),
        "sum_gain_val_pct": df["gain_val_pct"].sum(),
        "count_improved_targets": df["improved"].sum(),
    }
    for col in ["baseline_val", "best_val", "gain_val_pct", "val_improvement", "test_auc", "test_log_loss", "test_ap", "test_brier", "test_mse", "test_r2", "test_main_score"]:
        if col in df.columns:
            overall[f"mean_{col}"] = df[col].mean()

    pd.DataFrame([overall]).to_csv(outdir / "runs_new_summary_overall.csv", index=False)

    report_lines = [
        "Visão geral\n",
        f"Bases distintas: {overall['n_bases']}\n",
        f"Targets analisados: {overall['n_targets']}\n",
        f"Somatório ganho pct: {overall.get('sum_gain_val_pct', 0):.4f}\n",
        f"Somatório melhoria val: {overall.get('sum_val_improvement', 0):.4f}\n",
        f"Targets com melhoria: {overall.get('count_improved_targets', 0)}\n",
        "\n# Por categoria\n",
    ]
    for category, row in cat_summary.reset_index().iterrows():
        name = row["category"]
        report_lines.append(f"- {name}: bases={row['n_bases']}, targets={row['n_targets']}, ganho_sum={row['sum_gain_val_pct']:.4f}, val_improv_sum={row['sum_val_improvement']:.4f}\n")

    with open(outdir / "runs_new_report.txt", "w", encoding="utf-8") as fp:
        fp.writelines(report_lines)


def aggregate_runs_dir(runs_dir: Path, outdir: Path) -> None:
    runs_dir = runs_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summary_frames: List[pd.DataFrame] = []
    comp_frames: List[pd.DataFrame] = []
    hist_frames: List[pd.DataFrame] = []

    for summary_path in runs_dir.rglob("summary_generic_atl.csv"):
        split = _split_run_path(summary_path, runs_dir)
        if split is None:
            continue
        base_name, estimator, run_depth = split
        summ = _read_csv_safely(summary_path, expected=False)
        if summ is None or summ.empty:
            continue
        summ = summ.copy()
        summ.insert(0, "run_depth", run_depth)
        summ.insert(0, "estimator", estimator)
        summ.insert(0, "base_name", base_name)
        summary_frames.append(summ)

        base_dir = summary_path.parent
        comp = _read_csv_safely(base_dir / "comparisons_mtl.csv", expected=False)
        if comp is not None and len(comp):
            comp = comp.copy()
            comp.insert(0, "run_depth", run_depth)
            comp.insert(0, "estimator", estimator)
            comp.insert(0, "base_name", base_name)
            comp_frames.append(comp)

        hist = _read_csv_safely(base_dir / "history_generic_atl.csv", expected=False)
        if hist is not None and len(hist):
            hist = hist.copy()
            hist.insert(0, "run_depth", run_depth)
            hist.insert(0, "estimator", estimator)
            hist.insert(0, "base_name", base_name)
            hist_frames.append(hist)

    df_summ = concat_nonempty(summary_frames)
    df_comp = concat_nonempty(comp_frames)
    df_hist = concat_nonempty(hist_frames)

    if len(df_summ):
        df_summ.to_csv(outdir / "combined_summary.csv", index=False)
    if len(df_comp):
        df_comp.to_csv(outdir / "combined_comparisons.csv", index=False)
    if len(df_hist):
        df_hist.to_csv(outdir / "combined_history.csv", index=False)

    compute_runs_reports(df_summ, outdir)
    print(f"[OK] Agregação automática concluída. Saídas em: {outdir.resolve()}")



def _read_csv_safely(path: Path, expected: bool = True) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Falha ao ler {path}: {e}")
            return None
    else:
        if expected:
            print(f"[WARN] Arquivo ausente: {path}")
        return None


def load_base_triplet(base: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    base_dir = Path(base["dir"]).expanduser()
    comp = _read_csv_safely(base_dir / "comparisons_mtl.csv")
    hist = _read_csv_safely(base_dir / "history_generic_atl.csv")
    summ = _read_csv_safely(base_dir / "summary_generic_atl.csv")
    return comp, hist, summ


def add_base_name(df: Optional[pd.DataFrame], base_name: str) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    df = df.copy()
    df.insert(0, "base_name", base_name)
    return df


def concat_nonempty(frames: List[Optional[pd.DataFrame]]) -> pd.DataFrame:
    valid = [f for f in frames if f is not None and len(f) > 0]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def evaluate_clustering(gt_true: pd.Series, gt_pred: pd.Series) -> Dict[str, float]:
    res = {}
    if not _HAVE_SK:
        print("[WARN] scikit-learn/scipy não disponível. Pulando métricas de cluster.")
        return {"n": float(len(gt_true))}

    def to_codes(s: pd.Series) -> np.ndarray:
        return pd.Categorical(s.astype(str)).codes

    y_true = to_codes(gt_true)
    y_pred = to_codes(gt_pred)

    res["n"] = float(len(y_true))
    res["ari"] = float(adjusted_rand_score(y_true, y_pred))
    res["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
    res["homogeneity"] = float(homogeneity_score(y_true, y_pred))
    res["completeness"] = float(completeness_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    cost = cm.max() - cm 
    ri, ci = linear_sum_assignment(cost)
    acc = cm[ri, ci].sum() / cm.sum()
    res["perm_acc"] = float(acc)
    return res


def try_eval_ground_truth(base: Dict) -> Optional[Dict[str, float]]:
    if "gt_true_csv" not in base or "gt_pred_csv" not in base:
        return None
    tpath = Path(base["gt_true_csv"]).expanduser()
    ppath = Path(base["gt_pred_csv"]).expanduser()
    tdf = _read_csv_safely(tpath, expected=False)
    pdf = _read_csv_safely(ppath, expected=False)
    if tdf is None or pdf is None:
        return None

    def detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
        cols = [c.lower() for c in df.columns]
        id_candidates = ["id", "sample_id", "idx", "uid"]
        id_col = None
        for c in id_candidates:
            if c in cols:
                id_col = df.columns[cols.index(c)]
                break
        if id_col is None:
            id_col = df.columns[0]

        lab_candidates = ["label", "y", "group", "cluster", "class", "true", "target"]
        lab_col = None
        for c in lab_candidates:
            if c in cols:
                lab_col = df.columns[cols.index(c)]
                break
        if lab_col is None:
            lab_col = df.columns[min(1, len(df.columns)-1)]
        return id_col, lab_col

    t_id, t_lab = detect_cols(tdf)
    p_id, p_lab = detect_cols(pdf)

    merged = pd.merge(
        tdf[[t_id, t_lab]].rename(columns={t_id: "id", t_lab: "true"}),
        pdf[[p_id, p_lab]].rename(columns={p_id: "id", p_lab: "pred"}),
        on="id",
        how="inner",
        validate="one_to_one"
    )
    if merged.empty:
        print(f"[WARN] Merge vazio para base {base.get('name', base.get('dir'))}. Verifique colunas id/labels.")
        return None

    metrics = evaluate_clustering(merged["true"], merged["pred"])
    metrics["base_name"] = base.get("name", str(base.get("dir")))
    metrics["pairs_merged"] = float(len(merged))
    return metrics


def aggregate_from_config(bases: List[Dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    all_comp, all_hist, all_summ = [], [], []
    gt_rows = []

    for base in bases:
        name = base.get("name", str(base.get("dir")))
        print(f"[INFO] Lendo base: {name}")
        comp, hist, summ = load_base_triplet(base)

        comp = add_base_name(comp, name)
        hist = add_base_name(hist, name)
        summ = add_base_name(summ, name)

        all_comp.append(comp)
        all_hist.append(hist)
        all_summ.append(summ)

        gt = try_eval_ground_truth(base)
        if gt is not None:
            gt_rows.append(gt)

    df_comp = concat_nonempty(all_comp)
    df_hist = concat_nonempty(all_hist)
    df_summ = concat_nonempty(all_summ)
    df_gt = pd.DataFrame(gt_rows) if gt_rows else pd.DataFrame()

    comp_out = outdir / "combined_comparisons.csv"
    hist_out = outdir / "combined_history.csv"
    summ_out = outdir / "combined_summary.csv"
    gt_out = outdir / "ground_truth_eval.csv"

    if len(df_comp):
        df_comp.to_csv(comp_out, index=False)
    if len(df_hist):
        df_hist.to_csv(hist_out, index=False)
    if len(df_summ):
        df_summ.to_csv(summ_out, index=False)
    if len(df_gt):
        df_gt.to_csv(gt_out, index=False)

    lines = []
    lines.append("# Resumo\n")
    lines.append(f"Bases lidas: {len(bases)}\n")
    lines.append(f"comparisons: {len(df_comp)} linhas\n")
    lines.append(f"history: {len(df_hist)} linhas\n")
    lines.append(f"summary: {len(df_summ)} linhas\n")
    
    with open(outdir / "report.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)

    try:
        import matplotlib.pyplot as plt

        if len(df_summ):
            metric_candidates = [c for c in df_summ.columns if c.lower() in ("best_val", "score", "best_score", "val_score")]
            if metric_candidates:
                mcol = metric_candidates[0]
                plt.figure()
                (df_summ.groupby("base_name")[mcol].mean().sort_values(ascending=False)).plot(kind="bar")
                plt.ylabel(mcol)
                plt.title(f"Média de {mcol} por base")
                plt.tight_layout()
                plt.savefig(outdir / f"summary_{mcol}_by_base.png", dpi=200)
                plt.close()

        if len(df_gt):
            for m in ["ari", "nmi", "homogeneity", "completeness", "perm_acc"]:
                if m in df_gt.columns:
                    plt.figure()
                    df_gt.set_index("base_name")[m].sort_values(ascending=False).plot(kind="bar")
                    plt.ylabel(m)
                    plt.title(f"{m} por base (GT)")
                    plt.tight_layout()
                    plt.savefig(outdir / f"gt_{m}_by_base.png", dpi=200)
                    plt.close()
    except Exception as e:
        print(f"[WARN] Falha ao gerar figuras: {e}")

    print(f"[OK] Concluído. Saídas em: {outdir.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="Agregador de resultados da tese")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--bases", help="Caminho para bases.json")
    group.add_argument("--runs-dir", help="Diretório raiz com execuções (ex: runs_new)")
    ap.add_argument("--outdir", help="Diretório de saída (default depende do modo)")
    args = ap.parse_args()

    if args.bases:
        outdir = Path(args.outdir or "thesis_results")
        with open(args.bases, "r", encoding="utf-8") as f:
            bases: List[Dict] = json.load(f)
        aggregate_from_config(bases, outdir)
    else:
        runs_dir = Path(args.runs_dir)
        outdir = Path(args.outdir) if args.outdir else (runs_dir.parent / f"{runs_dir.name}_thesis_results")
        aggregate_runs_dir(runs_dir, outdir)


if __name__ == "__main__":
    main()
