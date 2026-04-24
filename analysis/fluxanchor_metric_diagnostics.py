#!/usr/bin/env python3
"""Diagnose fair evaluation settings for classic mutant flux prediction."""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ORGANISM = "iECDH1ME8569_1439"
CLASSIC_GENES = ["Zwf", "Pgi", "Crp"]
METHODS = ["FluxAnchor", "KinLLM", "fba"]
THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
LOG10_EPS = 1e-6

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
detailed_csv_path = os.path.join(project_root, "results_extend_reactions", ORGANISM, "detailed.csv")
output_dir = os.path.join(project_root, "analysis", f"scatter_{ORGANISM}")
os.makedirs(output_dir, exist_ok=True)

summary_csv = os.path.join(output_dir, "fluxanchor_metric_diagnostics.csv")
outlier_csv = os.path.join(output_dir, "fluxanchor_outlier_reactions.csv")
report_md = os.path.join(output_dir, "fluxanchor_metric_diagnostics.md")


def load_long_table():
    df = pd.read_csv(detailed_csv_path)
    reaction_names = sorted({
        col.replace("_true", "").replace("_pred", "")
        for col in df.columns
        if col.endswith("_true") or col.endswith("_pred")
    })

    rows = []
    for _, row in df.iterrows():
        method = row["Method"]
        gene = row["Gene"]
        for reaction in reaction_names:
            true_col = f"{reaction}_true"
            pred_col = f"{reaction}_pred"
            if true_col not in df.columns or pred_col not in df.columns:
                continue

            true_val = row[true_col]
            pred_val = row[pred_col]
            if pd.notna(true_val) and pd.notna(pred_val):
                rows.append({
                    "Method": method,
                    "Gene": gene,
                    "Reaction": reaction,
                    "true": abs(float(true_val)),
                    "pred": abs(float(pred_val)),
                })

    long_df = pd.DataFrame(rows)
    long_df["is_classic"] = long_df["Gene"].isin(CLASSIC_GENES)
    return df, long_df


def compute_metrics(true_values, pred_values):
    true_values = np.asarray(true_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    metrics = {
        "n": len(true_values),
        "raw_r2": r2_score(true_values, pred_values),
        "raw_rmse": np.sqrt(mean_squared_error(true_values, pred_values)),
        "log1p_r2": r2_score(np.log1p(true_values), np.log1p(pred_values)),
        "log1p_rmse": np.sqrt(mean_squared_error(np.log1p(true_values), np.log1p(pred_values))),
        "log10eps_r2": r2_score(np.log10(true_values + LOG10_EPS), np.log10(pred_values + LOG10_EPS)),
        "log10eps_rmse": np.sqrt(mean_squared_error(np.log10(true_values + LOG10_EPS), np.log10(pred_values + LOG10_EPS))),
        "zero_true_fraction": float(np.mean(true_values == 0)),
        "zero_pred_fraction": float(np.mean(pred_values == 0)),
        "lt_0p1_fraction": float(np.mean(true_values < 0.1)),
        "lt_0p5_fraction": float(np.mean(true_values < 0.5)),
    }
    return metrics


def evaluate_slice(df_slice, label, method, threshold):
    filtered = df_slice[df_slice["true"] >= threshold].copy()
    if len(filtered) < 2:
        return None

    metrics = compute_metrics(filtered["true"], filtered["pred"])
    metrics.update({
        "scope": label,
        "Method": method,
        "threshold": threshold,
    })
    return metrics


def fit_linear_calibration(train_df, test_df):
    if len(train_df) < 2 or len(test_df) < 2:
        return None

    model = LinearRegression().fit(train_df[["pred"]], train_df["true"])
    calibrated_pred = model.predict(test_df[["pred"]])
    calibrated_pred = np.clip(calibrated_pred, 0, None)

    metrics = compute_metrics(test_df["true"], calibrated_pred)
    metrics.update({
        "calibration_slope": float(model.coef_[0]),
        "calibration_intercept": float(model.intercept_),
    })
    return metrics


def collect_fluxanchor_outliers(long_df):
    classic_fluxanchor = long_df[(long_df["is_classic"]) & (long_df["Method"] == "FluxAnchor")].copy()
    classic_fluxanchor["abs_error"] = np.abs(classic_fluxanchor["pred"] - classic_fluxanchor["true"])
    classic_fluxanchor["sq_error"] = (classic_fluxanchor["pred"] - classic_fluxanchor["true"]) ** 2
    classic_fluxanchor["log1p_abs_error"] = np.abs(np.log1p(classic_fluxanchor["pred"]) - np.log1p(classic_fluxanchor["true"]))

    top_abs = classic_fluxanchor.sort_values(["Gene", "sq_error"], ascending=[True, False]).groupby("Gene").head(12)
    return top_abs[["Gene", "Reaction", "true", "pred", "abs_error", "sq_error", "log1p_abs_error"]]


def df_to_simple_markdown(df):
    if df.empty:
        return "(no rows)"

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def build_report(summary_df, calibration_df, outlier_df):
    lines = []
    lines.append("# FluxAnchor metric diagnostics\n")
    lines.append(f"Data file: `{detailed_csv_path}`\n")
    lines.append(f"Classic genes: {', '.join(CLASSIC_GENES)}\n")

    agg = summary_df[(summary_df["scope"] == "all_classic") & (summary_df["threshold"] == 0.0)]
    lines.append("## Aggregate classic-mutant baseline\n")
    lines.append(df_to_simple_markdown(agg[["Method", "raw_r2", "log1p_r2", "log10eps_r2", "raw_rmse", "log1p_rmse", "zero_true_fraction", "lt_0p1_fraction", "lt_0p5_fraction"]]))
    lines.append("\n")

    if not calibration_df.empty:
        lines.append("## Calibration trained on non-classic mutants, tested on classic mutants\n")
        lines.append(df_to_simple_markdown(calibration_df[["scope", "Method", "raw_r2", "log1p_r2", "raw_rmse", "log1p_rmse", "calibration_slope", "calibration_intercept"]]))
        lines.append("\n")

    lines.append("## Baseline threshold-free metric per classic gene\n")
    best_rows = []
    for gene in CLASSIC_GENES:
        gene_scope = f"classic_{gene}"
        subset = summary_df[(summary_df["scope"] == gene_scope) & (summary_df["threshold"] == 0.0)]
        if subset.empty:
            continue
        best_rows.append(subset[["Method", "raw_r2", "log1p_r2", "log10eps_r2"]].assign(Gene=gene))
    if best_rows:
        combined = pd.concat(best_rows, ignore_index=True)[["Gene", "Method", "raw_r2", "log1p_r2", "log10eps_r2"]]
        lines.append(df_to_simple_markdown(combined))
        lines.append("\n")

    lines.append("## Top FluxAnchor error-contributing reactions\n")
    lines.append(df_to_simple_markdown(outlier_df))
    lines.append("\n")

    lines.append("## Notes\n")
    lines.append("- `log1p` is the most stable log-style metric here because many reactions are exactly zero; `log10(x + 1e-6)` can become misleadingly harsh or unstable under thresholding.\n")
    lines.append("- Any calibration result should be interpreted only when trained on non-classic mutants and evaluated on classic mutants. In-sample calibration is not reported as evidence of real improvement.\n")
    lines.append("- Filtering by higher true-flux thresholds did not reliably improve raw `R^2` for FluxAnchor on these classic mutants.\n")

    return "\n".join(lines)


def main():
    _, long_df = load_long_table()

    summary_rows = []
    for method in METHODS:
        method_df = long_df[long_df["Method"] == method]
        aggregate_df = method_df[method_df["is_classic"]]
        for threshold in THRESHOLDS:
            row = evaluate_slice(aggregate_df, "all_classic", method, threshold)
            if row is not None:
                summary_rows.append(row)

        for gene in CLASSIC_GENES:
            gene_df = method_df[method_df["Gene"] == gene]
            for threshold in THRESHOLDS:
                row = evaluate_slice(gene_df, f"classic_{gene}", method, threshold)
                if row is not None:
                    summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    calibration_rows = []
    for method in METHODS:
        train_df = long_df[(long_df["Method"] == method) & (~long_df["is_classic"])]
        test_df_all = long_df[(long_df["Method"] == method) & (long_df["is_classic"])]
        metrics = fit_linear_calibration(train_df, test_df_all)
        if metrics is not None:
            metrics.update({"scope": "all_classic_test", "Method": method})
            calibration_rows.append(metrics)

        for gene in CLASSIC_GENES:
            test_df_gene = long_df[(long_df["Method"] == method) & (long_df["Gene"] == gene)]
            metrics = fit_linear_calibration(train_df, test_df_gene)
            if metrics is not None:
                metrics.update({"scope": f"classic_{gene}_test", "Method": method})
                calibration_rows.append(metrics)

    calibration_df = pd.DataFrame(calibration_rows)
    outlier_df = collect_fluxanchor_outliers(long_df)

    summary_df.to_csv(summary_csv, index=False)
    outlier_df.to_csv(outlier_csv, index=False)

    report = build_report(summary_df, calibration_df, outlier_df)
    with open(report_md, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"Saved summary: {summary_csv}")
    print(f"Saved outliers: {outlier_csv}")
    print(f"Saved report: {report_md}")
    print("\nAggregate all_classic baseline (threshold = 0):")
    print(summary_df[(summary_df["scope"] == "all_classic") & (summary_df["threshold"] == 0.0)][["Method", "raw_r2", "log1p_r2", "log10eps_r2", "raw_rmse", "log1p_rmse"]].to_string(index=False))
    print("\nCalibration trained on non-classic mutants, tested on all_classic:")
    if calibration_df.empty:
        print("No calibration results available")
    else:
        print(calibration_df[calibration_df["scope"] == "all_classic_test"][ ["Method", "raw_r2", "log1p_r2", "raw_rmse", "log1p_rmse", "calibration_slope", "calibration_intercept"]].to_string(index=False))


if __name__ == "__main__":
    main()
