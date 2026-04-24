#!/usr/bin/env python
"""Print true fluxes and predictions for selected reactions and methods."""

from __future__ import annotations

import os
from typing import Dict

import pandas as pd

INPUT_FILE = "./detailed_best.csv"
KNOCKOUT_GENE = "Pgi"
REACTIONS_OF_INTEREST = [
    "PGI",
    "PFK",
    "FBP",
    "FBA",
    "TPI",
    "G6PDH2r",
    "GND",
    "RPE",
    "RPE_reverse",
    "RPI",
    "RPI_reverse",
    "TKT1",
    "TKT1_reverse",
    "TKT2",
    "TKT2_reverse",
    "TALA",
    "TALA_reverse",
    "EDD",
    "EDA",
    "GAPD",
    "PGK_reverse",
    "PGM_reverse",
    "ENO",
    "PYK",
    "PDH",
    "CS",
    "ACONTa",
    "ACONTb",
    "ICDHyr",
    "AKGDH",
    "SUCOAS",
    "SUCOAS_reverse",
    "SUCDi",
    "FUM",
    "MDH",
    "PPCK",
    "PPC",
    "ME1",
    "ME2",
    "ICL",
    "MALS",
]
METHODS = ["FluxGen", "KinLLM", "fba"]


def _load_method_rows(gene_data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return the first row for each method so we can pull flux columns."""
    rows: Dict[str, pd.Series] = {}
    for method in METHODS:
        method_rows = gene_data[gene_data["Method"] == method]
        if method_rows.empty:
            print(f"Warning: method '{method}' has no data for gene {KNOCKOUT_GENE}")
            continue
        rows[method] = method_rows.iloc[0]
    return rows


def _get_value(row: pd.Series | None, column: str) -> float | None:
    if row is None or column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return float(value)


def _format_value(value: float | None) -> str:
    return f"{value:12.4f}" if value is not None else f"{'N/A':>12}"


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_FILE)

    df = pd.read_csv(input_path)
    gene_mask = df["Gene"].str.lower() == KNOCKOUT_GENE.lower()
    gene_data = df[gene_mask]

    if gene_data.empty:
        print(f"No entries found for gene '{KNOCKOUT_GENE}'.")
        return

    method_rows = _load_method_rows(gene_data)
    if not method_rows:
        print("No method data available, nothing to show.")
        return

    header = ["Reaction", "True"] + METHODS
    print(f"Knockout Gene: {KNOCKOUT_GENE}")
    print(
        f"{header[0]:<15} "
        + " ".join(f"{title:>12}" for title in header[1:])
    )
    print("-" * (15 + 1 + 12 * len(header[1:]) + len(header[1:])))

    for reaction in REACTIONS_OF_INTEREST:
        true_column = f"{reaction}_true"
        true_value = None
        for row in method_rows.values():
            true_value = _get_value(row, true_column)
            if true_value is not None:
                break

        predictions = []
        for method in METHODS:
            row = method_rows.get(method)
            pred_column = f"{reaction}_pred"
            predictions.append(_get_value(row, pred_column))

        formatted_preds = " ".join(_format_value(val) for val in predictions)
        print(f"{reaction:<15} {_format_value(true_value)} {formatted_preds}")


if __name__ == "__main__":
    main()
