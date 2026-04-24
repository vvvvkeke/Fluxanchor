#!/usr/bin/env python3
"""
WT Flux Template Parity Plot (Supplementary Figure)

Compares the FluxAnchor WT flux template (predicted/simulated) against
experimentally measured WT fluxes from Sauer et al. (Mol Syst Biol, 2011).

X-axis: Measured WT fluxes (13C-constrained flux analysis)
Y-axis: FluxAnchor WT template fluxes (genome-scale FBA constrained by 13C ratios)

Note: PYK is excluded because in the genome-scale model, PEP->PYR flux is
absorbed by the PTS system (GLCptspp), creating a known FBA degeneracy.
A version including PYK is also generated for full transparency.
"""
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import os

# ========================= Configuration =========================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 600

# FluxAnchor WT template file
WT_TEMPLATE_FILE = '/home/huangjiesheng/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/13C_analysis_data_extend_reactions.csv'

# Font setup: try Arial, fall back to DejaVu Sans
try:
    matplotlib.font_manager.findfont('Arial', fallback_to_default=False)
    FONT_FAMILY = 'Arial'
except:
    FONT_FAMILY = 'DejaVu Sans'

# ========================= Data =========================
# Sauer et al. (2011) Supplementary Table 2
# WT 13C-constrained flux analysis: absolute fluxes (mmol/gDW/h)
SAUER_WT_DATA = {
    # Reaction_ID: (measured, CI_95, display_name, exclude_from_main)
    'GLCptspp': (8.13, 0.34, 'pts',       False),
    'G6PDH2r':  (2.39, 0.30, 'zwf',       False),
    'GND':      (1.65, 0.44, 'gnd',       False),
    'PGI':      (5.71, 0.39, 'pgi',       False),
    'EDA':      (0.74, 0.65, 'edd/eda',   False),
    'FBA':      (6.46, 0.62, 'pfk/fba',   False),
    'TKT1':     (0.53, 0.15, 'tktA',      False),
    'TKT2':     (0.27, 0.15, 'tktB',      False),
    'TALA':     (0.53, 0.15, 'tal',       False),
    'GAPD':     (13.87, 0.84, 'gap/pgk',  False),
    'ENO':      (12.94, 0.84, 'eno',      False),
    'PYK':      (9.99, 0.98, 'pyk',       True),   # PTS degeneracy
    'PDH':      (9.14, 0.64, 'pdh',       False),
    'CS':       (2.20, 0.45, 'gltA',      False),
    'ICDHyr':   (2.20, 0.45, 'icd',       False),
    'AKGDH':    (1.29, 0.44, 'sucAB',     False),
    'SUCDi':    (1.29, 0.44, 'sdh/fum',   False),
    'MDH':      (0.81, 0.50, 'mdh',       False),
    'ME2':      (0.48, 0.73, 'maeB',      False),
    'PPCK':     (0.23, 0.26, 'pck',       False),
    'PPC':      (2.64, 0.78, 'ppc',       False),
    'PTAr':     (5.48, 0.57, 'pta/ackA',  False),
    'ICL':      (0.00, 0.00, 'aceA/B',    False),
}

# Pathway color mapping
PATHWAY_COLORS = {
    'Glycolysis':       '#2166AC',
    'PP pathway':       '#4DAF4A',
    'ED pathway':       '#FF7F00',
    'TCA cycle':        '#E31A1C',
    'Anaplerosis':      '#984EA3',
    'Acetate':          '#A65628',
    'Glyoxylate':       '#999999',
}

REACTION_PATHWAY = {
    'GLCptspp': 'Glycolysis', 'PGI': 'Glycolysis', 'FBA': 'Glycolysis',
    'GAPD': 'Glycolysis', 'ENO': 'Glycolysis', 'PYK': 'Glycolysis', 'PDH': 'Glycolysis',
    'G6PDH2r': 'PP pathway', 'GND': 'PP pathway',
    'TKT1': 'PP pathway', 'TKT2': 'PP pathway', 'TALA': 'PP pathway',
    'EDA': 'ED pathway',
    'CS': 'TCA cycle', 'ICDHyr': 'TCA cycle', 'AKGDH': 'TCA cycle',
    'SUCDi': 'TCA cycle', 'MDH': 'TCA cycle',
    'PPC': 'Anaplerosis', 'PPCK': 'Anaplerosis', 'ME2': 'Anaplerosis',
    'PTAr': 'Acetate',
    'ICL': 'Glyoxylate',
}

# Marker shapes per pathway
PATHWAY_MARKERS = {
    'Glycolysis':   'o',
    'PP pathway':   's',
    'ED pathway':   'D',
    'TCA cycle':    '^',
    'Anaplerosis':  'v',
    'Acetate':      'P',
    'Glyoxylate':   'X',
}


def load_wt_template():
    """Load FluxAnchor WT template fluxes from the analysis file."""
    df = pd.read_csv(WT_TEMPLATE_FILE)
    template = {}
    for _, row in df.iterrows():
        template[row['Reaction']] = row['Value']
    return template


def build_comparison_data(template, exclude_pyk=True):
    """Build paired measured vs predicted arrays."""
    measured_list, predicted_list, ci_list = [], [], []
    labels, pathways, rxn_ids = [], [], []

    for rxn_id, (measured, ci, name, excluded) in SAUER_WT_DATA.items():
        if exclude_pyk and excluded:
            continue

        fwd = template.get(rxn_id, 0.0)
        rev = template.get(f'{rxn_id}_reverse', 0.0)
        predicted = fwd - rev

        if measured > 0 and predicted < 0:
            predicted = abs(predicted)

        measured_list.append(measured)
        predicted_list.append(predicted)
        ci_list.append(ci)
        labels.append(name)
        pathways.append(REACTION_PATHWAY.get(rxn_id, 'Other'))
        rxn_ids.append(rxn_id)

    return (np.array(measured_list), np.array(predicted_list),
            np.array(ci_list), labels, pathways, rxn_ids)


def draw_parity_plot(measured, predicted, ci, labels, pathways, output_prefix,
                     title_suffix=''):
    """Draw the parity plot with diagonal, R^2, MSE, error bars, and annotations."""
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.weight': 'bold',
        'font.size': 12,
        'mathtext.default': 'regular',
    })

    fig, ax = plt.subplots(figsize=(7, 7))

    # Axis range
    all_vals = np.concatenate([measured + ci, predicted])
    max_val = max(all_vals) * 1.12
    min_val = -0.5

    # y = x diagonal
    ax.plot([min_val, max_val], [min_val, max_val], 'k-',
            linewidth=1.2, alpha=0.35, zorder=1, label='$y = x$')

    # Shaded +/- 1 mmol/gDW/h band around diagonal
    band = np.linspace(min_val, max_val, 100)
    ax.fill_between(band, band - 1, band + 1,
                    color='#CCCCCC', alpha=0.2, zorder=0)

    # Plot points by pathway
    unique_pathways = list(dict.fromkeys(pathways))
    for pathway in unique_pathways:
        mask = np.array([p == pathway for p in pathways])
        color = PATHWAY_COLORS.get(pathway, '#333333')
        marker = PATHWAY_MARKERS.get(pathway, 'o')

        # Error bars (95% CI)
        has_ci = ci[mask] > 0
        if has_ci.any():
            ax.errorbar(measured[mask][has_ci], predicted[mask][has_ci],
                        xerr=ci[mask][has_ci],
                        fmt='none', ecolor=color, elinewidth=1.0,
                        capsize=3, capthick=1.0, alpha=0.45, zorder=3)

        ax.scatter(measured[mask], predicted[mask],
                   c=color, s=90, alpha=0.85, edgecolor='white',
                   linewidth=0.8, zorder=5, label=pathway, marker=marker)

    # Statistics
    r, p_val = pearsonr(measured, predicted)
    r2 = r ** 2
    mse = mean_squared_error(measured, predicted)
    rmse = np.sqrt(mse)

    # Linear regression fit line
    slope, intercept, _, _, _ = stats.linregress(measured, predicted)
    x_fit = np.linspace(min_val, max_val, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='#D62728', linewidth=2, alpha=0.5,
            linestyle='--', zorder=2)

    # Statistics text box
    stats_text = (f'$R$ = {r:.3f}\n'
                  f'$R^2$ = {r2:.3f}\n'
                  f'RMSE = {rmse:.2f}\n'
                  f'$n$ = {len(measured)}')
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            fontsize=13, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92))

    # Annotate each point with gene name
    for i in range(len(measured)):
        deviation = abs(predicted[i] - measured[i])
        # Position annotation to avoid overlap
        if predicted[i] < measured[i]:
            offset = (6, -10)
            ha = 'left'
        else:
            offset = (6, 6)
            ha = 'left'

        # Only annotate points with notable deviation or key reactions
        if deviation > 0.8 or measured[i] > 8 or labels[i] in ['pts', 'pdh', 'pyk']:
            ax.annotate(labels[i], (measured[i], predicted[i]),
                        textcoords='offset points', xytext=offset,
                        fontsize=7.5, alpha=0.65, fontstyle='italic', ha=ha)

    # Axis labels
    ax.set_xlabel('Measured WT flux (mmol/gDW/h)', fontsize=14, fontweight='bold')
    ax.set_ylabel('FluxAnchor WT template (mmol/gDW/h)', fontsize=14, fontweight='bold')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')

    # Legend
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9,
              edgecolor='#CCCCCC', ncol=1, handletextpad=0.5)

    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    for ext in ['png', 'pdf', 'svg']:
        out_path = os.path.join(OUTPUT_DIR, f'{output_prefix}.{ext}')
        fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
        print(f'  Saved: {out_path}')

    plt.close(fig)
    return r, r2, mse, rmse


def save_comparison_table(measured, predicted, ci, labels, pathways, rxn_ids,
                          filename='wt_parity_comparison.csv'):
    """Save comparison data as CSV."""
    df = pd.DataFrame({
        'Reaction_ID': rxn_ids,
        'Gene_Name': labels,
        'Pathway': pathways,
        'Measured_Flux': measured,
        'Template_Flux': predicted,
        'CI_95': ci,
        'Absolute_Error': np.abs(predicted - measured),
        'Relative_Error': np.where(measured > 0,
                                    np.abs(predicted - measured) / measured,
                                    np.nan),
    })
    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, index=False, float_format='%.4f')
    print(f'  Saved: {out_path}')
    return df


def print_comparison(rxn_ids, measured, predicted):
    print(f'\n  {"Reaction":<12} {"Measured":>10} {"Template":>10} {"Abs Err":>10}')
    print(f'  {"-"*44}')
    for i, rxn in enumerate(rxn_ids):
        err = abs(predicted[i] - measured[i])
        print(f'  {rxn:<12} {measured[i]:>10.2f} {predicted[i]:>10.2f} {err:>10.2f}')


def main():
    print('=' * 70)
    print('  WT Flux Template Parity Plot')
    print('=' * 70)

    template = load_wt_template()
    print(f'  Loaded WT template: {len(template)} reactions')

    # ---- Main plot: excluding PYK (PTS degeneracy) ----
    print('\n--- Main plot (excluding PYK) ---')
    m, p, ci, labels, pathways, rxn_ids = build_comparison_data(template, exclude_pyk=True)
    print_comparison(rxn_ids, m, p)

    r, r2, mse, rmse = draw_parity_plot(
        m, p, ci, labels, pathways, 'wt_flux_parity_plot')

    print(f'\n  Pearson R  = {r:.4f}')
    print(f'  R-squared  = {r2:.4f}')
    print(f'  RMSE       = {rmse:.4f} mmol/gDW/h')
    print(f'  n          = {len(m)}')

    save_comparison_table(m, p, ci, labels, pathways, rxn_ids)

    # ---- Supplementary: including PYK ----
    print('\n--- Supplementary plot (including PYK) ---')
    m2, p2, ci2, labels2, pathways2, rxn_ids2 = build_comparison_data(template, exclude_pyk=False)

    r2_val, r2_sq, mse2, rmse2 = draw_parity_plot(
        m2, p2, ci2, labels2, pathways2, 'wt_flux_parity_plot_with_pyk')

    print(f'\n  Pearson R  = {r2_val:.4f}')
    print(f'  R-squared  = {r2_sq:.4f}')
    print(f'  RMSE       = {rmse2:.4f} mmol/gDW/h')
    print(f'  n          = {len(m2)}')

    save_comparison_table(m2, p2, ci2, labels2, pathways2, rxn_ids2,
                          'wt_parity_comparison_with_pyk.csv')

    print('\n' + '=' * 70)
    print('  Done. Output directory:', OUTPUT_DIR)
    print('=' * 70)


if __name__ == '__main__':
    main()
