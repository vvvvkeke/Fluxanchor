#!/usr/bin/env python3
"""
Plot scatterplots of predicted performance for classic mutants.

Select a few classic mutants (Δpgi, Δzwf, Δsdh, Δcyo, Δldh) and draw scatterplots for each subplot.
"""
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.stats import gaussian_kde


# =========================== Core parameter configuration ===========================
# Organism selection
# ORGANISM = "Bacillus subtilis"  # Options: "iECDH1ME8569_1439" or "Bacillus subtilis"
ORGANISM = "iECDH1ME8569_1439"  # Options: "iECDH1ME8569_1439" or "Bacillus subtilis"

# Training data ratio (must match the ratio used for training)
TRAIN_DATA_RATIO = "wildtype"  # Or a numeric value, e.g., 0.2

# Classic mutant list
# E. coli: Δpgi, Δzwf, Δsdh, Δcyo, Δldh
# B. subtilis: ΔpdhA, ΔpdhB, ΔpdhC, ΔpdhD, ΔcitZ
if ORGANISM == "iECDH1ME8569_1439":
    CLASSIC_MUTANTS = {
        'Zwf': 'ΔZwf',  # Pentose phosphate pathway, G6PDH2r reaction
        'Pgi': 'ΔPgi',  # Glycolysis, pgi reaction
        'Crp': 'ΔCrp',
    }
    organism_name = "E. coli"
elif ORGANISM == "Bacillus subtilis":
    CLASSIC_MUTANTS = {
        'pckA': 'ΔpckA',
        'pit': 'Δpit',
        'yhaU': 'ΔyhaU',
        'tuaA': 'ΔtuaA',
        'cydA': 'ΔcydA'
    }  # gndA
    organism_name = "B. subtilis"
else:
    raise ValueError(f"Unsupported organism: {ORGANISM}")

# Method list
METHODS = ['FluxAnchor', 'KinLLM', 'fba']
METHOD_LABELS = {'FluxAnchor': 'FluxAnchor', 'KinLLM': 'KinLLM', 'fba': 'FBA'}

# Color palette
COLORS = ['#96CCEA', '#B2A3DD', '#ED949A']

# Figure parameters
figsize_per_mutant = (7, 7)  # Figure size per mutant
confidence_level = 0.95  # Regression confidence interval level
dpi = 600  # Output resolution

# Edge plot version (2 = density plot, recommended to show distributions)
version = 2
# =====================================================================

# Set global plotting style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.weight': 'normal',
    'font.size': 14,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
})

# Determine file suffix based on TRAIN_DATA_RATIO
if TRAIN_DATA_RATIO == "wildtype":
    file_suffix = ""
else:
    file_suffix = f"_{TRAIN_DATA_RATIO}"

# Configure file paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ORGANISM == "iECDH1ME8569_1439":
    output_dir = os.path.join(project_root, f'analysis/scatter_{ORGANISM.replace(" ", "_")}')
    wildtype_csv_path = os.path.join(project_root, 'predict/iECDH1ME8569_1439/analysis/13C_analysis_data_extend_reactions.csv')
    detailed_csv_path = os.path.join(project_root, 'results_extend_reactions/iECDH1ME8569_1439/detailed.csv')
elif ORGANISM == "Bacillus subtilis":
    detailed_csv_path = os.path.join(project_root, f'results_threshold_0.01/Bacillus_subtilis/detailed{file_suffix}.csv')
    output_dir = os.path.join(project_root, f'analysis/scatter_{ORGANISM.replace(" ", "_")}')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

print("=" * 100)
print(f"Plotting {organism_name} classic mutant prediction scatterplots")
print("=" * 100)
print(f"Organism: {organism_name}")
print(f"Data file: {detailed_csv_path}")
print(f"Output directory: {output_dir}")
print(f"Classic mutants: {list(CLASSIC_MUTANTS.values())}")
print(f"Training data ratio: {TRAIN_DATA_RATIO}")
print("=" * 100)

# Read data
print("\nReading data...")
try:
    detailed_df = pd.read_csv(detailed_csv_path)
    print(f"Detailed data shape: {detailed_df.shape}")
except Exception as e:
    print(f"Error: unable to read data file {detailed_csv_path}")
    print(f"Exception: {str(e)}")
    exit(1)

# Extract all reaction names from the column names
all_columns = detailed_df.columns.tolist()
reaction_names = sorted(list(set([
    col.replace('_true', '').replace('_pred', '')
    for col in all_columns
    if col.endswith('_true') or col.endswith('_pred')
])))
print(f"Detected {len(reaction_names)} reactions")


def format_mutant_label_for_plot(label: str) -> str:
    """Return a mathtext string that renders the mutant name in italics."""
    latex_label = label.replace('Δ', r'\Delta ').strip()
    if not latex_label:
        latex_label = label
    return rf"$\mathit{{{latex_label}}}$"


def extract_mutant_data(detailed_df, gene_name, reaction_names):
    """
    Extract the true and predicted values for the specified mutant from detailed.csv.

    Returns: dict of {method: {'true': array, 'pred': array}}
    """
    mutant_data = {}

    for method in METHODS:
        # Filter rows for the current method and gene
        method_gene_df = detailed_df[(detailed_df['Method'] == method) &
                                     (detailed_df['Gene'] == gene_name)]

        if method_gene_df.empty:
            print(f"  Warning: no data found for {method} - {gene_name}")
            continue

        if len(method_gene_df) > 1:
            print(f"  Warning: {method} - {gene_name} has multiple rows, using the first one")
            method_gene_df = method_gene_df.iloc[0:1]

        # Extract all reaction true and predicted values
        true_values = []
        pred_values = []

        for reaction in reaction_names:
            true_col = f"{reaction}_true"
            pred_col = f"{reaction}_pred"

            if true_col in method_gene_df.columns and pred_col in method_gene_df.columns:
                true_val = method_gene_df[true_col].values[0]
                pred_val = method_gene_df[pred_col].values[0]

                # Keep non-NaN values only
                if pd.notna(true_val) and pd.notna(pred_val):
                    true_values.append(abs(true_val))
                    pred_values.append(abs(pred_val))

        if len(true_values) > 0:
            mutant_data[method] = {
                'true': np.array(true_values),
                'pred': np.array(pred_values)
            }
        else:
            print(f"  Warning: {method} - {gene_name} does not have valid data")

    return mutant_data


def plot_scatter_with_regression(mutant_name, mutant_label, mutant_data, output_path):
    """
    Plot a scatterplot for a single mutant (including all methods).
    """
    formatted_mutant_label = format_mutant_label_for_plot(mutant_label)

    # Create figure and axes
    fig = plt.figure(figsize=figsize_per_mutant)

    # Define axis positions
    left, width, bottom, height, spacing = 0.15, 0.60, 0.15, 0.60, 0.005

    # Main plot: scatter + regression line + confidence interval
    ax_scatter = plt.axes([left, bottom, width, height])

    # Top marginal plot: distribution of measured fluxes
    ax_histx = plt.axes([left, bottom + height + spacing, width, 0.15])

    # Right marginal plot: distribution of predicted fluxes
    ax_histy = plt.axes([left + width + spacing, bottom, 0.15, height])

    # Collect all data to unify axis limits
    all_true = []
    all_pred = []
    for method in METHODS:
        if method in mutant_data:
            all_true.extend(mutant_data[method]['true'])
            all_pred.extend(mutant_data[method]['pred'])

    if len(all_true) == 0:
        print(f"  Error: {mutant_label} does not have any valid data")
        plt.close(fig)
        return

    # Compute axis ranges
    all_values = all_true + all_pred
    min_val = min(all_values)
    max_val = max(all_values)
    padding = (max_val - min_val) * 0.08

    ax_scatter.set_xlim(min_val - padding, max_val + padding)
    ax_scatter.set_ylim(min_val - padding, max_val + padding)

    # Add diagonal line representing perfect prediction
    ax_scatter.plot([min_val - padding, max_val + padding],
                    [min_val - padding, max_val + padding],
                    'k--', linewidth=1.5, alpha=0.3, zorder=1, label='Perfect prediction')

    # Store regression results
    results = []
    group_data_list = []

    # Plot data for each method
    for i, method in enumerate(METHODS):
        if method not in mutant_data:
            continue

        color = COLORS[i % len(COLORS)]
        method_label = METHOD_LABELS.get(method, method)

        X = mutant_data[method]['true']
        y = mutant_data[method]['pred']

        group_data_list.append((method_label, X, y, color))

        # Plot scatter points
        ax_scatter.scatter(X, y, color=color, s=60, alpha=0.7,
                           edgecolors='none', linewidths=0, zorder=5)

        # Linear regression (for regression line and confidence interval)
        if len(X) > 1:
            X_2d = X.reshape(-1, 1)
            model = LinearRegression().fit(X_2d, y)
            y_pred = model.predict(X_2d)

            # Compute statistics
            slope = model.coef_[0]
            intercept = model.intercept_
            n = len(X)

            computed_r2 = r2_score(y, y_pred)
            computed_mse = mean_squared_error(y, y_pred)
            computed_rmse = np.sqrt(computed_mse)

            # Save metrics
            results.append({
                'method': method_label,
                'r2': computed_r2,
                'mse': computed_mse,
                'rmse': computed_rmse,
                'slope': slope,
                'intercept': intercept,
                'n': n,
                'color': color
            })

            # Regression line covering the axis range (start from zero)
            x_min, x_max = 0, ax_scatter.get_xlim()[1]
            x_vals = np.linspace(x_min, x_max, 100)
            y_vals = model.predict(x_vals.reshape(-1, 1))

            # Confidence interval
            mse_val = np.sum((y - y_pred) ** 2) / (n - 2)
            se = np.sqrt(mse_val * (1 / n + (x_vals - X.mean()) ** 2 / np.sum((X - X.mean()) ** 2)))
            t_val = stats.t.ppf((1 + confidence_level) / 2, n - 2)

            # Fill confidence band
            ax_scatter.fill_between(x_vals, y_vals - t_val * se, y_vals + t_val * se,
                                    color=color, alpha=0.2, linewidth=0, zorder=2)

            # Regression line
            ax_scatter.plot(x_vals, y_vals, color=color, linestyle='--',
                            linewidth=2.5, alpha=0.8, zorder=3)

    # Draw marginal density plots (version 2)
    if version == 2:
        all_x_densities = []
        all_y_densities = []

        for method_label, X, y, color in group_data_list:
            if len(X) > 1:
                x_kde = gaussian_kde(X)
                y_kde = gaussian_kde(y)

                x_plot = np.linspace(ax_scatter.get_xlim()[0], ax_scatter.get_xlim()[1], 200)
                y_plot = np.linspace(ax_scatter.get_ylim()[0], ax_scatter.get_ylim()[1], 200)

                x_density = x_kde(x_plot)
                y_density = y_kde(y_plot)

                all_x_densities.append(x_density)
                all_y_densities.append(y_density)

                ax_histx.plot(x_plot, x_density, color=color, linewidth=2)
                ax_histx.fill_between(x_plot, x_density, alpha=0.3, color=color)

                ax_histy.plot(y_density, y_plot, color=color, linewidth=2)
                ax_histy.fill_betweenx(y_plot, y_density, alpha=0.3, color=color)

        # Set shared density axis limits
        if all_x_densities and all_y_densities:
            max_x_density = max([density.max() for density in all_x_densities])
            max_y_density = max([density.max() for density in all_y_densities])
            unified_max_density = max(max_x_density, max_y_density) * 1.1

            ax_histx.set_ylim(0, unified_max_density)
            ax_histy.set_xlim(0, unified_max_density)

    # Add R^2 and RMSE text
    y_pos = 0.93
    for result in results:
        ax_scatter.text(
            0.03, y_pos,
            f"{result['method']}: $r^2$ = {result['r2']:.3f}, RMSE = {result['rmse']:.2f}",
            color=result['color'], transform=ax_scatter.transAxes,
            fontsize=12
        )
        y_pos -= 0.06
    ax_scatter.text(
            0.03, y_pos,
            formatted_mutant_label, transform=ax_scatter.transAxes,
            fontsize=12
        )
    # Axis labels and title
    ax_scatter.set_xlabel('Measured flux (mmol/gDW/h)', fontsize=16)
    ax_scatter.set_ylabel('Predicted flux (mmol/gDW/h)', fontsize=16)
    ax_scatter.set_title(f'{formatted_mutant_label} ({organism_name})', fontsize=18, pad=15)

    # Hide axes for marginal plots
    for ax in [ax_histx, ax_histy]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Keep axes synchronized
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Output regression summary
    print(f"  {mutant_label} regression summary:")
    for result in results:
        print(f"    {result['method']}: R^2 = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}, n = {result['n']}")


def extract_all_samples_average(detailed_df, reaction_names):
    """
    Extract the reaction-wise averages across all mutants (and wildtype if available).

    Returns: dict of {method: {'true': array, 'pred': array, 'reaction_names': list}}
    """
    average_data = {}

    for method in METHODS:
        # Filter all rows for the current method
        method_df = detailed_df[detailed_df['Method'] == method].copy()

        if method_df.empty:
            print(f"  Warning: no data found for {method}")
            continue

        print(f"  {method}: found {len(method_df)} samples")

        # Compute average per reaction
        reaction_true_means = []
        reaction_pred_means = []
        valid_reactions = []

        for reaction in reaction_names:
            true_col = f"{reaction}_true"
            pred_col = f"{reaction}_pred"

            if true_col in method_df.columns and pred_col in method_df.columns:
                true_values = method_df[true_col].dropna()
                pred_values = method_df[pred_col].dropna()

                # Only compute averages when both vectors are non-empty
                if len(true_values) > 0 and len(pred_values) > 0:
                    true_mean = abs(true_values.mean())
                    pred_mean = abs(pred_values.mean())

                    reaction_true_means.append(true_mean)
                    reaction_pred_means.append(pred_mean)
                    valid_reactions.append(reaction)

        if len(reaction_true_means) > 0:
            average_data[method] = {
                'true': np.array(reaction_true_means),
                'pred': np.array(reaction_pred_means),
                'reaction_names': valid_reactions
            }
            print(f"  {method}: valid reactions = {len(valid_reactions)}")
        else:
            print(f"  Warning: {method} does not have valid data")

    return average_data


def plot_average_scatter(average_data, output_path):
    """
    Plot the scatterplot of reaction averages across all samples.
    """
    fig = plt.figure(figsize=figsize_per_mutant)

    left, width, bottom, height, spacing = 0.15, 0.60, 0.15, 0.60, 0.005

    ax_scatter = plt.axes([left, bottom, width, height])
    ax_histx = plt.axes([left, bottom + height + spacing, width, 0.15])
    ax_histy = plt.axes([left + width + spacing, bottom, 0.15, height])

    all_true = []
    all_pred = []
    for method in METHODS:
        if method in average_data:
            all_true.extend(average_data[method]['true'])
            all_pred.extend(average_data[method]['pred'])

    if len(all_true) == 0:
        print("  Error: no valid data available")
        plt.close(fig)
        return

    all_values = all_true + all_pred
    min_val = min(all_values)
    max_val = max(all_values)
    padding = (max_val - min_val) * 0.08

    ax_scatter.set_xlim(min_val - padding, max_val + padding)
    ax_scatter.set_ylim(min_val - padding, max_val + padding)

    ax_scatter.plot([min_val - padding, max_val + padding],
                    [min_val - padding, max_val + padding],
                    'k--', linewidth=1.5, alpha=0.3, zorder=1, label='Perfect prediction')

    results = []
    group_data_list = []

    for i, method in enumerate(METHODS):
        if method not in average_data:
            continue

        color = COLORS[i % len(COLORS)]
        method_label = METHOD_LABELS.get(method, method)

        X = average_data[method]['true']
        y = average_data[method]['pred']

        group_data_list.append((method_label, X, y, color))

        ax_scatter.scatter(X, y, color=color, s=60, alpha=0.7,
                           edgecolors='none', linewidths=0, zorder=5)

        if len(X) > 1:
            X_2d = X.reshape(-1, 1)
            model = LinearRegression().fit(X_2d, y)
            y_pred = model.predict(X_2d)

            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            slope = model.coef_[0]
            intercept = model.intercept_
            n = len(X)

            results.append({
                'method': method_label,
                'r2': r2,
                'rmse': rmse,
                'slope': slope,
                'intercept': intercept,
                'n': n,
                'color': color
            })

            x_min, x_max = 0, ax_scatter.get_xlim()[1]
            x_vals = np.linspace(x_min, x_max, 100)
            y_vals = model.predict(x_vals.reshape(-1, 1))

            mse_val = np.sum((y - y_pred) ** 2) / (n - 2)
            se = np.sqrt(mse_val * (1 / n + (x_vals - X.mean()) ** 2 / np.sum((X - X.mean()) ** 2)))
            t_val = stats.t.ppf((1 + confidence_level) / 2, n - 2)

            ax_scatter.fill_between(x_vals, y_vals - t_val * se, y_vals + t_val * se,
                                    color=color, alpha=0.2, linewidth=0, zorder=2)

            ax_scatter.plot(x_vals, y_vals, color=color, linestyle='--',
                            linewidth=2.5, alpha=0.8, zorder=3)

    if version == 2:
        all_x_densities = []
        all_y_densities = []

        for method_label, X, y, color in group_data_list:
            if len(X) > 1:
                x_kde = gaussian_kde(X)
                y_kde = gaussian_kde(y)

                x_plot = np.linspace(ax_scatter.get_xlim()[0], ax_scatter.get_xlim()[1], 200)
                y_plot = np.linspace(ax_scatter.get_ylim()[0], ax_scatter.get_ylim()[1], 200)

                x_density = x_kde(x_plot)
                y_density = y_kde(y_plot)

                all_x_densities.append(x_density)
                all_y_densities.append(y_density)

                ax_histx.plot(x_plot, x_density, color=color, linewidth=2)
                ax_histx.fill_between(x_plot, x_density, alpha=0.3, color=color)

                ax_histy.plot(y_density, y_plot, color=color, linewidth=2)
                ax_histy.fill_betweenx(y_plot, y_density, alpha=0.3, color=color)

        if all_x_densities and all_y_densities:
            max_x_density = max([density.max() for density in all_x_densities])
            max_y_density = max([density.max() for density in all_y_densities])
            unified_max_density = max(max_x_density, max_y_density) * 1.1

            ax_histx.set_ylim(0, unified_max_density)
            ax_histy.set_xlim(0, unified_max_density)

    y_pos = 0.93
    for result in results:
        ax_scatter.text(
            0.03, y_pos,
            f"{result['method']}: $R^2$ = {result['r2']:.3f}, RMSE = {result['rmse']:.2f}",
            color=result['color'], transform=ax_scatter.transAxes,
            fontsize=12
        )
        y_pos -= 0.06

    ax_scatter.set_xlabel('Average measured flux (mmol/gDW/h)', fontsize=16)
    ax_scatter.set_ylabel('Average predicted flux (mmol/gDW/h)', fontsize=16)
    ax_scatter.set_title(f'All samples average ({organism_name})', fontsize=18, pad=15)

    for ax in [ax_histx, ax_histy]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")

    print(f"  All-sample average regression summary:")
    for result in results:
        print(f"    {result['method']}: R^2 = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}, n = {result['n']}")


# Main loop: plot scatterplots for each classic mutant
print("\nStarting scatterplot generation...")
for mutant_name, mutant_label in CLASSIC_MUTANTS.items():
    print(f"\nProcessing {mutant_label}...")

    # Extract data
    mutant_data = extract_mutant_data(detailed_df, mutant_name, reaction_names)

    if not mutant_data:
        print(f"  Skipping {mutant_label}: no valid data")
        continue

    # Plot and save
    output_filename = f"{mutant_label.replace('Δ', 'delta_')}_scatter.png"
    output_path = os.path.join(output_dir, output_filename)
    plot_scatter_with_regression(mutant_name, mutant_label, mutant_data, output_path)
    output_filename = f"{mutant_label.replace('Δ', 'delta_')}_scatter.pdf"
    output_path = os.path.join(output_dir, output_filename)
    plot_scatter_with_regression(mutant_name, mutant_label, mutant_data, output_path)

# print("\n" + "=" * 100)
# print("Plotting the reaction-average scatterplot for all mutants and wildtype...")
# print("=" * 100)

# # Extract reaction averages for all samples
# average_data = extract_all_samples_average(detailed_df, reaction_names)

# if average_data:
#     # Plot the averaged scatterplot
#     average_output_filename = "all_samples_average_scatter.png"
#     average_output_path = os.path.join(output_dir, average_output_filename)
#     plot_average_scatter(average_data, average_output_path)
# else:
#     print("  Skipping average scatterplot: no valid data")

print("\n" + "=" * 100)
print("All scatterplots generated!")
print(f"Output directory: {output_dir}")
print("=" * 100)
