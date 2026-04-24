import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
from scipy.stats import mannwhitneyu

# Define species information
species_info = {
    'Synechocystis_sp': {
        'name': 'Synechocystis sp',
        'original_data_path': '/home/zhangyangyu/kcat_km_predict/predict/Synechocystis sp/analysis/get_kcat_mw_by_EkiLLm/reaction_kcat_MW.csv',
        'optimized_data_path': '/home/zhangyangyu/kcat_km_predict/predict/Synechocystis sp/analysis/get_kcat_mw_by_EkiLLm/ecGEM/Bayesian/best_enzyme_df.csv',
        'params_path': '/home/zhangyangyu/kcat_km_predict/predict/Synechocystis sp/analysis/get_kcat_mw_by_EkiLLm/ecGEM/Bayesian/best_params.json'
    },
    'E_coli': {
        'name': 'E. coli',
        'original_data_path': '/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/reaction_kcat_MW.csv',
        'optimized_data_path': '/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/Bayesian/best_enzyme_df.csv',
        'params_path': '/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/Bayesian/best_params.json'
    },
    'Bacillus_subtilis': {
        'name': 'Bacillus subtilis',
        'original_data_path': '/home/zhangyangyu/kcat_km_predict/predict/Bacillus subtilis/analysis/get_kcat_mw_by_EkiLLm/reaction_kcat_MW.csv',
        'optimized_data_path': '/home/zhangyangyu/kcat_km_predict/predict/Bacillus subtilis/analysis/get_kcat_mw_by_EkiLLm/ecGEM/Bayesian/best_enzyme_df.csv',
        'params_path': '/home/zhangyangyu/kcat_km_predict/predict/Bacillus subtilis/analysis/get_kcat_mw_by_EkiLLm/ecGEM/Bayesian/best_params.json'
    }
}

# Output directory
output_dir = '/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/Bayesian'

# Set up matplotlib with safe fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

def classify_enzyme(catalytic_efficiency_value, t1, t2):
    """Classify enzyme based on catalytic efficiency thresholds"""
    if catalytic_efficiency_value <= t1:
        return 'Limiting'
    elif catalytic_efficiency_value <= t2:
        return 'Normal'
    else:
        return 'Ultra-fast'

def analyze_species(species_key, species_data):
    """Analyze a single species and create visualization"""
    print(f"\n=== Analyzing {species_data['name']} ===")

    # Check if all required files exist
    if not all(os.path.exists(path) for path in [species_data['original_data_path'],
                                                species_data['optimized_data_path'],
                                                species_data['params_path']]):
        print(f"Missing files for {species_data['name']}, skipping...")
        return None

    # Load the data
    try:
        original_data = pd.read_csv(species_data['original_data_path'])
        optimized_data = pd.read_csv(species_data['optimized_data_path'])

        # Load threshold parameters
        with open(species_data['params_path'], 'r') as f:
            params = json.load(f)

        t1 = params['best_thresholds']['t1']  # Lower threshold
        t2 = params['best_thresholds']['t2']  # Upper threshold
        limiting_scale = params['best_scales']['limiting']
        normal_scale = params['best_scales']['normal']
        ultra_scale = params['best_scales']['ultra']

        print(f"Thresholds: t1={t1:.0f}, t2={t2:.0f}")
        print(f"Scales: limiting={limiting_scale:.3f}, normal={normal_scale:.3f}, ultra={ultra_scale:.3f}")

    except Exception as e:
        print(f"Error loading data for {species_data['name']}: {e}")
        return None

    # Extract kcat values and apply log10 transformation (removing zeros and negative values)
    original_kcat = original_data['kcat'][original_data['kcat'] > 0]
    optimized_kcat = optimized_data['kcat'][optimized_data['kcat'] > 0]

    # Create safe species name for file naming
    safe_species_name = species_key

    # Log10 transformation for better visualization
    log_original_kcat = np.log10(original_kcat)
    log_optimized_kcat = np.log10(optimized_kcat)

    # Add enzyme classification based on catalytic efficiency (CORRECTED LOGIC)
    original_data_with_class = original_data[original_data['kcat'] > 0].copy()
    original_data_with_class['enzyme_class'] = original_data_with_class['catalytic_efficiency'].apply(
        lambda x: classify_enzyme(x, t1, t2))
    original_data_with_class['log_original_kcat'] = np.log10(original_data_with_class['kcat'])

    optimized_data_with_class = optimized_data[optimized_data['kcat'] > 0].copy()
    # Use original classification for optimized data to maintain consistency
    optimized_data_with_class['enzyme_class'] = original_data_with_class['enzyme_class'].values
    optimized_data_with_class['log_optimized_kcat'] = np.log10(optimized_data_with_class['kcat'])

    # Create the figure with three subplots
    fig = plt.figure(figsize=(20, 6))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    # Plot 1: KDE plot
    hist_original, bins_original, _ = ax1.hist(log_original_kcat, bins=30, alpha=0.6, density=True,
             label='Original DL Prediction', color='skyblue', edgecolor='black')
    hist_optimized, bins_optimized, _ = ax1.hist(log_optimized_kcat, bins=30, alpha=0.6, density=True,
             label='After Bayesian Optimization', color='lightcoral', edgecolor='black')

    # Export histogram data for GraphPad
    # Calculate bin centers for histogram data
    bin_centers_original = (bins_original[:-1] + bins_original[1:]) / 2
    bin_centers_optimized = (bins_optimized[:-1] + bins_optimized[1:]) / 2

    # Create histogram export dataframe
    max_bins = max(len(bin_centers_original), len(bin_centers_optimized))
    hist_export_df = pd.DataFrame()

    # Pad shorter arrays with NaN
    if len(bin_centers_original) == max_bins:
        hist_export_df['Original_bin_centers'] = bin_centers_original
        hist_export_df['Original_density'] = hist_original
    else:
        padded_centers = np.pad(bin_centers_original, (0, max_bins - len(bin_centers_original)), constant_values=np.nan)
        padded_density = np.pad(hist_original, (0, max_bins - len(hist_original)), constant_values=np.nan)
        hist_export_df['Original_bin_centers'] = padded_centers
        hist_export_df['Original_density'] = padded_density

    if len(bin_centers_optimized) == max_bins:
        hist_export_df['Optimized_bin_centers'] = bin_centers_optimized
        hist_export_df['Optimized_density'] = hist_optimized
    else:
        padded_centers = np.pad(bin_centers_optimized, (0, max_bins - len(bin_centers_optimized)), constant_values=np.nan)
        padded_density = np.pad(hist_optimized, (0, max_bins - len(hist_optimized)), constant_values=np.nan)
        hist_export_df['Optimized_bin_centers'] = padded_centers
        hist_export_df['Optimized_density'] = padded_density

    hist_export_df.to_csv(f'{output_dir}/histogram_data_for_graphpad_{safe_species_name}.csv', index=False)

    # Add KDE curves
    kde_original = stats.gaussian_kde(log_original_kcat)
    kde_optimized = stats.gaussian_kde(log_optimized_kcat)

    x_range = np.linspace(min(min(log_original_kcat), min(log_optimized_kcat)),
                          max(max(log_original_kcat), max(log_optimized_kcat)), 200)

    # Calculate KDE curves for plotting
    kde_original_values = kde_original(x_range)
    kde_optimized_values = kde_optimized(x_range)

    ax1.plot(x_range, kde_original_values, color='blue', linewidth=2, linestyle='--')
    ax1.plot(x_range, kde_optimized_values, color='red', linewidth=2, linestyle='--')

    # Export KDE curve data for GraphPad
    kde_export_df = pd.DataFrame({
        'log10_kcat_x_axis': x_range,
        'Original_DL_KDE_density': kde_original_values,
        'Bayesian_Optimized_KDE_density': kde_optimized_values
    })
    kde_export_df.to_csv(f'{output_dir}/kde_curves_for_graphpad_{safe_species_name}.csv', index=False)

    ax1.set_xlabel('log₁₀(kcat) [s⁻¹]', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'{species_data["name"]}: Distribution Comparison of kcat Values', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    original_mean = np.mean(log_original_kcat)
    optimized_mean = np.mean(log_optimized_kcat)
    original_std = np.std(log_original_kcat)
    optimized_std = np.std(log_optimized_kcat)

    stats_text = f'Original Data:\nMean = {original_mean:.2f}\nStd = {original_std:.2f}\n\nOptimized:\nMean = {optimized_mean:.2f}\nStd = {optimized_std:.2f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Box plot comparison
    data_for_box = [log_original_kcat, log_optimized_kcat]
    labels = ['Original DL\nPrediction', 'Bayesian\nOptimized']

    box_plot = ax2.boxplot(data_for_box, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))

    # Color the boxes differently
    box_plot['boxes'][0].set_facecolor('skyblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')

    ax2.set_ylabel('log₁₀(kcat) [s⁻¹]', fontsize=12)
    ax2.set_title(f'{species_data["name"]}: Box Plot Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add median values as text
    medians = [np.median(data) for data in data_for_box]
    for i, median in enumerate(medians):
        ax2.text(i+1, median, f'Median: {median:.2f}', ha='center', va='bottom', fontweight='bold')

    # Statistical test
    statistic, p_value = mannwhitneyu(log_original_kcat, log_optimized_kcat, alternative='two-sided')

    # Add statistical test result
    test_text = f'Mann-Whitney U test:\np-value = {p_value:.2e}\n{"Significant" if p_value < 0.05 else "Not significant"}'
    ax2.text(0.02, 0.98, test_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Plot 3: Categorical box plot by enzyme type
    enzyme_classes = ['Limiting', 'Normal', 'Ultra-fast']

    # Prepare data for categorical box plot - ALWAYS show all three enzyme classes
    categorical_data_original = []
    categorical_data_optimized = []
    class_labels = []
    positions_with_data = []

    for i, enzyme_class in enumerate(enzyme_classes):
        class_mask = original_data_with_class['enzyme_class'] == enzyme_class
        class_count = class_mask.sum()

        if class_count > 0:
            categorical_data_original.append(original_data_with_class[class_mask]['log_original_kcat'].values)
            categorical_data_optimized.append(optimized_data_with_class[class_mask]['log_optimized_kcat'].values)
            positions_with_data.append(i)
        else:
            # Add placeholder for empty categories
            categorical_data_original.append([np.nan])  # Single NaN value for empty boxplot
            categorical_data_optimized.append([np.nan])

        class_labels.append(f'{enzyme_class}\n(n={class_count})')

    # Always create grouped box plot for all three categories
    positions_orig = np.arange(1, len(enzyme_classes) * 2, 2)
    positions_opt = np.arange(2, len(enzyme_classes) * 2 + 1, 2)

    # Plot boxes individually for better control
    for i, (orig_data, opt_data) in enumerate(zip(categorical_data_original, categorical_data_optimized)):
        if len(orig_data) > 1 and not np.all(np.isnan(orig_data)):
            # Plot original data box
            ax3.boxplot([orig_data], positions=[positions_orig[i]], widths=0.6,
                       patch_artist=True, boxprops=dict(facecolor='skyblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))

            # Plot optimized data box
            ax3.boxplot([opt_data], positions=[positions_opt[i]], widths=0.6,
                       patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))

    # Set labels and formatting - always show all three categories
    ax3.set_xticks(np.arange(1.5, len(enzyme_classes) * 2, 2))
    ax3.set_xticklabels(class_labels)
    ax3.set_ylabel('log₁₀(kcat) [s⁻¹]', fontsize=12)
    ax3.set_title(f'{species_data["name"]}: kcat by Enzyme Category\n(Blue: Original, Red: Optimized)',
                 fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add text annotation for empty categories
    for i, enzyme_class in enumerate(enzyme_classes):
        class_mask = original_data_with_class['enzyme_class'] == enzyme_class
        if class_mask.sum() == 0:
            x_pos = 1.5 + i * 2
            ax3.text(x_pos, ax3.get_ylim()[1] * 0.9, 'No Data',
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

    # Add legend for the third plot
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='skyblue', lw=4, label='Original DL Prediction'),
                       Line2D([0], [0], color='lightcoral', lw=4, label='Bayesian Optimized')]
    ax3.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{output_dir}/kcat_distribution_comparison_{safe_species_name}.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/kcat_distribution_comparison_{safe_species_name}.pdf',
                bbox_inches='tight')

    # Export data for GraphPad Prism
    export_df = pd.DataFrame({
        'Original_DL_Prediction_log10_kcat': pd.Series(log_original_kcat.values),
        'Bayesian_Optimized_log10_kcat': pd.Series(log_optimized_kcat.values)
    })
    export_df.to_csv(f'{output_dir}/kcat_data_for_graphpad_{safe_species_name}.csv', index=False)

    # Also export raw kcat values
    export_raw_df = pd.DataFrame({
        'Original_DL_Prediction_kcat': pd.Series(original_kcat.values),
        'Bayesian_Optimized_kcat': pd.Series(optimized_kcat.values)
    })
    export_raw_df.to_csv(f'{output_dir}/kcat_raw_data_for_graphpad_{safe_species_name}.csv', index=False)

    # Export categorical data for GraphPad
    categorical_export = pd.DataFrame({
        'reaction': original_data_with_class['reactions'],
        'enzyme_class': original_data_with_class['enzyme_class'],
        'catalytic_efficiency': original_data_with_class['catalytic_efficiency'],
        'original_log10_kcat': original_data_with_class['log_original_kcat'],
        'optimized_log10_kcat': optimized_data_with_class['log_optimized_kcat']
    })
    categorical_export.to_csv(f'{output_dir}/kcat_categorical_data_for_graphpad_{safe_species_name}.csv', index=False)

    # Print enzyme classification statistics
    print(f"\nEnzyme Classification Summary for {species_data['name']}:")
    print(f"Thresholds: t1={t1:.0f}, t2={t2:.0f}")

    for i, enzyme_class in enumerate(enzyme_classes):
        class_mask = original_data_with_class['enzyme_class'] == enzyme_class
        class_count = class_mask.sum()

        print(f"\n{enzyme_class} enzymes (n={class_count}):")

        if class_count > 0:
            # Calculate statistics for this enzyme class
            orig_data = categorical_data_original[i]
            opt_data = categorical_data_optimized[i]

            if len(orig_data) > 0 and not np.all(np.isnan(orig_data)):
                orig_median = np.median(orig_data)
                opt_median = np.median(opt_data)
                fold_change = 10**(opt_median - orig_median)

                print(f"  Original median log10(kcat): {orig_median:.3f}")
                print(f"  Optimized median log10(kcat): {opt_median:.3f}")
                print(f"  Median fold change: {fold_change:.2f}x")

                # Show catalytic efficiency range for this class
                class_data = original_data_with_class[class_mask]
                ce_min = class_data['catalytic_efficiency'].min()
                ce_max = class_data['catalytic_efficiency'].max()
                print(f"  Catalytic efficiency range: {ce_min:.0f} - {ce_max:.0f}")

                # Statistical test for each category
                try:
                    stat, p_val = stats.mannwhitneyu(orig_data, opt_data, alternative='two-sided')
                    print(f"  Statistical significance: {'Significant' if p_val < 0.05 else 'Not significant'} (p = {p_val:.2e})")
                except:
                    print(f"  Statistical test: Unable to perform (insufficient data)")
            else:
                print(f"  No valid data for analysis")
        else:
            print(f"  No enzymes detected in this category")

    # Print summary statistics
    print(f"\n=== kcat Distribution Summary for {species_data['name']} ===")
    print(f"\nOriginal Deep Learning Prediction Data:")
    print(f"  Sample size: {len(original_kcat)}")
    print(f"  log10(kcat) mean: {original_mean:.3f}")
    print(f"  log10(kcat) std: {original_std:.3f}")
    print(f"  kcat geometric mean: {10**original_mean:.2f} s^-1")
    print(f"  kcat range: {original_kcat.min():.2f} - {original_kcat.max():.2f} s^-1")

    print(f"\nAfter Bayesian Optimization:")
    print(f"  Sample size: {len(optimized_kcat)}")
    print(f"  log10(kcat) mean: {optimized_mean:.3f}")
    print(f"  log10(kcat) std: {optimized_std:.3f}")
    print(f"  kcat geometric mean: {10**optimized_mean:.2f} s^-1")
    print(f"  kcat range: {optimized_kcat.min():.2f} - {optimized_kcat.max():.2f} s^-1")

    print(f"\nChange Analysis:")
    fold_change = 10**(optimized_mean - original_mean)
    print(f"  Geometric mean fold change: {fold_change:.2f}x")
    print(f"  Standard deviation change: {optimized_std/original_std:.2f}x")
    print(f"  Statistical significance: {'Significant difference' if p_value < 0.05 else 'No significant difference'} (p = {p_value:.2e})")

    plt.show()

    return {
        'species': species_data['name'],
        'original_mean': original_mean,
        'optimized_mean': optimized_mean,
        'fold_change': fold_change,
        'p_value': p_value,
        'sample_size': len(original_kcat),
        'enzyme_counts': {enzyme_class: (original_data_with_class['enzyme_class'] == enzyme_class).sum()
                         for enzyme_class in enzyme_classes}
    }

# Main analysis
print("Starting Multi-Species Enzyme Kinetics Analysis")
print("=" * 60)

results = {}
for species_key, species_data in species_info.items():
    result = analyze_species(species_key, species_data)
    if result:
        results[species_key] = result

# Create summary comparison
print("\n" + "=" * 60)
print("MULTI-SPECIES COMPARISON SUMMARY")
print("=" * 60)

if results:
    summary_df = pd.DataFrame(results).T
    print("\nSpecies comparison:")
    print(f"{'Species':<20} {'Sample Size':<12} {'Fold Change':<12} {'P-value':<12} {'Significance':<15}")
    print("-" * 75)

    for species_key, result in results.items():
        significance = "Significant" if result['p_value'] < 0.05 else "Not significant"
        print(f"{result['species']:<20} {result['sample_size']:<12} {result['fold_change']:<12.2f} {result['p_value']:<12.2e} {significance:<15}")

    print(f"\nFiles generated in: {output_dir}")
    print("- PNG and PDF plots for each species")
    print("- CSV data files for GraphPad Prism analysis:")
    print("  * kcat_data_for_graphpad_[species].csv (log10 transformed data)")
    print("  * kcat_raw_data_for_graphpad_[species].csv (raw kcat values)")
    print("  * kcat_categorical_data_for_graphpad_[species].csv (enzyme classification data)")
    print("  * kde_curves_for_graphpad_[species].csv (KDE density curves for Figure 1)")
    print("  * histogram_data_for_graphpad_[species].csv (histogram data for Figure 1)")
    print("- Species-specific enzyme classification results")

print("\nAnalysis complete!")