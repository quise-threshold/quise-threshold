def class_stability(df_moe):
    import pandas as pd
    import numpy as np

    estimate_col = 'NOMINAL'
    indicator = 'Index_Rep'
    geoid_col = 'GEOID'  # Use GEOID for alignment
    
    # Get the columns corresponding to the 80 replicates
    columns = [col for col in df_moe.columns if col.startswith(indicator)]
    #print("Replicate columns:", columns)
    
    # Step 1: Calculate the variance and MOE based on replicates (for the values, not ranks)
    squared_differences = (df_moe[columns] - df_moe[estimate_col].values.reshape(-1, 1)) ** 2
    variance = (4 / 80) * squared_differences.sum(axis=1, min_count=1)
    df_moe['MOE'] = 1.645 * np.sqrt(variance)  # 90% confidence interval

    # Step 2: Rank the 'NOMINAL' column and all 80 replicates, ensure GEOID alignment
    rank_columns = [f'Rank_{col}' for col in [estimate_col] + columns]
    ranks = pd.concat([df_moe[geoid_col], pd.concat([df_moe[col].rank(method='min').rename(f'Rank_{col}') for col in [estimate_col] + columns], axis=1)], axis=1)

    # Step 3: Define quantile classes with specific breakpoints (quartiles)
    quantiles = [0, 0.25, 0.5, 0.75, 1]

    # Quantile classification for 'NOMINAL' and all 80 replicates, ensure GEOID alignment
    quantiles_df = pd.concat([df_moe[geoid_col], pd.concat([pd.qcut(ranks[f'Rank_{col}'], quantiles, labels=False).rename(f'Quantile_{col}') 
                                for col in [estimate_col] + columns], axis=1)], axis=1)

    # Step 4: Merge ranks and quantiles based on 'GEOID' to ensure alignment
    df_moe = pd.merge(df_moe, ranks, on=geoid_col, how='left')
    df_moe = pd.merge(df_moe, quantiles_df, on=geoid_col, how='left')

    # Step 5: Calculate confidence intervals for the 'NOMINAL' estimate
    df_moe[f'Lower_CI_{estimate_col}'] = df_moe[estimate_col] - df_moe['MOE']
    df_moe[f'Upper_CI_{estimate_col}'] = df_moe[estimate_col] + df_moe['MOE']

    # Step 6: Quantile classification for the confidence intervals
    lower_upper_quantiles = pd.concat([
        pd.qcut(df_moe[f'Lower_CI_{estimate_col}'], quantiles, labels=False).rename(f'Lower_Quantile_{estimate_col}'),
        pd.qcut(df_moe[f'Upper_CI_{estimate_col}'], quantiles, labels=False).rename(f'Upper_Quantile_{estimate_col}')
    ], axis=1)

    # Step 7: Merge the confidence interval quantiles back into the DataFrame
    df_moe = pd.concat([df_moe, lower_upper_quantiles], axis=1)

    # Step 8: Check the stability by comparing each replicate's quantile to NOMINAL's quantile
    stability_cols = [f'Quantile_{col}' for col in columns]  # Replicate quantile columns
    df_moe['Stability_NOMINAL'] = (df_moe[stability_cols] == df_moe[f'Quantile_{estimate_col}'].values.reshape(-1, 1)).all(axis=1)

    # Step 9: Calculate how many moved by 1, 2, or 3 quantile classes
    quantile_diffs = df_moe[stability_cols].sub(df_moe[f'Quantile_{estimate_col}'], axis=0)

    move_1_class = (quantile_diffs.abs() == 1).sum(axis=1) / len(columns)
    move_2_class = (quantile_diffs.abs() == 2).sum(axis=1) / len(columns)
    move_3_class = (quantile_diffs.abs() == 3).sum(axis=1) / len(columns)

    # Calculate percentage of rows for 1, 2, and 3 quantile class movement
    move_1_class_percentage = (move_1_class > 0).mean() * 100
    move_2_class_percentage = (move_2_class > 0).mean() * 100
    move_3_class_percentage = (move_3_class > 0).mean() * 100

    # Step 10: Calculate the percentage of tracts that stayed in the same quantile
    stay_in_quantile_percentage = df_moe['Stability_NOMINAL'].mean() * 100

    rank_nominal_col = f'Rank_{estimate_col}'  # Column for the rank of NOMINAL

    # Step 1: Calculate the absolute rank differences between NOMINAL and each replicate
    abs_rank_diffs = df_moe[rank_columns].sub(df_moe[rank_nominal_col], axis=0).abs()

    # Step 2: Calculate the mean absolute difference for each tract (row)
    df_moe['Mean_Absolute_Difference'] = abs_rank_diffs.mean(axis=1)

    # Step 3: Calculate the overall mean of the mean values across all tracts
    overall_mean_of_means = df_moe['Mean_Absolute_Difference'].mean()

    # Step 4: Calculate the standard deviation of the mean values across all tracts
    overall_std_of_means = df_moe['Mean_Absolute_Difference'].std()

    # Print results
    print(f'Mean of the mean absolute rank differences across all tracts: {overall_mean_of_means:.4f}')
    print(f'Standard deviation of the mean absolute rank differences across all tracts: {overall_std_of_means:.4f}')

    # Display results
    #print(f'Stability Rate for NOMINAL Quantiles (No Movement): {stay_in_quantile_percentage:.2f}%')
    #print(f'Percentage of tracts that move by exactly 1 quantile class: {move_1_class_percentage:.2f}%')
    #print(f'Percentage of tracts that move by exactly 2 quantile classes: {move_2_class_percentage:.2f}%')
    #print(f'Percentage of tracts that move by exactly 3 quantile classes: {move_3_class_percentage:.2f}%')

    return df_moe


import pandas as pd

def count_designation(df, threshold):
    # Step 1: Create a new column 'nom_desig' where NOMINAL >= 0.99 is 1 and 0 otherwise
    df['nom_desig'] = (df['NOMINAL'] >= threshold).astype(int)
    number_of_designated_tracts = df['nom_desig'].sum()  # Total designated by estimate

    # Step 2: Create a list of all columns that need designation checks
    cols_to_check = ['NOMINAL'] + [f'Index_Rep{i}' for i in range(1, 81)]

    # Step 3: Count how often the tract is designated across all columns
    df['count_design'] = (df[cols_to_check] >= threshold).sum(axis=1)

    # Step 4: Calculate total number of tracts
    total_tracts = len(df)  # Total tracts in the DataFrame

    # Step 5: Calculate number of tracts that are designated based on estimate
    designated_by_estimate = number_of_designated_tracts

    # Step 6: Count tracts that are always designated across all replicates
    always_designated_count = len(df[df['count_design'] == 81])  # Count of always designated

    # Step 7: Count tracts that are designated at least once among replicates
    at_least_once_designated = len(df[df['count_design'] > 0])  # Count of tracts designated at least once

    # Step 8: Count tracts designated among replicates with more than 8 designations
    designated_with_eight_or_more = len(df[df['count_design'] >= 8])

    # Step 9: Calculate the percentage of always designated tracts from the designated_by_estimate
    percentage_always_designated = (always_designated_count / designated_by_estimate) * 100 if designated_by_estimate > 0 else 0

    df['count_design_filtered'] = df['count_design'].apply(lambda x: x if x >= 8 else None)

    # Step 10: Print the summary statistics
    print("Summary Statistics without Confidence Interval for Designation:")
    print(f"Total number of tracts: {total_tracts}")
    print(f"Number of tracts designated based on Estimate: {designated_by_estimate}")
    print(f"Number of tracts designated at least once across replicates: {at_least_once_designated}")
    print(f"Number of tracts always designated across replicates: {always_designated_count}")
    print(f"Percentage of always designated among the designated by estimate: {percentage_always_designated:.2f}%")

    print("Considering Designation Confidence with 8 or more designations:")
    print(f"Number of tracts designated among replicates with more than 8 designations: {designated_with_eight_or_more}")




    return df
    
import numpy as np
import pandas as pd

def analyze_moe_regions(df, nominal_col='NOMINAL', var_rep_prefix='Index_Rep',
                       n_replicates=80, reference_line=0.75, scale_to_100=True):
    """
    Calculate descriptive statistics for MOE plot regions without modifying the plotting function.
    Uses the same logic as the plotting function to ensure consistency.
    """
    
    # Make a copy to avoid modifying the original
    df_moe = df.copy()
    
    # Scale factor for display
    scale_factor = 100 if scale_to_100 else 1
    threshold = reference_line * scale_factor
    
    # Get all replicate columns
    rep_cols = [f"{var_rep_prefix}{i}" for i in range(1, n_replicates+1) if f"{var_rep_prefix}{i}" in df_moe.columns]
    
    # Apply the census formula for calculating variance (same as plot function)
    squared_differences = np.zeros(len(df_moe))
    for rep_col in rep_cols:
        squared_diff = (df_moe[rep_col] - df_moe[nominal_col]) ** 2
        squared_differences += squared_diff
    
    # Standard census formula for variance
    variance = (4 / n_replicates) * squared_differences
    
    # Calculate MOEs for different confidence levels
    df_moe['MOE_65'] = 0.93 * np.sqrt(variance)  # 65% CI
    df_moe['MOE_90'] = 1.645 * np.sqrt(variance)  # 90% CI
    df_moe['MOE_99'] = 2.576 * np.sqrt(variance)  # 99% CI
    
    # Calculate confidence intervals
    df_moe['lower_65'] = df_moe[nominal_col] - df_moe['MOE_65']
    df_moe['upper_65'] = df_moe[nominal_col] + df_moe['MOE_65']
    df_moe['lower_90'] = df_moe[nominal_col] - df_moe['MOE_90']
    df_moe['upper_90'] = df_moe[nominal_col] + df_moe['MOE_90']
    df_moe['lower_99'] = df_moe[nominal_col] - df_moe['MOE_99']
    df_moe['upper_99'] = df_moe[nominal_col] + df_moe['MOE_99']
    
    # Clean the dataframe
    check_columns = [nominal_col, 'lower_65', 'upper_65', 'lower_90', 'upper_90', 'upper_99', 'lower_99']
    df_clean = df_moe.dropna(subset=check_columns).copy()
    
    # Scale values to 0-100 range if requested
    if scale_to_100:
        for col in [nominal_col, 'lower_65', 'upper_65', 'lower_90', 'upper_90',
                  'MOE_65', 'MOE_90','upper_99', 'lower_99', 'MOE_99']:
            if col in df_clean.columns:
                df_clean[col] *= scale_factor
    
    # Sort by nominal value
    df_sorted = df_clean.sort_values(by=nominal_col)
    
    # Define helper to compute empirical designation probabilities per bin (same as plot function)
    def compute_binwise_designation_prob(df, threshold, bins=np.arange(0, 101, 1)):
        df = df.copy()
        df['bin'] = pd.cut(df[nominal_col], bins=bins, labels=bins[:-1], include_lowest=True)
        
        bin_summary = df.groupby('bin').agg(
            n=('GEOID', 'count'),
            prob_65_up=('upper_65', lambda x: (x > threshold).mean()),
            prob_90_up=('upper_90', lambda x: (x > threshold).mean()),
            prob_99_up=('upper_99', lambda x: (x > threshold).mean()),
            prob_65_lo=('lower_65', lambda x: (x < threshold).mean()),
            prob_90_lo=('lower_90', lambda x: (x < threshold).mean()),
            prob_99_lo=('lower_99', lambda x: (x < threshold).mean()),
        ).reset_index()
        
        bin_summary['bin'] = bin_summary['bin'].astype(float)
        return bin_summary

    # Calculate empirical designation probabilities
    bin_df = compute_binwise_designation_prob(df_sorted, threshold)

    # Get crossings from smoothed empirical probabilities (same logic as plot)
    def get_crossing_x(bin_df, colname, cutoff, side='left'):
        temp = bin_df[bin_df[colname] >= cutoff] if 'up' in colname else bin_df[bin_df[colname] > cutoff]
        if len(temp) == 0:
            return None
        return temp['bin'].min() if side == 'left' else temp['bin'].max()

    # Calculate crossing points (same as plot function)
    upper_cross_x_99 = get_crossing_x(bin_df, 'prob_99_up', 0.5, side='left')
    upper_cross_x_90 = get_crossing_x(bin_df, 'prob_90_up', 0.5, side='left')
    upper_cross_x_65 = get_crossing_x(bin_df, 'prob_65_up', 0.5, side='left')

    lower_cross_x_65 = get_crossing_x(bin_df, 'prob_65_lo', 0.5, side='right')
    lower_cross_x_90 = get_crossing_x(bin_df, 'prob_90_lo', 0.5, side='right')
    lower_cross_x_99 = get_crossing_x(bin_df, 'prob_99_lo', 0.5, side='right')
    
    # Fixed boundary at x=75 (same as plot)
    x_boundary = 75
    
    # Calculate statistics for each region
    total_tracts = len(df_sorted)
    
    # Region 1: Never designated (0 to first crossing)
    never_designated = df_sorted[df_sorted[nominal_col] <= upper_cross_x_99]
    
    # Region 2: 99% CI false negative 
    fn_99 = df_sorted[(df_sorted[nominal_col] > upper_cross_x_99) & 
                     (df_sorted[nominal_col] <= upper_cross_x_90)]
    
    # Region 3: 90% CI false negative
    fn_90 = df_sorted[(df_sorted[nominal_col] > upper_cross_x_90) & 
                     (df_sorted[nominal_col] <= upper_cross_x_65)]
    
    # Region 4: Left uncertainty zone
    left_uncertain = df_sorted[(df_sorted[nominal_col] > upper_cross_x_65) & 
                              (df_sorted[nominal_col] <= x_boundary)]
    
    # Region 5: Right uncertainty zone
    right_uncertain = df_sorted[(df_sorted[nominal_col] > x_boundary) & 
                               (df_sorted[nominal_col] <= lower_cross_x_65)]
    
    # Region 6: 90% CI false positive
    fp_90 = df_sorted[(df_sorted[nominal_col] > lower_cross_x_65) & 
                     (df_sorted[nominal_col] <= lower_cross_x_90)]
    
    # Region 7: 99% CI false positive
    fp_99 = df_sorted[(df_sorted[nominal_col] > lower_cross_x_90) & 
                     (df_sorted[nominal_col] <= lower_cross_x_99)]
    
    # Region 8: Always designated
    always_designated = df_sorted[df_sorted[nominal_col] > lower_cross_x_99]
    
    # Compile results
    region_stats = {
        'never_designated': {
            'count': len(never_designated),
            'percentage': len(never_designated) / total_tracts * 100,
            'range': f"0-{upper_cross_x_99:.0f}",
            'description': 'Never designated'
        },
        'fn_99': {
            'count': len(fn_99),
            'percentage': len(fn_99) / total_tracts * 100,
            'range': f"{upper_cross_x_99:.0f}-{upper_cross_x_90:.0f}",
            'description': '99% CI false negative'
        },
        'fn_90': {
            'count': len(fn_90),
            'percentage': len(fn_90) / total_tracts * 100,
            'range': f"{upper_cross_x_90:.0f}-{upper_cross_x_65:.0f}",
            'description': '90% CI false negative'
        },
        'left_uncertain': {
            'count': len(left_uncertain),
            'percentage': len(left_uncertain) / total_tracts * 100,
            'range': f"{upper_cross_x_65:.0f}-{x_boundary}",
            'description': 'Left uncertainty zone'
        },
        'right_uncertain': {
            'count': len(right_uncertain),
            'percentage': len(right_uncertain) / total_tracts * 100,
            'range': f"{x_boundary}-{lower_cross_x_65:.0f}",
            'description': 'Right uncertainty zone'
        },
        'fp_90': {
            'count': len(fp_90),
            'percentage': len(fp_90) / total_tracts * 100,
            'range': f"{lower_cross_x_65:.0f}-{lower_cross_x_90:.0f}",
            'description': '90% CI false positive'
        },
        'fp_99': {
            'count': len(fp_99),
            'percentage': len(fp_99) / total_tracts * 100,
            'range': f"{lower_cross_x_90:.0f}-{lower_cross_x_99:.0f}",
            'description': '99% CI false positive'
        },
        'always_designated': {
            'count': len(always_designated),
            'percentage': len(always_designated) / total_tracts * 100,
            'range': f"{lower_cross_x_99:.0f}-100",
            'description': 'Always designated'
        }
    }
    
    # Calculate summary statistics
    total_false_negative = region_stats['fn_99']['count'] + region_stats['fn_90']['count']
    total_false_positive = region_stats['fp_90']['count'] + region_stats['fp_99']['count']
    total_uncertainty = (region_stats['left_uncertain']['count'] + 
                        region_stats['right_uncertain']['count'])
    total_stable = (region_stats['never_designated']['count'] + 
                   region_stats['always_designated']['count'])
    
    # Print comprehensive summary
    print("="*80)
    print("MOE PLOT REGION ANALYSIS SUMMARY")
    print("="*80)
    print(f"Threshold: {reference_line}")
    print(f"Total tracts analyzed: {total_tracts:,}")
    print()
    
    print("DETAILED REGION BREAKDOWN:")
    print("-" * 50)
    for region_name, stats in region_stats.items():
        print(f"{stats['description']:.<25} {stats['count']:>8,} ({stats['percentage']:>5.1f}%) [{stats['range']}]")
    
    print("\n" + "="*50)
    print("SUMMARY BY CLASSIFICATION TYPE:")
    print("="*50)
    print(f"Stable classification:      {total_stable:>8,} ({total_stable/total_tracts*100:>5.1f}%)")
    print(f"  - Never designated        {region_stats['never_designated']['count']:>8,} ({region_stats['never_designated']['percentage']:>5.1f}%)")
    print(f"  - Always designated       {region_stats['always_designated']['count']:>8,} ({region_stats['always_designated']['percentage']:>5.1f}%)")
    print()
    print(f"Uncertainty zones:          {total_uncertainty:>8,} ({total_uncertainty/total_tracts*100:>5.1f}%)")
    print(f"  - Left uncertainty        {region_stats['left_uncertain']['count']:>8,} ({region_stats['left_uncertain']['percentage']:>5.1f}%)")
    print(f"  - Right uncertainty       {region_stats['right_uncertain']['count']:>8,} ({region_stats['right_uncertain']['percentage']:>5.1f}%)")
    print()
    print(f"False negative prone:       {total_false_negative:>8,} ({total_false_negative/total_tracts*100:>5.1f}%)")
    print(f"  - 99% CI false negative   {region_stats['fn_99']['count']:>8,} ({region_stats['fn_99']['percentage']:>5.1f}%)")
    print(f"  - 90% CI false negative   {region_stats['fn_90']['count']:>8,} ({region_stats['fn_90']['percentage']:>5.1f}%)")
    print()
    print(f"False positive prone:       {total_false_positive:>8,} ({total_false_positive/total_tracts*100:>5.1f}%)")
    print(f"  - 90% CI false positive   {region_stats['fp_90']['count']:>8,} ({region_stats['fp_90']['percentage']:>5.1f}%)")
    print(f"  - 99% CI false positive   {region_stats['fp_99']['count']:>8,} ({region_stats['fp_99']['percentage']:>5.1f}%)")
    print()
    
    print("KEY INSIGHTS:")
    print("-" * 20)
    if total_false_positive > 0:
        print(f"• False negative/false positive ratio: {total_false_negative/total_false_positive:.2f}:1")
    
    total_at_risk = total_false_negative + total_false_positive + total_uncertainty
    print(f"• Total tracts at risk of misclassification: {total_at_risk:,} ({total_at_risk/total_tracts*100:.1f}%)")
    print(f"• Stable vs uncertain classification ratio: {total_stable/total_at_risk:.2f}:1")
    
    # Basic designation statistics for context
    designated_by_nominal = len(df_sorted[df_sorted[nominal_col] >= threshold])
    not_designated_by_nominal = total_tracts - designated_by_nominal
    
    print(f"\nCONTEXT:")
    print(f"• Tracts designated by point estimate: {designated_by_nominal:,} ({designated_by_nominal/total_tracts*100:.1f}%)")
    print(f"• Tracts not designated by point estimate: {not_designated_by_nominal:,} ({not_designated_by_nominal/total_tracts*100:.1f}%)")
    
    return region_stats

# Usage:
# region_stats = analyze_moe_regions(df, reference_line=0.75)