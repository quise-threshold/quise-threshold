
import pandas as pd
import numpy as np
import os
from IPython.display import display


# Function to calculate MOE for zero estimates based on total population
def calculate_zero_moe(total_population, average_weight=16):
    if total_population <= 4999:
        k_value = 4
    elif total_population <= 9999:
        k_value = 8
    elif total_population <= 19999:
        k_value = 10
    elif total_population <= 29999:
        k_value = 14
    elif total_population <= 49999:
        k_value = 18
    else:
        k_value = 22
    return 1.645 * np.sqrt(average_weight * k_value)

def calculate_indicators(df_dict):

    import os

    # Adding total population column
    df_totpop = df_dict['B01001'].copy()
    df_totpop = df_totpop[['GEOID', 'NAME', 'TITLE', 'ESTIMATE']]
    df_totpop = df_totpop[df_totpop['TITLE'] == 'Total:'].copy()
    df_totpop = df_totpop.rename(columns={'ESTIMATE': 'POP'})
    final_results = df_totpop[['GEOID', 'POP']]  
    df_totpop = final_results.copy()


    # Dictionary that links indicators to their respective tables
    indicator_table_map = {
        'POVTY': 'B17001',  # Poverty
        'HOCOB': 'B25070',  # Housing Cost Burden
        'NOHIG': 'B15002',  # No Highschool Diploma
        'NOHEA': 'B27001',  # No Health Insurance
        'AGE65': 'B01001',  # Age 65 and Older
        'AGE17': 'B01001',  # Age 17 and Younger
        'DISBL': 'B18101',  # Disability
        'SNGPH': 'B11012',  # Single-Parent Households
        'LANGU': 'C16001',  # Language Proficiency
        'MINRTY': 'B03002',     # Minorities (handled separately)
        'MUUNS': 'B25024',  # Multi-unit Structure
        'MOHOM': 'B25024',  # Mobile Homes
        'CROWD': 'B25014',  # Crowding
        'NOVEH': 'B25044'  # No Vehicle
    }

    title_criteria = {
        'POVTY': 'Income in the past 12 months below poverty level:',
        'HOCOB': ['40.0 to 49.9 percent', '50.0 percent or more'],
        'NOHIG': ['No schooling completed', 'Nursery to 4th grade', '5th and 6th grade', '7th and 8th grade', '9th grade', '10th grade', '11th grade', '12th grade, no diploma'],
        'NOHEA': ['No health insurance coverage'],
        'AGE65': ['65 and 66 years', '67 to 69 years', '70 to 74 years', '75 to 79 years', '80 to 84 years', '85 years and over'],
        'AGE17': ['Under 5 years', '5 to 9 years', '10 to 14 years', '15 to 17 years'],
        'DISBL': 'With a disability',
        'SNGPH': 'With children of the householder under 18 years',
        'LANGU': 'Speak English less than "very well"',
        'MINRTY': ['Total:', 'White alone'], #actually not important -> handled differently below based separate minority groups!
        'MUUNS': ['10 to 19', '20 to 49', '50 or more'],
        'MOHOM': 'Mobile home',
        'CROWD': ['1.01 to 1.50 occupants per room', '1.51 to 2.00 occupants per room', '2.01 or more occupants per room'],
        'NOVEH': 'No vehicle available'
    }

    # Process each indicator except minorities (handled separately)
    for indicator, table_name in indicator_table_map.items():

        print(f"Processing {indicator}...")
        
        # Get the correct table for the indicator
        df = df_dict[table_name]

        if indicator == 'MINRTY':
            df_minrty = df_dict[table_name].copy()
            df_minrty = df_minrty[df_minrty['ORDER'].isin(list(range(4, 10)) + list(range(13, 20)))]

            minority_estimates = []

            for idx, (geoid, group) in enumerate(df_minrty.groupby('GEOID')):
                group = group.sort_values('ORDER')

                # Only print the first two validation examples
                if idx < 2:
                    print(f"\n--- Raw rows for GEOID: {geoid} (ORDERs 4–9 and 13–19) ---")
                    print(group[['ORDER', 'TITLE', 'ESTIMATE'] + [f'Var_Rep{i}' for i in range(1, 6)]])

                minority_row = {'GEOID': geoid}
                minority_row['ESTIMATE'] = group['ESTIMATE'].sum()
                for i in range(1, 81):
                    minority_row[f'Var_Rep{i}'] = group[f'Var_Rep{i}'].sum()

                if idx < 2:
                    print(f"\n--- Summed MINRTY row for GEOID: {geoid} ---")
                    print({k: minority_row[k] for k in ['GEOID', 'ESTIMATE'] + [f'Var_Rep{i}' for i in range(1, 6)]})

                minority_estimates.append(minority_row)

            summed_df = pd.DataFrame(minority_estimates)
        elif indicator == 'SNGPH':
            df_sngph = df_dict[table_name].copy()

            # These correspond to:
            # ORDER 10: Female householder with children < 18
            # ORDER 15: Male householder with children < 18
            df_sngph = df_sngph[df_sngph['ORDER'].isin([10, 15])]

            sngph_estimates = []

            for idx, (geoid, group) in enumerate(df_sngph.groupby('GEOID')):
                group = group.sort_values('ORDER')

                # Only print the first two validation examples
                if idx < 2:
                    print(f"\n--- Raw rows for GEOID: {geoid} (ORDERs 10 & 15) ---")
                    print(group[['ORDER', 'TITLE', 'ESTIMATE'] + [f'Var_Rep{i}' for i in range(1, 6)]])

                sngph_row = {'GEOID': geoid}
                sngph_row['ESTIMATE'] = group['ESTIMATE'].sum()
                for i in range(1, 81):
                    sngph_row[f'Var_Rep{i}'] = group[f'Var_Rep{i}'].sum()

                if idx < 2:
                    print(f"\n--- Summed SNGPH row for GEOID: {geoid} ---")
                    print({k: sngph_row[k] for k in ['GEOID', 'ESTIMATE'] + [f'Var_Rep{i}' for i in range(1, 6)]})

                sngph_estimates.append(sngph_row)

            summed_df = pd.DataFrame(sngph_estimates)

        else:
            # Get the correct table for the indicator
            df = df_dict[table_name]

            # Filter the rows based on the 'TITLE' column
            titles = title_criteria[indicator]
            filtered_df = df[df['TITLE'].isin(titles) if isinstance(titles, list) else df['TITLE'] == titles]
            sample_geoids = filtered_df['GEOID'].drop_duplicates().sample(5, random_state=1).tolist()
            for geoid in sample_geoids:
                print(f"\n--- Raw data for {indicator}, GEOID: {geoid} ---")
                print(filtered_df[filtered_df['GEOID'] == geoid][['TITLE', 'ESTIMATE'] + [f'Var_Rep{i}' for i in range(1, 6)]].to_string(index=False))

            # Perform groupby operation to sum 'ESTIMATE' and all 'Var_Rep' columns
            summed_df = filtered_df.groupby('GEOID').agg(
                {f'Var_Rep{i}': 'sum' for i in range(1, 81)} | {'ESTIMATE': 'sum'}
            ).reset_index()

        print(f"\n--- Aggregated values for {indicator} ---")
        print(summed_df[summed_df['GEOID'].isin(sample_geoids)][['GEOID', 'ESTIMATE'] + [f'Var_Rep{i}' for i in range(1, 5)]])
        manual_sum = filtered_df[['ESTIMATE'] + [f'Var_Rep{i}' for i in range(1, 5)]].sum()
        grouped_sum = summed_df[['ESTIMATE'] + [f'Var_Rep{i}' for i in range(1, 5)]].sum()
        print("\n🔍 Manual sum vs Grouped sum:")
        print(pd.concat([manual_sum, grouped_sum], axis=1, keys=['Manual', 'Grouped']))
        print(f"Original rows: {len(filtered_df)}, Summed rows: {len(summed_df)}")

        # variance calculation
        columns = [f'Var_Rep{i}' for i in range(1, 81)]
        valid_rows = summed_df[columns].notna().all(axis=1)
        squared_differences = (summed_df.loc[valid_rows, columns].values - summed_df.loc[valid_rows, 'ESTIMATE'].values.reshape(-1, 1)) ** 2
        variance = (4 / 80) * squared_differences.sum(axis=1)

        # Replace inf values with NaN
        variance = np.where(np.isinf(variance), np.nan, variance)

        # Calculate the Margin of Error (MOE)
        summed_df.loc[valid_rows, 'MOE'] = 1.645 * np.sqrt(variance)

        # Apply special handling for cases where ESTIMATE == 0
        summed_df = summed_df.merge(df_totpop, on='GEOID', how='left')
        zero_estimates = summed_df['ESTIMATE'] == 0
        summed_df.loc[zero_estimates, 'MOE'] = summed_df[zero_estimates].apply(
                lambda row: calculate_zero_moe(row['POP']), axis=1
            )

        # Calculate bounds
        summed_df[f'E_{indicator}'] = summed_df['ESTIMATE']
        summed_df[f'L_{indicator}'] = summed_df['ESTIMATE'] - summed_df['MOE']
        summed_df[f'U_{indicator}'] = summed_df['ESTIMATE'] + summed_df['MOE']

        path = os.path.join(os.path.dirname(os.getcwd()), '0_data', 'output/demo', 'calculated_indicator_tables')
    
        if not os.path.exists(path):
            os.makedirs(path)
        
        file_name = indicator + '.csv'
        file_path = os.path.join(path, file_name) 
        summed_df.to_csv(file_path, index=False)

        # Select relevant columns to merge
        merged_cols = ['GEOID', f'E_{indicator}', f'L_{indicator}', f'U_{indicator}']
        if final_results.empty:
            final_results = summed_df[merged_cols]
        else:
            final_results = final_results.merge(summed_df[merged_cols], on='GEOID', how='left')

    
    return final_results



def check_moe(B01001):

    # Function to calculate MOE for zero estimates
    def calculate_zero_moe(total_population, average_weight=16):
        if total_population <= 4999:
            k_value = 4
        elif total_population <= 9999:
            k_value = 8
        elif total_population <= 19999:
            k_value = 10
        elif total_population <= 29999:
            k_value = 14
        elif total_population <= 49999:
            k_value = 18
        else:
            k_value = 22
        moe = 1.645 * np.sqrt(average_weight * k_value)
        return moe

    # Assuming B17001 is a DataFrame containing the necessary columns
    # Adding total population column
    df_totpop = B01001.copy()
    df_totpop = df_totpop[df_totpop['TITLE'] == 'Total:'].copy()
    df_totpop = df_totpop.rename(columns={'ESTIMATE': 'TOTPOP'})

    # Calculate the variance (assuming Var_Rep columns exist in df_totpop)
    columns = [f'Var_Rep{i}' for i in range(1, 81)]
    valid_rows = df_totpop[columns].notna().all(axis=1)
    squared_differences = (df_totpop.loc[valid_rows, columns].values - df_totpop.loc[valid_rows, 'TOTPOP'].values.reshape(-1, 1)) ** 2
    variance = (4 / 80) * squared_differences.sum(axis=1)

    # Replace inf values with NaN
    variance = np.where(np.isinf(variance), np.nan, variance)

    # Calculate the Margin of Error (MOE)
    df_totpop.loc[valid_rows, 'MOE_calculated'] = 1.645 * np.sqrt(variance)

    # Calculate MOE for zero estimates (if any)
    zero_estimates = df_totpop['TOTPOP'] == 0
    df_totpop.loc[zero_estimates, 'MOE_calculated'] = df_totpop.loc[zero_estimates, 'TOTPOP'].apply(calculate_zero_moe)

    # Compare with the existing MOE
    df_totpop['DIFFERENCE'] = df_totpop['MOE_calculated'] - df_totpop['MOE']

    df_totpop = df_totpop[['GEOID', 'TOTPOP', 'MOE', 'MOE_calculated', 'DIFFERENCE']]

    return df_totpop

    

def calculate_denominators(df_dict):
    result = pd.DataFrame()  # Empty dataframe to store the results

    # Example dictionary and dataframe dictionary
    data_dict = {
        'TOTHH': 'B11012',  # Total Households
        'TOTHU': 'B25024',  # Total Housing Units
        'TOTPOP': 'B01001'  # Total Population
}
    
    # Loop over each indicator and its corresponding dataframe key
    for indicator, dataset_name in data_dict.items():
        df = df_dict[dataset_name].copy()
        
        # Filter rows where TITLE is "Total:"
        df_total = df[df['TITLE'] == 'Total:'].copy()

        path = os.path.join(os.path.dirname(os.getcwd()), '0_data', 'output/demo', 'denominator_tables')
    
        if not os.path.exists(path):
            os.makedirs(path)
        
        file_name = indicator + '.csv'
        file_path = os.path.join(path, file_name) 
        df_total.to_csv(file_path, index=False)
        
        # Rename the columns for estimate and MOE with indicator-specific names
        df_total.rename(columns={
            'ESTIMATE': f'E_{indicator}', 
            'MOE': f'MOE_{indicator}'
        }, inplace=True)
        
        # Calculate lower and upper bounds for MOE
        df_total[f'L_{indicator}'] = df_total[f'E_{indicator}'] - df_total[f'MOE_{indicator}']
        df_total[f'U_{indicator}'] = df_total[f'E_{indicator}'] + df_total[f'MOE_{indicator}']
        
        # Keep only the necessary columns (GEOID, estimate, lower, upper)
        df_total = df_total[['GEOID', f'E_{indicator}', f'L_{indicator}', f'U_{indicator}']]
        
        # Merge the current result with the new data
        if result.empty:
            result = df_total
        else:
            result = result.merge(df_total, on='GEOID', how='outer')
    
    return result