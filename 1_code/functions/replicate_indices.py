import os
import pandas as pd
import numpy as np

def construct_nominal_index(df, normalization, index_pct):
    # Define and calculate theme scores
    themes = {
        'THEME1': ['ND_HOCOB', 'ND_POVTY', 'ND_NOHIG', 'ND_NOHEA'],
        'THEME2': ['ND_AGE65', 'ND_AGE17', 'ND_DISBL', 'ND_SNGPH', 'ND_LANGU'],
        'THEME3': ['ND_MINRTY'],
        'THEME4': ['ND_MUUNS', 'ND_MOHOM', 'ND_CROWD', 'ND_NOVEH']
    }

    # Calculate the weighted sum for each theme
    for theme, components in themes.items():
        df[theme] = df[components].sum(axis=1) / len(components)

        # If index_pct is True, apply percentile normalization to each theme
        if index_pct == True:
            df[theme] = df[theme].rank(pct=True)

    # Calculate the total score from the themes
    df['S_NOMINAL'] = df[list(themes.keys())].sum(axis=1) / len(themes)
    
    if index_pct == True:
        # Rank the total score with the highest value as rank 1
        df['S_NOMINAL'] = df['S_NOMINAL'].rank(pct=True)

    # Rank the total score with the highest value as rank 1
    df['RS_NOMINAL'] = df['S_NOMINAL'].rank(ascending=False)

    return df

def calculate_index_80_times(df_nominal, calculated_indicator_path, denominator_table_map, indicator_table_map, indicators_denominators, normalization, index_pct):
    # Load the nominal file
    df_nominal = df_nominal[['GEOID', 'S_NOMINAL']].copy()

    if index_pct == True:
        df_nominal['S_NOMINAL'] = df_nominal['S_NOMINAL'].rank(pct=True)
    
    df_nominal.rename(columns={'S_NOMINAL': 'NOMINAL'}, inplace=True)

    # Loop through each replicate (Var_Rep1 to Var_Rep80)
    for i in range(1, 81):
        # Create a new dataframe for this replicate
        df_rep = pd.DataFrame({'GEOID': df_nominal['GEOID']})
        var_rep_1 = pd.DataFrame({'GEOID': df_nominal['GEOID']})

        # Load the denominator tables for this replicate
        df_denominators = pd.DataFrame({'GEOID': df_nominal['GEOID']})
        for denom, denom_table in denominator_table_map.items():

            path_denom = os.path.join(os.path.dirname(os.getcwd()), '0_data', 'output/demo', 'denominator_tables')
            table_file_path = os.path.join(path_denom, f'{denom}.csv')
            df_denom = pd.read_csv(table_file_path)

            # Extract the Var_Rep{i} column from the denominator table
            var_rep_col = f'Var_Rep{i}'
            denom_col_name = f'{denom}_Rep{i}'
            df_denom.rename(columns={var_rep_col: denom_col_name}, inplace=True)

            # Merge the denominators to the denominator dataframe
            df_denominators = df_denominators.merge(df_denom[['GEOID', denom_col_name]], on='GEOID', how='left')
            if i == 1:
                var_rep_1 = var_rep_1.merge(df_denominators, on='GEOID', how='left')

        # Loop through each indicator and table name
        for indicator, table_name in indicator_table_map.items():
            # Load the corresponding indicator table
            table_file_path = os.path.join(calculated_indicator_path, f'{indicator}.csv')
            df_indicator = pd.read_csv(table_file_path)

            # Get the Var_Rep{i} column for this indicator
            var_rep_col = f'Var_Rep{i}'
            indicator_col_name = f'{indicator}_Rep{i}'
            df_indicator.rename(columns={var_rep_col: indicator_col_name}, inplace=True)
            if i == 1:
             var_rep_1 = var_rep_1.merge(df_indicator[['GEOID', indicator_col_name]], on='GEOID', how='left')

            df_indicator = df_indicator.merge(df_denominators, on='GEOID', how='left')
            


            # Get the corresponding denominator
            denominator = indicators_denominators.get(indicator, None)
            if denominator is not None:
                denominator_col_name = f'{denominator}_Rep{i}'
                
                
                # Check if the denominator column exists after merging
                if denominator_col_name in df_indicator.columns:
                    # Denominate the indicator by the denominator
                    df_indicator[f'D_{indicator}'] = np.where(
                        df_indicator[denominator_col_name] == 0,  # Handle division by zero
                        np.nan,
                        df_indicator[indicator_col_name] / df_indicator[denominator_col_name] * 100
                    )
                else:
                    print(f'Denominator column {denominator_col_name} not found in the merged dataframe for indicator {indicator}.')
            else:
                print(f'No denominator found for indicator {indicator}.')

            # Add the denominated column to the replicate dataframe
            df_rep = df_rep.merge(df_indicator[['GEOID', f'D_{indicator}']], on='GEOID', how='left')

        # Normalize the indicators (assuming all D_* columns are normalized to create ND_* columns)
        d_columns = df_rep.columns[df_rep.columns.str.startswith('D_')]

        if normalization == 'minmax':
            for col in d_columns:
                df_rep[f'N{col}'] = 1 + ((df_rep[col] - df_rep[col].min()) / (df_rep[col].max() - df_rep[col].min())) * 99
        elif normalization == 'pct':
            for col in d_columns:
                df_rep[f'N{col}'] = df_rep[col].rank(pct=True)


        # Calculate the index for this replicate
        df_rep = construct_nominal_index(df_rep, normalization, index_pct)

        if i == 1:
            var_rep_1 = var_rep_1.merge(df_rep, on='GEOID', how='left')
            output_path = os.path.join(os.path.dirname(os.getcwd()), '0_data', 'output/demo', f'var_rep_1_{normalization}.csv')
            var_rep_1.to_csv(output_path, index=False)


        # Merge the calculated index into the nominal dataframe
        df_rep.rename(columns={'S_NOMINAL': f'Index_Rep{i}'}, inplace=True)
        df_nominal = df_nominal.merge(df_rep[['GEOID', f'Index_Rep{i}']], on='GEOID', how='left')
        

    return df_nominal

def construct_replicate_indices(df_nominal, normalization, index_pct):
   
    calculated_indicator_path = os.path.join(os.path.dirname(os.getcwd()), '0_data', 'output/demo', 'calculated_indicator_tables')

    # Define the table map for denominators
    denominator_table_map = {
        'TOTHH': 'B11012',  # Total Households
        'TOTHU': 'B25024',  # Total Housing Units
        'TOTPOP': 'B01001'  # Total Population
    }

    # Define the table map for indicators
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
        'MINRTY': 'minority_aggregated',  # Minorities (handled separately)
        'MUUNS': 'B25024',  # Multi-unit Structure
        'MOHOM': 'B25024',  # Mobile Homes
        'CROWD': 'B25014',  # Crowding
        'NOVEH': 'B25044'   # No Vehicle
    }

    # Define the list of indicators and corresponding denominators
    indicators_denominators = {
        'POVTY': 'TOTPOP',  # Poverty / Population
        'HOCOB': 'TOTHU',   # Housing Cost Burden / Housing Units
        'NOHIG': 'TOTPOP',  # No Highschool / Population
        'NOHEA': 'TOTPOP',  # No Health Insurance / Population
        'AGE65': 'TOTPOP',  # 65+ / Population
        'AGE17': 'TOTPOP',  # 17- / Population
        'DISBL': 'TOTPOP',  # Disability / Population
        'SNGPH': 'TOTHH',   # Single Parent Household / Universe Households
        'LANGU': 'TOTPOP',  # Language / Population
        'MINRTY': 'TOTPOP', # Minority / Population (allow larger than total population)
        'MUUNS': 'TOTHU',   # Multi Unit Structure / Housing Units
        'MOHOM': 'TOTHU',   # Mobile Homes / Universe Housing Units
        'CROWD': 'TOTHU',   # Crowding / Housing Units
        'NOVEH': 'TOTHU'    # No Vehicle / Universe Occupied housing units
    }

    # Call the function to calculate the index 80 times
    df_final_nominal = calculate_index_80_times(df_nominal, calculated_indicator_path, denominator_table_map, indicator_table_map, indicators_denominators, normalization, index_pct)

    return df_final_nominal

