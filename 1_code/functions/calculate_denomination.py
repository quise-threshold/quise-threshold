
import numpy as np


def estimate_denomination(df_indicators, df_denominators):
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

    # Loop through each indicator and corresponding denominator, only for 'E_' prefix
    for indicator, denominator in indicators_denominators.items():
        indicator_col = f'E_{indicator}'
        denominator_col = f'E_{denominator}'

        # Check if both columns exist in their respective dataframes
        if indicator_col in df_indicators.columns and denominator_col in df_denominators.columns:
            # Create the ratio column name with 'D_' prefix
            ratio_col_name = f'D_{indicator}'
            
            # Divide the indicator by the denominator, setting result to 0 when the denominator is zero
            df_indicators[ratio_col_name] = np.where(
                df_denominators[denominator_col] == 0, 
                np.nan,  # Set to NaN if denominator is zero
                df_indicators[indicator_col] / df_denominators[denominator_col] * 100
            )
            # Assert that all values are within the range of 0 to 100
            invalid_rows = df_indicators[(df_indicators[ratio_col_name] < 0) | (df_indicators[ratio_col_name] > 100)]
            assert invalid_rows.empty, f"Out of range values found in {ratio_col_name}:\n{invalid_rows[['GEOID', ratio_col_name]]}"

    # After processing all indicators, print confirmation that all values are in range
    print("All denominated indicators (D_ prefixed) are within the range of 0 to 100.")

    # Return the updated dataframe with the denominated columns
    return df_indicators
