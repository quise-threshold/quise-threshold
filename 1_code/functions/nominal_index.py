
def construct_nominal_index(df, normalization, index_pct):

    # normalize data columns startig with D_
    d_columns = df.columns[df.columns.str.startswith('D_')]

    if normalization == 'minmax':
        for col in d_columns:
            df[f'N{col}'] = 1 + ((df[col] - df[col].min()) / (df[col].max() - df[col].min())) * 99
    elif normalization == 'pct':
        for col in d_columns:
            df[f'N{col}'] = df[col].rank(pct=True)


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

    # Remove tracts with no population
    #df = df[df['TOTPOP'] > 0]

    # Calculate the total score from the themes
    df['S_NOMINAL'] = df[list(themes.keys())].sum(axis=1) / len(themes)

    if index_pct == True:
        #Rank the total score with the highest value as rank 1
        df['S_NOMINAL'] = df['S_NOMINAL'].rank(pct=True)

    # Rank the total score with the highest value as rank 1
    df['RS_NOMINAL'] = df['S_NOMINAL'].rank(ascending=False)

    

    return df




