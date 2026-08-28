import os
import pandas as pd


def read_from_csv(name, folder):

    path = os.path.join(os.path.dirname(os.getcwd()), '0_data', folder)
    file_name = name
    file_path = os.path.join(path, file_name) 
    df = pd.read_csv(file_path)

    return df


def save_to_csv(df, folder, filename):

    path = os.path.join(os.path.dirname(os.getcwd()), '0_data', folder)
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    file_name = filename
    file_path = os.path.join(path, file_name) 
    df.to_csv(file_path, index=False)

    print(f'File saved to {file_path}')

    return 