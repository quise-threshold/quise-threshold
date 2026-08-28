
import requests
import os
from tqdm import tqdm
import pandas as pd
import os
import zipfile

#function to download replicate tables for txt list of tables
def download_replicates(list_of_tables, year):

    # creating path variables for quick access
    path = os.path.join(os.path.dirname(os.getcwd()))
    ipath = os.path.join(path,'0_data', 'input')


    # Read the state codes and table names from the text files
    with open(os.path.join(ipath, 'FIPS_states.txt'), 'r') as f:
        state_codes = f.read().splitlines()

    table_names = list_of_tables

    # The directory where you want to save the zip files
    save_dir = os.path.join(path,'0_data','gitignore', f'{year}_zip_archives')

    # Create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Base URL for downloading the files
    base_url = f"https://www2.census.gov/programs-surveys/acs/replicate_estimates/{year}/data/5-year/140/"

    # Directory to save the downloaded files
    output_directory = os.path.join(path, '0_data', 'gitignore', f'{year}_zip_archives')

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    successful_downloads = 0
    failed_downloads = []

    for table in table_names:
        for state_code in tqdm(state_codes, desc=f"Downloading {table}"):
            state_code = state_code.strip()
            filename = f"{table}_{state_code}.csv.zip"
            url = base_url + filename

            try:
                response = requests.get(url)
                response.raise_for_status()

                filepath = os.path.join(output_directory, filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                successful_downloads += 1

            except requests.exceptions.HTTPError:
                failed_downloads.append(filename)

    # Reporting at the end
    print(f"Successfully downloaded {successful_downloads} files.")
    if failed_downloads:
        print(f"Failed to download the following files:")
        for file in failed_downloads:
            print(file)


def combine_states(year):
    # creating path variables for quick access
    path = os.path.join(os.path.dirname(os.getcwd()))
    opath = os.path.join(path,'0_data','gitignore',f'{year}_us_combined_tables')
    
    if not os.path.exists(opath):
        os.makedirs(opath)

    # Base directory where the CSV.ZIP files are stored
    base_dir = os.path.join(path,'0_data','gitignore',f'{year}_zip_archives')

    # List all ZIP files in the directory
    zip_files = [f for f in os.listdir(base_dir) if f.endswith('.csv.zip')]

    # Dictionary to hold combined data for each table
    combined_data = {}

    # Process each ZIP file
    for zip_file in zip_files:
        # Extract the table name from the file name (assuming the format is TableName_State.csv.zip)
        table_name = zip_file.split('_')[0]

        # Read the ZIP file and extract the CSV
        with zipfile.ZipFile(os.path.join(base_dir, zip_file), 'r') as z:
            # Assuming there is only one CSV file in each ZIP archive
            csv_file = z.namelist()[0]
            
            try:
                df = pd.read_csv(z.open(csv_file), encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(z.open(csv_file), encoding='latin-1')
                except UnicodeDecodeError:
                    df = pd.read_csv(z.open(csv_file), encoding='ISO-8859-1', encoding_errors='replace')
            
            # If the table already exists in our combined data, append the new data
            if table_name in combined_data:
                combined_data[table_name] = pd.concat([combined_data[table_name], df])
            else:
                combined_data[table_name] = df

    # Save each combined DataFrame to a CSV file
    for table_name, df in combined_data.items():
        output_file = os.path.join(opath, f'{table_name}.csv')
        df.to_csv(output_file, index=False)
        print(f'Combined data for {table_name} saved to {output_file}')

    # After combining the data in the combined_data dictionary
    for table_name, df in combined_data.items():
        # This will create a variable with the same name as the table_name and assign the dataframe to it
        globals()[table_name] = df


# load replicate tables into python environment
def load_replicates(year):
    # Path to the data directory
    data_directory = os.path.join(os.path.dirname(os.getcwd()), '0_data', 'gitignore', f'{year}_us_combined_tables')

    # List to store table names
    table_names = []

    # Dictionary to store dataframes
    dataframes = {}

    # Get the list of CSV files in the directory
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            table_name = filename.replace('.csv', '')
            table_names.append(table_name)

            csv_path = os.path.join(data_directory, filename)
            dataframes[table_name] = pd.read_csv(csv_path, dtype={'GEOID': str})

    return dataframes, table_names