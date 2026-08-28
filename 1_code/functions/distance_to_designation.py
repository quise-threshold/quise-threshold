import numpy as np
import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from csv_file import save_to_csv, read_from_csv
from worker import calculate_steps_to_nearest_nom_desig, process_tract, init_worker  # Import the function from worker
from libpysal.weights import Queen, shimbel

def calculate_steps(merged_gdf, filtered_tracts):

    # List to store results
    results = []

    # Use ProcessPoolExecutor to parallelize the processing of tracts
    with ProcessPoolExecutor(initializer=init_worker, initargs=(merged_gdf,)) as executor:
        futures = {executor.submit(process_tract, tract): tract for index, tract in filtered_tracts.iterrows()}

        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating steps to nearest nom_desig"):
            tract = futures[future]
            tract_id, steps = future.result()
            results.append((tract_id, steps))

    # Update the merged_gdf with the results
    for tract_id, steps in results:
        merged_gdf.loc[merged_gdf['GEOID'] == tract_id, 'steps_to_nom_desig'] = steps

    # Example to check the updated merged_gdf
    #print(merged_gdf[['GEOID', 'steps_to_nom_desig']].head())

    # Calculate mean, median, and standard deviation
    mean_steps = merged_gdf['steps_to_nom_desig'].replace(-999, np.nan).mean()
    median_steps = merged_gdf['steps_to_nom_desig'].replace(-999, np.nan).median()
    std_steps = merged_gdf['steps_to_nom_desig'].replace(-999, np.nan).std()

    # Print statistics
    print(f"Mean steps to nearest nom_desig: {mean_steps}")
    print(f"Median steps to nearest nom_desig: {median_steps}")
    print(f"Standard deviation of steps to nearest nom_desig: {std_steps}")

    # Count number of -999 entries
    num_neg999 = (merged_gdf['steps_to_nom_desig'] == -999).sum()
    #print(f"Number of -999 entries in steps_to_nom_desig: {num_neg999}")

    # Get the maximum value in steps_to_nom_desig column, excluding NaN and -999
    max_step = merged_gdf['steps_to_nom_desig'].loc[~merged_gdf['steps_to_nom_desig'].isin([-999, np.nan])].max()

    # Plot the histogram with bins for each integer step
    plt.figure(figsize=(10, 6))
    merged_gdf.loc[~merged_gdf['steps_to_nom_desig'].isin([-999, np.nan]), 'steps_to_nom_desig'].hist(
        bins=np.arange(1, max_step + 2) - 0.5, edgecolor='black'
    )
    plt.title('Distribution of Steps to Nearest Nom Desig')
    plt.xlabel('Steps to Nearest Nom Desig')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()



    return merged_gdf