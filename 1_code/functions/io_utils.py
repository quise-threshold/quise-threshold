import os
import pandas as pd
import geopandas as gpd
from shapely.wkb import loads as wkb_loads
from shapely.wkt import loads as wkt_loads


def read_from_csv(name, folder, geoid_column):
    path = os.path.join(os.path.dirname(os.getcwd()), '0_data', folder)
    file_path = os.path.join(path, name)
    
    # Normalize the path to handle inconsistent slashes
    file_path = os.path.normpath(file_path)
    
    
    # Define dtype for the geoid_column if provided
    dtype = {geoid_column: str} if geoid_column else None
    
    df = pd.read_csv(file_path, dtype=dtype)
    return df


def save_to_csv(df, folder, filename):
    path = os.path.join(os.path.dirname(os.getcwd()), '0_data', folder)
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, filename)
    df.to_csv(file_path, index=False)
    print(f'File saved to {file_path}')
    return


def write_geojson(gdf, folder, filename):
    path = os.path.join(os.path.dirname(os.getcwd()), '0_data', folder)
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, filename)
    gdf.to_file(file_path, driver='GeoJSON')
    print(f'GeoJSON file saved to {file_path}')
    return


def read_geojson(folder, filename):
    path = os.path.join(os.path.dirname(os.getcwd()), '0_data', folder)
    file_path = os.path.join(path, filename)
    gdf = gpd.read_file(file_path)
    return gdf


def write_shp(gdf, folder, filename):
    path = os.path.join(os.path.dirname(os.getcwd()), '0_data', folder)
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, filename)
    gdf.to_file(file_path, driver='ESRI Shapefile')
    print(f'Shapefile saved to {file_path}')
    return


def read_shp(folder, filename):
    path = os.path.join(os.path.dirname(os.getcwd()), '0_data', folder)
    file_path = os.path.join(path, filename)
    gdf = gpd.read_file(file_path)
    return gdf


import geopandas as gpd
import os
from shapely.wkb import dumps as wkb_dumps

def write_parquet(gdf, folder, filename):
    """
    Save a GeoDataFrame to a Parquet file, converting geometry to WKB and saving CRS.
    """
    # Construct the file path
    folder_path = os.path.join(os.path.dirname(os.getcwd()), '0_data', folder)  
    file_path = os.path.join(folder_path, filename)
    file_path = os.path.normpath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Convert geometry to WKB
    gdf_copy = gdf.copy()
    
    if 'geometry' in gdf_copy.columns:
        # Save CRS as a text column
        gdf_copy['crs_text'] = gdf.crs.to_string() if gdf.crs else None

        # Save to Parquet
        gdf_copy.to_parquet(file_path, index=False, engine="pyarrow", compression="snappy")
        print(f"GeoDataFrame saved to {file_path} with CRS: {gdf.crs}")
    else:
        gdf_copy.to_parquet(file_path, index=False, engine="pyarrow", compression="snappy")
        print(f"DataFrame saved to {file_path}")




import geopandas as gpd
import pandas as pd
from shapely.wkb import loads as wkb_loads
import os

def read_parquet(folder, filename):
    """
    Reads a Parquet file and restores it as a GeoDataFrame, converting WKB back to Shapely geometry and restoring the CRS.
    """
    # Construct the file path
    folder_path = os.path.join(os.path.dirname(os.getcwd()), '0_data', folder)
    file_path = os.path.join(folder_path, filename)
    file_path = os.path.normpath(file_path)

    # Read the Parquet file
    df = pd.read_parquet(file_path)

    if 'geometry' in df.columns:
        # Convert the 'geometry' column to Shapely objects
        def convert_geometry(value):
            if isinstance(value, bytes):  # Handle WKB
                return wkb_loads(value)
            elif isinstance(value, str):  # Handle WKT
                return wkt_loads(value)
            else:
                return value  # Already a Shapely object or None
        
        df['geometry'] = df['geometry'].apply(convert_geometry)


        # Extract CRS from the 'crs_text' column
        crs_text = df['crs_text'].iloc[0] if 'crs_text' in df.columns and not df['crs_text'].isna().all() else None
        if 'crs_text' in df.columns:
            df = df.drop(columns=['crs_text'])  # Remove 'crs_text' column after extracting CRS
            print(f"CRS restored from file: {crs_text}")

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs_text)
        if not gdf.crs:
            print("Warning: CRS not found. GeoDataFrame created without CRS.")

        return gdf
    else:
        print("No geometry column found. Returning as a regular DataFrame.")
        return df




  


