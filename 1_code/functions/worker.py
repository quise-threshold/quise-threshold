# worker.py

import numpy as np
from libpysal.weights import Queen, shimbel
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx

# Global variable to store the GeoDataFrame
GLOBAL_MERGED_GDF = None

def init_worker(merged_gdf_data):
    """Initialize worker process with global read-only GeoDataFrame"""
    global GLOBAL_MERGED_GDF
    GLOBAL_MERGED_GDF = merged_gdf_data

def calculate_steps_to_nearest_nom_desig(tract_id, geometry, initial_buffer=1000, max_buffer=130000, buffer_increment=5000):
    
    global GLOBAL_MERGED_GDF
    
    buffer_distance = initial_buffer
    while buffer_distance <= max_buffer:
        buffered_area = geometry.buffer(buffer_distance)
        nearby_tracts = GLOBAL_MERGED_GDF[GLOBAL_MERGED_GDF.intersects(buffered_area)].reset_index(drop=True).copy()
        nom_desig_tracts = nearby_tracts[(nearby_tracts['nom_desig'] == 1)].copy()
        
        if not nom_desig_tracts.empty:
            w = Queen.from_dataframe(nearby_tracts, use_index=True)
            G = w.to_networkx()
            
            # Find source node
            tract_indices = nearby_tracts.index[nearby_tracts['GEOID'] == tract_id].tolist()
            if tract_indices:
                source_node = tract_indices[0]
                shortest_paths = nx.single_source_shortest_path_length(G, source=source_node)
                
                # Filter to target nodes
                valid_indices = nom_desig_tracts.index.tolist()
                valid_distances = [dist for i, dist in shortest_paths.items() if i in valid_indices]
                
                if valid_distances:
                    return min(valid_distances), nearby_tracts
        
        # Increase buffer distance
        buffer_distance *= 2
    
    return -999, None


def process_tract(tract):
    tract_id = tract['GEOID']
    geometry = tract.geometry
    try:
        # Calculate steps to the nearest nom_desig
        steps, _ = calculate_steps_to_nearest_nom_desig(tract_id, geometry)
        return tract_id, steps
    except Exception as e:
        print(f"Error processing tract {tract_id}: {e}")
        return tract_id, -999  # Use -999 for errors
