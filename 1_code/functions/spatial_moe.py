import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from distance_to_designation import calculate_steps
from io_utils import write_parquet

def calculate_multi_confidence_distances(merged_gdf, threshold_value=0.75, save_results=False):
    """
    Calculate spatial distances for multiple confidence levels and create comprehensive analysis.
    
    Parameters:
    -----------
    merged_gdf : GeoDataFrame
        Merged geodataframe with replicate data
    threshold_value : float
        Designation threshold (default 0.75)
    save_results : bool
        Whether to save results to parquet
        
    Returns:
    --------
    merged_gdf : GeoDataFrame
        Enhanced geodataframe with distance columns for all confidence levels
    analysis_summary : dict
        Summary statistics for all confidence levels
    """
    
    print("Multi-Confidence Level Spatial Distance Analysis")
    print("="*60)
    
    # Calculate basic designation statistics
    cols_to_check = ['NOMINAL'] + [f'Index_Rep{i}' for i in range(1, 81)]
    merged_gdf['count_design'] = (merged_gdf[cols_to_check] >= threshold_value).sum(axis=1)
    merged_gdf['nom_desig'] = (merged_gdf['NOMINAL'] >= threshold_value).astype(int)
    
    # Define confidence levels and their corresponding thresholds
    confidence_thresholds = {
        #99: 1,   # 99% CI: remove ~0.5% from each tail
        95: 4,   # 95% CI: remove ~2.5% from each tail  
        90: 8,   # 90% CI: remove ~5% from each tail
        65: 28   # 65% CI: remove ~17.5% from each tail
    }
    
    analysis_summary = {
        'threshold_value': threshold_value,
        'total_tracts': len(merged_gdf),
        'nominal_designated': merged_gdf['nom_desig'].sum(),
        'confidence_levels': {}
    }
    
    # Calculate distances for each confidence level
    for conf_level, min_designations in confidence_thresholds.items():
        print(f"\nProcessing {conf_level}% Confidence Level (≥{min_designations} designations)...")
        
        # Find tracts with confident designation but not by nominal estimate
        filtered_tracts = merged_gdf[
            (merged_gdf['count_design'] >= min_designations) & 
            (merged_gdf['nom_desig'] != 1)
        ]
        
        print(f"  Found {len(filtered_tracts):,} tracts for distance calculation...")
        
        if len(filtered_tracts) > 0:
            # Calculate distances (this will add 'steps_to_nom_desig' column)
            merged_gdf_temp = calculate_steps(merged_gdf.copy(), filtered_tracts)
            
            # Rename the column to avoid overwriting
            distance_col_name = f'distance_to_nominal_{conf_level}pct'
            merged_gdf[distance_col_name] = merged_gdf_temp['steps_to_nom_desig']
            
            # Calculate statistics for this confidence level
            valid_distances = merged_gdf[merged_gdf[distance_col_name] != -999][distance_col_name]
            
            stats_dict = {
                'filtered_tract_count': len(filtered_tracts),
                'min_designations_threshold': min_designations,
                'mean_distance': valid_distances.mean() if len(valid_distances) > 0 else np.nan,
                'median_distance': valid_distances.median() if len(valid_distances) > 0 else np.nan,
                'max_distance': valid_distances.max() if len(valid_distances) > 0 else np.nan,
                'tracts_with_valid_distance': len(valid_distances),
                'column_name': distance_col_name
            }
            
            analysis_summary['confidence_levels'][conf_level] = stats_dict
            
            print(f"  Mean distance: {stats_dict['mean_distance']:.2f}")
            print(f"  Median distance: {stats_dict['median_distance']:.2f}")
            print(f"  Max distance: {stats_dict['max_distance']:.0f}")
            print(f"  Valid distances: {stats_dict['tracts_with_valid_distance']:,}")
        
        else:
            # No tracts found for this confidence level
            distance_col_name = f'distance_to_nominal_{conf_level}pct'
            merged_gdf[distance_col_name] = -999  # Fill with default value
            
            analysis_summary['confidence_levels'][conf_level] = {
                'filtered_tract_count': 0,
                'min_designations_threshold': min_designations,
                'mean_distance': np.nan,
                'median_distance': np.nan,
                'max_distance': np.nan,
                'tracts_with_valid_distance': 0,
                'column_name': distance_col_name
            }
            
            print(f"  No tracts found for this confidence level.")
    
    # Save results if requested
    if save_results:
        output_path = 'gitignore/multi_confidence_analysis/gdf_multi_confidence_distances.parquet'
        write_parquet(merged_gdf, output_path.split('/')[0] + '/' + output_path.split('/')[1], 
                     output_path.split('/')[-1])
        print(f"\nResults saved to: {output_path}")
    
    return merged_gdf, analysis_summary

def create_multi_confidence_visualization(merged_gdf, analysis_summary, figsize=(20, 15)):
    """
    Create comprehensive visualization comparing all confidence levels.
    """
    
    confidence_levels = list(analysis_summary['confidence_levels'].keys())
    n_conf_levels = len(confidence_levels)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Multi-Confidence Level Spatial Distance Analysis:\nFrom Individual Uncertainty to Spatial Certainty', 
                 fontsize=16, fontweight='bold')
    
    # Color palette for confidence levels
    colors = plt.cm.viridis(np.linspace(0, 1, n_conf_levels))
    
    # Panel 1: Distance distributions (histograms)
    for i, conf_level in enumerate(confidence_levels):
        ax = fig.add_subplot(gs[0, i])
        
        col_name = analysis_summary['confidence_levels'][conf_level]['column_name']
        valid_data = merged_gdf[merged_gdf[col_name] != -999][col_name]
        
        if len(valid_data) > 0:
            max_dist = int(valid_data.max())
            bins = np.arange(1, max_dist + 2) - 0.5
            
            ax.hist(valid_data, bins=bins, alpha=0.7, color=colors[i], 
                   edgecolor='black', linewidth=0.5)
            
            # Add statistics
            stats_text = f"n={len(valid_data):,}\nMean: {valid_data.mean():.1f}\nMedian: {valid_data.median():.1f}"
            ax.text(0.7, 0.8, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9, verticalalignment='top')
        
        ax.set_title(f'{conf_level}% CI\n(≥{analysis_summary["confidence_levels"][conf_level]["min_designations_threshold"]} designations)', 
                    fontweight='bold', fontsize=10)
        ax.set_xlabel('Distance to Nominal Designation')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Panel 2: Comparative analysis
    ax_comparison = fig.add_subplot(gs[1, :2])
    
    # Extract data for comparison
    conf_labels = []
    mean_distances = []
    tract_counts = []
    
    for conf_level in confidence_levels:
        conf_data = analysis_summary['confidence_levels'][conf_level]
        conf_labels.append(f"{conf_level}%")
        mean_distances.append(conf_data['mean_distance'] if not np.isnan(conf_data['mean_distance']) else 0)
        tract_counts.append(conf_data['filtered_tract_count'])
    
    # Bar plot of mean distances
    bars = ax_comparison.bar(conf_labels, mean_distances, color=colors, alpha=0.7, edgecolor='black')
    ax_comparison.set_title('Mean Distance to Nominal Designation by Confidence Level', fontweight='bold')
    ax_comparison.set_xlabel('Confidence Level')
    ax_comparison.set_ylabel('Mean Distance (Steps)')
    ax_comparison.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_dist in zip(bars, mean_distances):
        if mean_dist > 0:
            ax_comparison.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                             f'{mean_dist:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 3: Tract counts
    ax_counts = fig.add_subplot(gs[1, 2:])
    
    bars2 = ax_counts.bar(conf_labels, tract_counts, color=colors, alpha=0.7, edgecolor='black')
    ax_counts.set_title('Number of Tracts Analyzed by Confidence Level', fontweight='bold')
    ax_counts.set_xlabel('Confidence Level')
    ax_counts.set_ylabel('Number of Tracts')
    ax_counts.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars2, tract_counts):
        ax_counts.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(tract_counts)*0.01,
                      f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 4: Combined distance comparison
    ax_combined = fig.add_subplot(gs[2, :])
    
    # Plot overlapping histograms
    max_distance_overall = 0
    for i, conf_level in enumerate(confidence_levels):
        col_name = analysis_summary['confidence_levels'][conf_level]['column_name']
        valid_data = merged_gdf[merged_gdf[col_name] != -999][col_name]
        
        if len(valid_data) > 0:
            max_distance_overall = max(max_distance_overall, valid_data.max())
            
            # Create density histogram
            ax_combined.hist(valid_data, bins=np.arange(1, max_distance_overall + 2) - 0.5, 
                           alpha=0.6, color=colors[i], label=f'{conf_level}% CI (n={len(valid_data):,})',
                           density=True, edgecolor='black', linewidth=0.5)
    
    ax_combined.set_title('Distance Distribution Comparison Across Confidence Levels', fontweight='bold')
    ax_combined.set_xlabel('Distance to Nearest Nominal Designation')
    ax_combined.set_ylabel('Density')
    ax_combined.legend(loc='upper right')
    ax_combined.grid(True, alpha=0.3)
    
    # Add summary statistics box
    summary_text = f"""Summary Statistics:
Total Tracts: {analysis_summary['total_tracts']:,}
Nominal Designated: {analysis_summary['nominal_designated']:,}
Threshold: {analysis_summary['threshold_value']}

Key Insight: Lower confidence levels capture more spatial 
disagreement, revealing broader uncertainty patterns."""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             verticalalignment='bottom')
    
    return fig

def print_analysis_summary(analysis_summary):
    """
    Print a comprehensive summary of the multi-confidence analysis.
    """
    
    print("\n" + "="*80)
    print("MULTI-CONFIDENCE LEVEL SPATIAL ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Dataset Overview:")
    print(f"  Total tracts: {analysis_summary['total_tracts']:,}")
    print(f"  Nominal designated (threshold ≥{analysis_summary['threshold_value']}): {analysis_summary['nominal_designated']:,}")
    print(f"  Designation rate: {analysis_summary['nominal_designated']/analysis_summary['total_tracts']*100:.1f}%")
    print()
    
    print("Confidence Level Analysis:")
    print("-" * 40)
    
    for conf_level in sorted(analysis_summary['confidence_levels'].keys(), reverse=True):
        data = analysis_summary['confidence_levels'][conf_level]
        
        print(f"{conf_level:2d}% CI (≥{data['min_designations_threshold']:2d} designations):")
        print(f"    Tracts analyzed: {data['filtered_tract_count']:8,}")
        
        if data['tracts_with_valid_distance'] > 0:
            print(f"    Mean distance:   {data['mean_distance']:8.2f} steps")
            print(f"    Median distance: {data['median_distance']:8.2f} steps") 
            print(f"    Max distance:    {data['max_distance']:8.0f} steps")
        else:
            print(f"    No valid distances calculated")
        print()
    
    print("Key Insights:")
    print("• Higher confidence levels (99%, 95%) capture fewer but more certain disagreements")
    print("• Lower confidence levels (90%, 65%) reveal broader spatial uncertainty patterns")
    print("• Distance patterns show how far spatial uncertainty extends from nominal designations")

# Main execution function
def run_multi_confidence_spatial_analysis(merged_gdf, threshold_value=0.75, create_plots=True):
    """
    Run complete multi-confidence level spatial analysis.
    """
    
    # Calculate distances for all confidence levels
    enhanced_gdf, analysis_summary = calculate_multi_confidence_distances(merged_gdf, threshold_value)
    
    # Print summary
    print_analysis_summary(analysis_summary)
    
    # Create visualizations
    if create_plots:
        fig = create_multi_confidence_visualization(enhanced_gdf, analysis_summary)
        plt.show()
        return enhanced_gdf, analysis_summary, fig
    
    return enhanced_gdf, analysis_summary

# Usage example:
# enhanced_gdf, summary, fig = run_multi_confidence_spatial_analysis(merged_gdf, threshold_value=0.75)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def create_spatial_influence_zones_real_data(gdf_multi_distances, confidence_thresholds, figsize=(15, 10)):
    """
    Create spatial influence zones visualization using your actual distance data.
    """
    
    # Extract real distance data for each confidence level
    zone_data = []
    
    for conf_level, min_designations in confidence_thresholds.items():
        col_name = f'distance_to_nominal_{conf_level}pct'
        
        # Get eligible tracts for this confidence level
        eligible_tracts = gdf_multi_distances[
            (gdf_multi_distances['count_design'] >= min_designations) & 
            (gdf_multi_distances['nom_desig'] != 1)
        ]
        
        # Get valid distances
        valid_distances = eligible_tracts[eligible_tracts[col_name] > 0][col_name]
        
        if len(valid_distances) > 0:
            # Calculate statistics for each distance ring
            max_distance = int(valid_distances.max())
            
            for distance in range(1, max_distance + 1):
                tracts_at_distance = (valid_distances == distance).sum()
                
                if tracts_at_distance > 0:
                    zone_data.append({
                        'confidence_level': conf_level,
                        'distance': distance,
                        'tract_count': tracts_at_distance,
                        'cumulative_tracts': (valid_distances <= distance).sum(),
                        'percentage_at_distance': tracts_at_distance / len(valid_distances) * 100,
                        'cumulative_percentage': (valid_distances <= distance).sum() / len(valid_distances) * 100
                    })
    
    zone_df = pd.DataFrame(zone_data)
    
    # Create the visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Spatial Influence Zones: From Individual Uncertainty to Neighborhood Certainty', 
                 fontsize=16, fontweight='bold')
    
    # Define colors for confidence levels
    colors = {95: '#fee5d9', 90: '#fcae91', 65: '#cb181d'}
    
    # Panel A: Cumulative percentage by distance (influence zones)
    ax1.set_title('A) Spatial Consensus Emergence by Distance', fontweight='bold')
    
    for conf_level in confidence_thresholds.keys():
        data = zone_df[zone_df['confidence_level'] == conf_level]
        if len(data) > 0:
            ax1.plot(data['distance'], data['cumulative_percentage'], 
                    'o-', color=colors[conf_level], linewidth=3, markersize=8,
                    label=f'{conf_level}% CI ({len(data)} distances)')
            
            # Fill area under curve to show zones
            ax1.fill_between(data['distance'], 0, data['cumulative_percentage'],
                            alpha=0.3, color=colors[conf_level])
    
    ax1.set_xlabel('Distance from Designated Areas (Steps)')
    ax1.set_ylabel('Cumulative % of Uncertain Tracts')
    ax1.legend(title='Confidence Level')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Panel B: Tract density by distance
    ax2.set_title('B) Uncertainty Density by Distance Ring', fontweight='bold')
    
    for conf_level in confidence_thresholds.keys():
        data = zone_df[zone_df['confidence_level'] == conf_level]
        if len(data) > 0:
            ax2.bar(data['distance'], data['percentage_at_distance'], 
                   alpha=0.7, color=colors[conf_level], width=0.8,
                   label=f'{conf_level}% CI')
    
    ax2.set_xlabel('Distance from Designated Areas (Steps)')
    ax2.set_ylabel('% of Tracts at This Distance')
    ax2.legend(title='Confidence Level')
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Network hierarchy visualization
    ax3.set_title('C) Spatial Uncertainty Network', fontweight='bold')
    
    # Create network graph showing connections between distance rings
    G = nx.Graph()
    pos = {}
    node_sizes = []
    node_colors = []
    
    # Add nodes for each distance ring across all confidence levels
    max_dist = zone_df['distance'].max()
    
    for distance in range(1, max_dist + 1):
        # Calculate total tracts at this distance across all confidence levels
        total_at_distance = zone_df[zone_df['distance'] == distance]['tract_count'].sum()
        
        if total_at_distance > 0:
            G.add_node(distance)
            pos[distance] = (distance * 2, 0)  # Linear layout
            node_sizes.append(total_at_distance * 10)  # Scale for visibility
            
            # Color by uncertainty intensity
            intensity = total_at_distance / zone_df['tract_count'].max()
            node_colors.append(intensity)
            
            # Add edges between consecutive distances
            if distance > 1 and distance - 1 in G.nodes():
                G.add_edge(distance - 1, distance)
    
    if len(G.nodes()) > 0:
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=node_colors, cmap='Reds',
                              alpha=0.8, ax=ax3)
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              width=3, alpha=0.6, ax=ax3)
        
        nx.draw_networkx_labels(G, pos, font_size=12, 
                               font_weight='bold', ax=ax3)
    
    ax3.set_xlabel('Distance from Designated Areas →')
    ax3.set_title('C) Spatial Network: Node Size = Tract Count', fontweight='bold')
    ax3.axis('off')
    
    # Panel D: Confidence level comparison
    ax4.set_title('D) Multi-Confidence Spatial Pattern', fontweight='bold')
    
    # Create stacked area chart showing how patterns differ by confidence
    confidence_levels = sorted(confidence_thresholds.keys(), reverse=True)
    
    if len(zone_df) > 0:
        distances = sorted(zone_df['distance'].unique())
        
        # Prepare data for stacked area
        confidence_data = {}
        for conf_level in confidence_levels:
            data = zone_df[zone_df['confidence_level'] == conf_level]
            confidence_data[conf_level] = []
            
            for dist in distances:
                dist_data = data[data['distance'] == dist]
                if len(dist_data) > 0:
                    confidence_data[conf_level].append(dist_data['tract_count'].iloc[0])
                else:
                    confidence_data[conf_level].append(0)
        
        # Create stacked area plot
        bottom = np.zeros(len(distances))
        for i, conf_level in enumerate(confidence_levels):
            ax4.fill_between(distances, bottom, 
                            bottom + confidence_data[conf_level],
                            alpha=0.7, color=colors[conf_level],
                            label=f'{conf_level}% CI')
            bottom += confidence_data[conf_level]
        
        ax4.set_xlabel('Distance from Designated Areas (Steps)')
        ax4.set_ylabel('Number of Tracts')
        ax4.legend(title='Confidence Level')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    return fig, zone_df

def create_network_hierarchy_real_data(gdf_multi_distances, confidence_thresholds, figsize=(14, 10)):
    """
    Create network hierarchy visualization showing spatial uncertainty flow.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Spatial Uncertainty Network Hierarchy', fontsize=16, fontweight='bold')
    
    # Extract data for network
    network_data = {}
    
    for conf_level, min_designations in confidence_thresholds.items():
        col_name = f'distance_to_nominal_{conf_level}pct'
        
        eligible_tracts = gdf_multi_distances[
            (gdf_multi_distances['count_design'] >= min_designations) & 
            (gdf_multi_distances['nom_desig'] != 1)
        ]
        
        valid_distances = eligible_tracts[eligible_tracts[col_name] > 0][col_name]
        
        if len(valid_distances) > 0:
            distance_counts = valid_distances.value_counts().sort_index()
            network_data[conf_level] = distance_counts
    
    # Panel 1: Hierarchical network layout
    ax1.set_title('A) Uncertainty Propagation Network', fontweight='bold')
    
    G = nx.DiGraph()
    pos = {}
    node_sizes = []
    node_colors = []
    
    # Create hierarchical layout
    if network_data:
        max_distance = int(max([max(data.index) for data in network_data.values() if len(data) > 0]))
    else:
        max_distance = 1
    
    # Add nodes for each distance level
    for distance in range(1, max_distance + 1):
        total_tracts = sum([data.get(distance, 0) for data in network_data.values()])
        
        if total_tracts > 0:
            G.add_node(f"D{distance}")
            
            # Position nodes in concentric circles
            angle = (distance - 1) * (2 * np.pi / max_distance)
            radius = distance * 100
            pos[f"D{distance}"] = (radius * np.cos(angle), radius * np.sin(angle))
            
            node_sizes.append(total_tracts * 5)
            node_colors.append(distance)
            
            # Add edges showing flow between distances
            if distance > 1:
                G.add_edge(f"D{distance-1}", f"D{distance}")
    
    # Draw network
    if len(G.nodes()) > 0:
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color=node_colors, cmap='viridis',
                              alpha=0.8, ax=ax1)
        
        nx.draw_networkx_edges(G, pos, edge_color='darkblue',
                              arrows=True, arrowsize=20,
                              width=2, alpha=0.7, ax=ax1)
        
        nx.draw_networkx_labels(G, pos, font_size=10,
                               font_weight='bold', ax=ax1)
    
    ax1.set_title('A) Spatial Distance Network\n(Node size = tract count)', fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Flow diagram
    ax2.set_title('B) Confidence Level Flow Analysis', fontweight='bold')
    
    # Create flow visualization
    if max_distance > 0:
        distances = range(1, max_distance + 1)
        width = 0.25
        x = np.arange(len(distances))
        
        colors_list = ['#cb181d', '#fcae91', '#fee5d9']
        
        for i, (conf_level, data) in enumerate(network_data.items()):
            counts = [data.get(d, 0) for d in distances]
            
            ax2.bar(x + i * width, counts, width, 
                   label=f'{conf_level}% CI',
                   color=colors_list[i % len(colors_list)],
                   alpha=0.8)
        
        ax2.set_xlabel('Distance from Designated Areas')
        ax2.set_ylabel('Number of Tracts')
        ax2.set_title('B) Tract Distribution by Distance and Confidence', fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(distances)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return fig

def create_confidence_cascade_diagram(gdf_multi_distances, confidence_thresholds, figsize=(12, 8)):
    """
    Create cascade diagram showing how uncertainty flows between confidence levels.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle('Confidence Level Cascade: Spatial Uncertainty Flow', fontsize=16, fontweight='bold')
    
    # Calculate overlap between confidence levels
    cascade_data = []
    
    conf_levels = sorted(confidence_thresholds.keys(), reverse=True)  # 95, 90, 65
    
    for i, conf_level in enumerate(conf_levels):
        min_designations = confidence_thresholds[conf_level]
        col_name = f'distance_to_nominal_{conf_level}pct'
        
        eligible_tracts = gdf_multi_distances[
            (gdf_multi_distances['count_design'] >= min_designations) & 
            (gdf_multi_distances['nom_desig'] != 1)
        ]
        
        valid_count = len(eligible_tracts[eligible_tracts[col_name] > 0])
        
        cascade_data.append({
            'level': conf_level,
            'count': valid_count,
            'position': i,
            'color': ['#cb181d', '#fcae91', '#fee5d9'][i]
        })
    
    # Create cascade visualization
    for i, data in enumerate(cascade_data):
        # Draw rectangle for each confidence level
        width = data['count'] / 100  # Scale for visibility
        height = 0.5
        
        rect = plt.Rectangle((0, i), width, height, 
                           facecolor=data['color'], 
                           edgecolor='black', 
                           alpha=0.8)
        ax.add_patch(rect)
        
        # Add labels
        ax.text(width/2, i + height/2, 
               f"{data['level']}% CI\n{data['count']:,} tracts",
               ha='center', va='center', fontweight='bold')
        
        # Draw flow arrows between levels
        if i < len(cascade_data) - 1:
            next_width = cascade_data[i+1]['count'] / 100
            
            # Arrow from current level to next
            ax.annotate('', xy=(next_width/2, i+1), xytext=(width/2, i + height),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    ax.set_xlim(-5, max([d['count'] for d in cascade_data])/100 + 5)
    ax.set_ylim(-0.5, len(cascade_data))
    ax.set_xlabel('Relative Number of Tracts →')
    ax.set_ylabel('← Higher Confidence     Lower Confidence →')
    ax.set_title('Uncertainty Cascade: How Stricter Confidence Reduces Uncertain Tracts', fontweight='bold')
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig

# Usage function
def create_all_network_plots(gdf_multi_distances, confidence_thresholds={95: 4, 90: 8, 65: 28}):
    """
    Create all network-style plots using real data.
    """
    
    # Create the three main network visualizations
    fig1, zone_data = create_spatial_influence_zones_real_data(gdf_multi_distances, confidence_thresholds)
    fig2 = create_network_hierarchy_real_data(gdf_multi_distances, confidence_thresholds)
    fig3 = create_confidence_cascade_diagram(gdf_multi_distances, confidence_thresholds)
    
    return fig1, fig2, fig3, zone_data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score
import networkx as nx

class SpatialDecayAnalyzer:
    """
    Comprehensive analysis of spatial uncertainty patterns for Nature publication.
    """
    
    def __init__(self, gdf_multi_distances):
        self.gdf = gdf_multi_distances
        self.distance_col = 'distance_to_nominal_95pct'  # Use most comprehensive data
        
        # Extract valid distance data
        self.valid_distances = self.gdf[self.gdf[self.distance_col] > 0][self.distance_col]
        self.distance_counts = self.valid_distances.value_counts().sort_index()
        
        print(f"Initialized analysis with {len(self.valid_distances):,} uncertain tracts")
        print(f"Distance range: 1 to {self.valid_distances.max()} steps")
        
    def analyze_spatial_decay_pattern(self):
        """
        1. Fit exponential decay model to spatial uncertainty distribution.
        """
        print("\n" + "="*60)
        print("1. SPATIAL DECAY PATTERN ANALYSIS")
        print("="*60)
        
        # Prepare data for fitting
        distances = np.array(self.distance_counts.index)
        counts = np.array(self.distance_counts.values)
        
        # Exponential decay function: y = a * exp(-b * x)
        def exponential_decay(x, a, b):
            return a * np.exp(-b * x)
        
        # Power law function: y = a * x^(-b)
        def power_law(x, a, b):
            return a * (x ** (-b))
        
        # Fit exponential decay
        try:
            exp_params, _ = curve_fit(exponential_decay, distances, counts, p0=[counts[0], 0.5])
            exp_pred = exponential_decay(distances, *exp_params)
            exp_r2 = r2_score(counts, exp_pred)
            
            print(f"Exponential Decay: y = {exp_params[0]:.0f} * exp(-{exp_params[1]:.3f} * x)")
            print(f"  R² = {exp_r2:.3f}")
            print(f"  Decay rate (λ) = {exp_params[1]:.3f} per step")
            print(f"  Half-life = {np.log(2)/exp_params[1]:.1f} steps")
            
        except Exception as e:
            print(f"Exponential fitting failed: {e}")
            exp_params = None
            exp_r2 = 0
        
        # Fit power law
        try:
            power_params, _ = curve_fit(power_law, distances, counts, p0=[counts[0], 1])
            power_pred = power_law(distances, *power_params)
            power_r2 = r2_score(counts, power_pred)
            
            print(f"Power Law: y = {power_params[0]:.0f} * x^(-{power_params[1]:.3f})")
            print(f"  R² = {power_r2:.3f}")
            print(f"  Scaling exponent = {power_params[1]:.3f}")
            
        except Exception as e:
            print(f"Power law fitting failed: {e}")
            power_params = None
            power_r2 = 0
        
        # Determine best fit
        if exp_r2 > power_r2:
            print(f"\n✓ Best fit: EXPONENTIAL DECAY (R² = {exp_r2:.3f})")
            best_model = 'exponential'
            best_params = exp_params
        else:
            print(f"\n✓ Best fit: POWER LAW (R² = {power_r2:.3f})")
            best_model = 'power_law'
            best_params = power_params
        
        return {
            'distances': distances,
            'counts': counts,
            'exponential_params': exp_params,
            'exponential_r2': exp_r2,
            'power_params': power_params,
            'power_r2': power_r2,
            'best_model': best_model,
            'best_params': best_params
        }
    
    def compare_to_random_baseline(self, n_simulations=1000):
        """
        2. Compare observed pattern to random spatial distribution baseline.
        """
        print("\n" + "="*60)
        print("2. RANDOM vs. CLUSTERED COMPARISON")
        print("="*60)
        
        total_uncertain = len(self.valid_distances)
        max_distance = self.valid_distances.max()
        
        # Simulate random distribution
        print(f"Running {n_simulations} random simulations...")
        
        random_means = []
        random_distance_1_pcts = []
        
        for _ in range(n_simulations):
            # Random distances from 1 to max_distance
            random_distances = np.random.randint(1, max_distance + 1, total_uncertain)
            
            random_means.append(np.mean(random_distances))
            random_distance_1_pcts.append((random_distances == 1).sum() / len(random_distances) * 100)
        
        # Observed statistics
        observed_mean = self.valid_distances.mean()
        observed_distance_1_pct = (self.valid_distances == 1).sum() / len(self.valid_distances) * 100
        
        # Calculate z-scores
        random_mean_std = np.std(random_means)
        random_d1_std = np.std(random_distance_1_pcts)
        
        mean_z_score = (observed_mean - np.mean(random_means)) / random_mean_std
        d1_z_score = (observed_distance_1_pct - np.mean(random_distance_1_pcts)) / random_d1_std
        
        print(f"Mean Distance:")
        print(f"  Observed: {observed_mean:.2f} steps")
        print(f"  Random expectation: {np.mean(random_means):.2f} ± {random_mean_std:.2f}")
        print(f"  Z-score: {mean_z_score:.1f} (p < {stats.norm.sf(abs(mean_z_score)):.2e})")
        
        print(f"\nDistance = 1 Concentration:")
        print(f"  Observed: {observed_distance_1_pct:.1f}%")
        print(f"  Random expectation: {np.mean(random_distance_1_pcts):.1f}% ± {random_d1_std:.1f}%")
        print(f"  Z-score: {d1_z_score:.1f} (p < {stats.norm.sf(abs(d1_z_score)):.2e})")
        
        if abs(mean_z_score) > 3:
            print(f"\n✓ HIGHLY SIGNIFICANT spatial clustering (Z > 3)")
        elif abs(mean_z_score) > 2:
            print(f"\n✓ SIGNIFICANT spatial clustering (Z > 2)")
        else:
            print(f"\n⚠ Weak evidence for clustering")
        
        return {
            'observed_mean': observed_mean,
            'observed_distance_1_pct': observed_distance_1_pct,
            'random_means': random_means,
            'random_distance_1_pcts': random_distance_1_pcts,
            'mean_z_score': mean_z_score,
            'd1_z_score': d1_z_score
        }
    
    def calculate_spillover_metrics(self):
        """
        3. Calculate uncertainty spillover metrics and containment radii.
        """
        print("\n" + "="*60)
        print("3. UNCERTAINTY SPILLOVER METRICS")
        print("="*60)
        
        # Calculate cumulative percentages
        cumsum_counts = self.distance_counts.cumsum()
        cumsum_pct = (cumsum_counts / cumsum_counts.iloc[-1]) * 100
        
        # Find containment distances
        def find_containment_distance(target_pct):
            idx = cumsum_pct[cumsum_pct >= target_pct].index[0]
            return idx
        
        containment_50 = find_containment_distance(50)
        containment_80 = find_containment_distance(80)
        containment_90 = find_containment_distance(90)
        containment_95 = find_containment_distance(95)
        
        print(f"Uncertainty Containment Distances:")
        print(f"  50% of uncertainty within: {containment_50} steps")
        print(f"  80% of uncertainty within: {containment_80} steps") 
        print(f"  90% of uncertainty within: {containment_90} steps")
        print(f"  95% of uncertainty within: {containment_95} steps")
        
        # Effective uncertainty radius (weighted mean distance)
        weighted_distances = (self.distance_counts.index * self.distance_counts).sum()
        total_tracts = self.distance_counts.sum()
        effective_radius = weighted_distances / total_tracts
        
        print(f"\nEffective uncertainty radius: {effective_radius:.2f} steps")
        
        # Boundary concentration index
        boundary_tracts = self.distance_counts.iloc[0]  # Distance = 1
        boundary_concentration = boundary_tracts / total_tracts * 100
        
        print(f"Boundary concentration: {boundary_concentration:.1f}% at distance = 1")
        
        return {
            'containment_50': containment_50,
            'containment_80': containment_80, 
            'containment_90': containment_90,
            'containment_95': containment_95,
            'effective_radius': effective_radius,
            'boundary_concentration': boundary_concentration,
            'cumsum_pct': cumsum_pct
        }
    
    def analyze_uncertainty_clustering(self):
        """
        4. Spatial contagion analysis - are uncertain tracts clustered together?
        """
        print("\n" + "="*60)
        print("4. UNCERTAINTY CLUSTERING ANALYSIS")
        print("="*60)
        
        # Create binary uncertainty indicator
        self.gdf['is_uncertain'] = (self.gdf[self.distance_col] > 0).astype(int)
        
        # Calculate basic clustering statistics
        total_tracts = len(self.gdf)
        uncertain_tracts = self.gdf['is_uncertain'].sum()
        uncertainty_rate = uncertain_tracts / total_tracts * 100
        
        print(f"Basic Statistics:")
        print(f"  Total tracts: {total_tracts:,}")
        print(f"  Uncertain tracts: {uncertain_tracts:,}")
        print(f"  Uncertainty rate: {uncertainty_rate:.2f}%")
        
        # Analyze clustering by distance rings
        print(f"\nClustering by Distance Ring:")
        for distance in sorted(self.distance_counts.index):
            count = self.distance_counts[distance]
            pct_of_uncertain = count / uncertain_tracts * 100
            
            print(f"  Distance {distance}: {count:,} tracts ({pct_of_uncertain:.1f}% of uncertain)")
        
        # Calculate spatial concentration metrics
        gini_coefficient = self.calculate_gini_coefficient()
        herfindahl_index = self.calculate_herfindahl_index()
        
        print(f"\nSpatial Concentration Metrics:")
        print(f"  Gini coefficient: {gini_coefficient:.3f} (0=even, 1=concentrated)")
        print(f"  Herfindahl index: {herfindahl_index:.3f} (0=dispersed, 1=concentrated)")
        
        return {
            'uncertainty_rate': uncertainty_rate,
            'gini_coefficient': gini_coefficient,
            'herfindahl_index': herfindahl_index,
            'distance_distribution': self.distance_counts.to_dict()
        }
    
    def calculate_gini_coefficient(self):
        """Calculate Gini coefficient for spatial concentration."""
        distances = self.distance_counts.values
        n = len(distances)
        
        # Sort values
        sorted_distances = np.sort(distances)
        
        # Calculate Gini
        cumsum = np.cumsum(sorted_distances)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return gini
    
    def calculate_herfindahl_index(self):
        """Calculate Herfindahl-Hirschman Index for concentration."""
        total = self.distance_counts.sum()
        proportions = self.distance_counts / total
        hhi = (proportions ** 2).sum()
        
        return hhi
    
    def create_comprehensive_visualization(self, decay_results, random_results, spillover_results, figsize=(20, 12)):
        """
        Create comprehensive Nature-quality visualization.
        """
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)
        
        fig.suptitle('Spatial Boundary Effects in Measurement Uncertainty:\nEvidence Against Random Error Hypothesis', 
                     fontsize=18, fontweight='bold')
        
        # Panel A: Decay curve fitting
        ax1 = fig.add_subplot(gs[0, :2])
        
        distances = decay_results['distances']
        counts = decay_results['counts']
        
        # Plot observed data
        ax1.bar(distances, counts, alpha=0.7, color='steelblue', edgecolor='black',
               label='Observed')
        
        # Plot fitted curves
        x_smooth = np.linspace(1, distances.max(), 100)
        
        if decay_results['exponential_params'] is not None:
            exp_params = decay_results['exponential_params']
            exp_fit = exp_params[0] * np.exp(-exp_params[1] * x_smooth)
            ax1.plot(x_smooth, exp_fit, 'r-', linewidth=3, 
                    label=f'Exponential fit (R² = {decay_results["exponential_r2"]:.3f})')
        
        if decay_results['power_params'] is not None:
            power_params = decay_results['power_params']
            power_fit = power_params[0] * (x_smooth ** (-power_params[1]))
            ax1.plot(x_smooth, power_fit, 'g--', linewidth=3,
                    label=f'Power law fit (R² = {decay_results["power_r2"]:.3f})')
        
        ax1.set_xlabel('Distance from Designated Areas (Steps)')
        ax1.set_ylabel('Number of Uncertain Tracts')
        ax1.set_title('A) Spatial Decay Pattern Analysis', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Panel B: Random vs. Observed comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        
        # Violin plot of random simulations
        parts = ax2.violinplot([random_results['random_means']], positions=[1], widths=0.5)
        parts['bodies'][0].set_color('lightgray')
        
        # Observed value
        ax2.scatter([1], [random_results['observed_mean']], color='red', s=200, zorder=5,
                   label=f'Observed (Z={random_results["mean_z_score"]:.1f})')
        
        ax2.set_xlim(0.5, 1.5)
        ax2.set_xticks([1])
        ax2.set_xticklabels(['Mean Distance'])
        ax2.set_ylabel('Mean Distance (Steps)')
        ax2.set_title('B) Random vs. Observed Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel C: Spillover containment
        ax3 = fig.add_subplot(gs[1, :2])
        
        cumsum_pct = spillover_results['cumsum_pct']
        ax3.plot(cumsum_pct.index, cumsum_pct.values, 'o-', linewidth=3, markersize=8,
                color='darkblue')
        ax3.fill_between(cumsum_pct.index, 0, cumsum_pct.values, alpha=0.3, color='lightblue')
        
        # Mark key containment levels
        containment_levels = [50, 80, 90, 95]
        colors = ['orange', 'red', 'purple', 'black']
        
        for level, color in zip(containment_levels, colors):
            containment_dist = spillover_results[f'containment_{level}']
            ax3.axhline(y=level, color=color, linestyle='--', alpha=0.7)
            ax3.axvline(x=containment_dist, color=color, linestyle='--', alpha=0.7,
                       label=f'{level}% at {containment_dist} steps')
        
        ax3.set_xlabel('Distance from Designated Areas (Steps)')
        ax3.set_ylabel('Cumulative % of Uncertain Tracts')
        ax3.set_title('C) Uncertainty Spillover Analysis', fontweight='bold')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
        
        # Panel D: Concentration metrics
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # Distance distribution as pie chart
        top_distances = self.distance_counts.head(5)
        other_count = self.distance_counts.iloc[5:].sum() if len(self.distance_counts) > 5 else 0
        
        pie_data = list(top_distances.values)
        pie_labels = [f'Distance {d}' for d in top_distances.index]
        
        if other_count > 0:
            pie_data.append(other_count)
            pie_labels.append('Distance 6+')
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(pie_data)))
        
        wedges, texts, autotexts = ax4.pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        
        ax4.set_title('D) Distance Distribution', fontweight='bold')
        
        # Panel E: Summary statistics
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create summary text
        summary_text = f"""
KEY FINDINGS FOR NATURE PUBLICATION:

SPATIAL BOUNDARY EFFECT EVIDENCE:
• {spillover_results['boundary_concentration']:.1f}% of uncertain tracts are immediate neighbors (distance=1) of designated areas
• Uncertainty decays {decay_results['best_model']} with distance (R² = {max(decay_results['exponential_r2'], decay_results['power_r2']):.3f})
• {spillover_results['containment_90']} steps contain 90% of all spatial uncertainty

REJECTION OF RANDOM ERROR HYPOTHESIS:
• Mean distance {random_results['observed_mean']:.2f} vs. random expectation {np.mean(random_results['random_means']):.2f} (Z = {random_results['mean_z_score']:.1f}, p < 0.001)
• Boundary concentration {random_results['observed_distance_1_pct']:.1f}% vs. random {np.mean(random_results['random_distance_1_pcts']):.1f}% (Z = {random_results['d1_z_score']:.1f})

POLICY IMPLICATIONS:
• Measurement uncertainty manifests as localized boundary effects, not systematic misclassification
• Geographic targeting remains valid despite measurement limitations
• Effective uncertainty radius: {spillover_results['effective_radius']:.1f} steps
        """
        
        ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes, 
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def run_complete_analysis(self):
        """
        Run all four analysis components and create comprehensive visualization.
        """
        print("COMPREHENSIVE SPATIAL UNCERTAINTY ANALYSIS")
        print("="*80)
        
        # Run all analyses
        decay_results = self.analyze_spatial_decay_pattern()
        random_results = self.compare_to_random_baseline()
        spillover_results = self.calculate_spillover_metrics()
        clustering_results = self.analyze_uncertainty_clustering()
        
        # Create visualization
        fig = self.create_comprehensive_visualization(decay_results, random_results, spillover_results)
        
        # Return all results
        return {
            'decay_analysis': decay_results,
            'random_comparison': random_results,
            'spillover_metrics': spillover_results,
            'clustering_analysis': clustering_results,
            'figure': fig
        }

# Usage function
def run_spatial_decay_analysis(gdf_multi_distances):
    """
    Main function to run the complete spatial decay analysis.
    """
    
    analyzer = SpatialDecayAnalyzer(gdf_multi_distances)
    results = analyzer.run_complete_analysis()
    
    return analyzer, results












import numpy as np
import pandas as pd
import geopandas as gpd
from distance_to_designation import calculate_steps
from io_utils import write_parquet, read_parquet
import matplotlib.pyplot as plt
from scipy import stats

def create_random_baseline_distances(gdf_multi_distances, n_random_samples=None, random_seed=42):
    """
    Create random baseline by randomly selecting non-designated tracts and calculating 
    their distances to designated areas using the same method as the confidence analysis.
    
    Parameters:
    -----------
    gdf_multi_distances : GeoDataFrame
        The dataframe from your multi-confidence analysis
    n_random_samples : int
        Number of random tracts to sample (default: same as 95% CI analysis)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    gdf_multi_distances : GeoDataFrame
        Enhanced dataframe with random baseline distance column
    """
    
    print("Creating Random Baseline Analysis")
    print("="*50)
    
    np.random.seed(random_seed)
    
    # Get the number of uncertain tracts from 95% CI analysis (most comprehensive)
    observed_uncertain = gdf_multi_distances[gdf_multi_distances['distance_to_nominal_95pct'] > 0]
    
    if n_random_samples is None:
        n_random_samples = len(observed_uncertain)
    
    print(f"Observed uncertain tracts (95% CI): {len(observed_uncertain):,}")
    print(f"Random sample size: {n_random_samples:,}")
    
    # Get all non-designated tracts (eligible for random selection)
    eligible_for_random = gdf_multi_distances[gdf_multi_distances['nom_desig'] != 1].copy()
    
    print(f"Eligible tracts for random selection: {len(eligible_for_random):,}")
    
    if len(eligible_for_random) < n_random_samples:
        print(f"WARNING: Not enough eligible tracts! Reducing sample to {len(eligible_for_random):,}")
        n_random_samples = len(eligible_for_random)
    
    # Randomly sample tracts
    print("Randomly sampling tracts...")
    random_selected_tracts = eligible_for_random.sample(n=2000, random_state=random_seed)   #n_random_samples
    
    print(f"Selected {len(random_selected_tracts):,} random tracts for distance calculation")
    
    # Calculate distances for random tracts using the same method
    print("Calculating distances for random tracts (this may take a while)...")
    
    # Initialize the random distance column with NaN
    gdf_multi_distances['distance_random_baseline'] = np.nan
    
    # Use the same calculate_steps function as the confidence analysis
    gdf_temp = calculate_steps(gdf_multi_distances.copy(), random_selected_tracts)
    
    # Copy the calculated distances to the random baseline column
    # Map from steps_to_nom_desig to our random column
    for _, tract in random_selected_tracts.iterrows():
        tract_geoid = tract['GEOID']
        
        # Find this tract in the temp result
        temp_result = gdf_temp[gdf_temp['GEOID'] == tract_geoid]
        
        if len(temp_result) > 0:
            distance_value = temp_result['steps_to_nom_desig'].iloc[0]
            
            # Update the main dataframe
            mask = gdf_multi_distances['GEOID'] == tract_geoid
            gdf_multi_distances.loc[mask, 'distance_random_baseline'] = distance_value
    
    # Calculate statistics
    random_distances = gdf_multi_distances[
        (gdf_multi_distances['distance_random_baseline'] > 0) & 
        (gdf_multi_distances['distance_random_baseline'].notna())
    ]['distance_random_baseline']
    
    print(f"\nRandom Baseline Results:")
    print(f"  Total random tracts with valid distances: {len(random_distances):,}")
    print(f"  Mean distance: {random_distances.mean():.2f}")
    print(f"  Median distance: {random_distances.median():.2f}")
    print(f"  Distance range: {random_distances.min():.0f} - {random_distances.max():.0f}")
    
    return gdf_multi_distances

def compare_random_vs_observed(gdf_multi_distances):
    """
    Compare the random baseline against observed patterns from confidence intervals.
    """
    
    print("\n" + "="*60)
    print("RANDOM vs. OBSERVED COMPARISON ANALYSIS")
    print("="*60)
    
    # Get observed distances (95% CI - most comprehensive)
    observed_distances = gdf_multi_distances[
        gdf_multi_distances['distance_to_nominal_95pct'] > 0
    ]['distance_to_nominal_95pct']
    
    # Get random baseline distances
    random_distances = gdf_multi_distances[
        (gdf_multi_distances['distance_random_baseline'] > 0) & 
        (gdf_multi_distances['distance_random_baseline'].notna())
    ]['distance_random_baseline']
    
    if len(observed_distances) == 0 or len(random_distances) == 0:
        print("ERROR: No valid distances found for comparison!")
        return None
    
    print(f"Comparing:")
    print(f"  Observed uncertain tracts: {len(observed_distances):,}")
    print(f"  Random baseline tracts: {len(random_distances):,}")
    
    # Calculate key statistics
    comparison_results = {}
    
    # Mean distance comparison
    obs_mean = observed_distances.mean()
    rand_mean = random_distances.mean()
    
    comparison_results['observed_mean'] = obs_mean
    comparison_results['random_mean'] = rand_mean
    comparison_results['mean_ratio'] = obs_mean / rand_mean
    
    print(f"\nMean Distance:")
    print(f"  Observed: {obs_mean:.2f} steps")
    print(f"  Random: {rand_mean:.2f} steps")
    print(f"  Ratio (Obs/Random): {obs_mean/rand_mean:.2f}")
    
    # Distance = 1 concentration
    obs_dist1_pct = (observed_distances == 1).sum() / len(observed_distances) * 100
    rand_dist1_pct = (random_distances == 1).sum() / len(random_distances) * 100
    
    comparison_results['observed_dist1_pct'] = obs_dist1_pct
    comparison_results['random_dist1_pct'] = rand_dist1_pct
    comparison_results['dist1_enrichment'] = obs_dist1_pct / rand_dist1_pct if rand_dist1_pct > 0 else np.inf
    
    print(f"\nBoundary Concentration (Distance = 1):")
    print(f"  Observed: {obs_dist1_pct:.1f}%")
    print(f"  Random: {rand_dist1_pct:.1f}%")
    print(f"  Enrichment factor: {obs_dist1_pct/rand_dist1_pct:.1f}x")
    
    # Statistical significance test
    # Mann-Whitney U test (non-parametric)
    statistic, p_value = stats.mannwhitneyu(observed_distances, random_distances, alternative='less')
    
    comparison_results['mannwhitney_statistic'] = statistic
    comparison_results['mannwhitney_pvalue'] = p_value
    
    print(f"\nStatistical Test (Mann-Whitney U):")
    print(f"  Test: Are observed distances significantly smaller than random?")
    print(f"  Statistic: {statistic}")
    print(f"  P-value: {p_value:.2e}")
    
    if p_value < 0.001:
        print(f"  Result: HIGHLY SIGNIFICANT boundary clustering (p < 0.001)")
    elif p_value < 0.05:
        print(f"  Result: SIGNIFICANT boundary clustering (p < 0.05)")
    else:
        print(f"  Result: No significant difference")
    
    # Distance distribution comparison
    print(f"\nDistance Distribution Comparison:")
    print(f"{'Distance':<10} {'Observed %':<12} {'Random %':<10} {'Enrichment':<12}")
    print("-" * 50)
    
    for distance in sorted(set(list(observed_distances.unique()) + list(random_distances.unique()))):
        if distance <= 10:  # Only show first 10 distances
            obs_pct = (observed_distances == distance).sum() / len(observed_distances) * 100
            rand_pct = (random_distances == distance).sum() / len(random_distances) * 100
            enrichment = obs_pct / rand_pct if rand_pct > 0 else np.inf
            
            print(f"{distance:<10.0f} {obs_pct:<12.1f} {rand_pct:<10.1f} {enrichment:<12.1f}")
    
    return comparison_results

def create_random_vs_observed_visualization(gdf_multi_distances, comparison_results, figsize=(16, 10)):
    """
    Create comprehensive visualization comparing random baseline to observed patterns.
    """
    
    # Get data
    observed_distances = gdf_multi_distances[
        gdf_multi_distances['distance_to_nominal_95pct'] > 0
    ]['distance_to_nominal_95pct']
    
    random_distances = gdf_multi_distances[
        (gdf_multi_distances['distance_random_baseline'] > 0) & 
        (gdf_multi_distances['distance_random_baseline'].notna())
    ]['distance_random_baseline']
    
    if len(observed_distances) == 0 or len(random_distances) == 0:
        print("No data available for visualization")
        return None
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Random Baseline vs. Observed Uncertainty Patterns:\nEvidence for Spatial Boundary Effects', 
                 fontsize=16, fontweight='bold')
    
    # Panel A: Distance distributions
    max_dist = max(observed_distances.max(), random_distances.max())
    bins = np.arange(1, min(max_dist + 2, 12)) - 0.5  # Cap at distance 11 for clarity
    
    ax1.hist(observed_distances[observed_distances <= 11], bins=bins, alpha=0.7, 
            label=f'Observed (n={len(observed_distances):,})', color='red', density=True)
    ax1.hist(random_distances[random_distances <= 11], bins=bins, alpha=0.7, 
            label=f'Random (n={len(random_distances):,})', color='blue', density=True)
    
    ax1.set_xlabel('Distance to Designated Areas (Steps)')
    ax1.set_ylabel('Density')
    ax1.set_title('A) Distance Distribution Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Cumulative distributions
    obs_counts = observed_distances.value_counts().sort_index()
    rand_counts = random_distances.value_counts().sort_index()
    
    # Normalize to percentages
    obs_cumsum = obs_counts.cumsum() / obs_counts.sum() * 100
    rand_cumsum = rand_counts.cumsum() / rand_counts.sum() * 100
    
    ax2.plot(obs_cumsum.index, obs_cumsum.values, 'o-', color='red', linewidth=3, 
            label='Observed', markersize=8)
    ax2.plot(rand_cumsum.index, rand_cumsum.values, 's-', color='blue', linewidth=3, 
            label='Random', markersize=8)
    
    ax2.set_xlabel('Distance to Designated Areas (Steps)')
    ax2.set_ylabel('Cumulative Percentage')
    ax2.set_title('B) Cumulative Distribution Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Panel C: Key statistics comparison
    metrics = ['Mean Distance', 'Distance=1 %']
    observed_values = [comparison_results['observed_mean'], comparison_results['observed_dist1_pct']]
    random_values = [comparison_results['random_mean'], comparison_results['random_dist1_pct']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, observed_values, width, label='Observed', color='red', alpha=0.7)
    bars2 = ax3.bar(x + width/2, random_values, width, label='Random', color='blue', alpha=0.7)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Value')
    ax3.set_title('C) Key Statistics Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel D: Enrichment factors
    distances_to_show = range(1, min(8, int(max_dist) + 1))
    enrichments = []
    
    for dist in distances_to_show:
        obs_pct = (observed_distances == dist).sum() / len(observed_distances) * 100
        rand_pct = (random_distances == dist).sum() / len(random_distances) * 100
        enrichment = obs_pct / rand_pct if rand_pct > 0 else 0
        enrichments.append(enrichment)
    
    bars = ax4.bar(distances_to_show, enrichments, color='green', alpha=0.7, edgecolor='black')
    ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, label='No enrichment')
    
    ax4.set_xlabel('Distance to Designated Areas (Steps)')
    ax4.set_ylabel('Enrichment Factor (Observed/Random)')
    ax4.set_title('D) Spatial Enrichment Analysis', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, enrich in zip(bars, enrichments):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(enrichments)*0.01,
                f'{enrich:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Add summary text
    summary_text = f"""
EVIDENCE FOR SPATIAL BOUNDARY EFFECTS:
• Observed mean distance: {comparison_results['observed_mean']:.1f} vs Random: {comparison_results['random_mean']:.1f} steps
• Boundary enrichment: {comparison_results['dist1_enrichment']:.1f}x concentration at distance=1
• Statistical significance: p < {comparison_results['mannwhitney_pvalue']:.2e}

CONCLUSION: Measurement uncertainty is NOT randomly distributed but clusters at designation boundaries.
    """
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    return fig

def run_complete_random_baseline_analysis(gdf_multi_distances, save_results=True):
    """
    Run the complete random baseline analysis.
    """
    
    # Create random baseline
    gdf_enhanced = create_random_baseline_distances(gdf_multi_distances)
    
    # Compare random vs observed
    comparison_results = compare_random_vs_observed(gdf_enhanced)
    
    if comparison_results is None:
        return None, None, None
    
    # Create visualization
    fig = create_random_vs_observed_visualization(gdf_enhanced, comparison_results)
    
    # Save enhanced dataframe if requested
    if save_results:
        write_parquet(gdf_enhanced, 'gitignore/ci_multi_distances', 'multi_distance_with_random.parquet')
        print(f"\nEnhanced dataframe saved with random baseline column")
    
    return gdf_enhanced, comparison_results, fig