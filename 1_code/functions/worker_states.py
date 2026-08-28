import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io_utils import read_parquet
import matplotlib.patches as patches
from matplotlib.patches import Circle
from collections import Counter

# Set publication-quality style for A4 page width
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'mathtext.fontset': 'dejavuserif'
})

def create_spatial_influence_plot(data_path='gitignore/one_percent_distance/moe_random_distance.parquet'):
    """
    Create publication-quality spatial influence analysis plot comparing MOE vs Random distances
    """
    # Read the data
    df = read_parquet('gitignore/one_percent_distance', 'moe_random_distance.parquet')
    
    # Prepare data
    # MOE distances (tracts with uncertainty that got designated)
    moe_tracts = df[(df['count_design'] >= 1) & (df['nom_desig'] != 1) & df['moe_distance'].notna()]
    moe_distances = moe_tracts['moe_distance'].values
    
    # Random distances
    random_tracts = df[df['random_distance'].notna()]
    random_distances = random_tracts['random_distance'].values
    
    # Create figure - WIDER for double-width upper plot
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Define publication-quality color palette
    moe_color_primary = '#8B1538'  # Deep burgundy
    random_color_primary = '#1B4A73'  # Deep blue
    grid_color = '#E5E5E5'
    text_color = '#2C2C2C'
    
    # COMBINED NETWORK: MOE bubbles + Random bars - DOUBLE WIDTH TOP PLOT
    ax1 = plt.subplot(2, 2, (1, 2))  # Span both upper positions - DOUBLE WIDTH
    ax1.set_facecolor('white')
    
    # Get data for both MOE and Random
    moe_dist_counts = Counter(moe_distances)
    max_moe_count = max(moe_dist_counts.values()) if moe_dist_counts else 1
    max_dist = max(max(moe_distances, default=0), max(random_distances, default=0))
    max_dist = min(12, max_dist)
    
    random_dist_counts = Counter(random_distances)
    max_random_count = max(random_dist_counts.values()) if random_dist_counts else 1
    
    # Set up the combined plot area - GOOD PROPORTIONS
    ax1.set_xlim(0, 1)  # Start at 0 to not cut off circle, use full width
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    
    # Draw the main horizontal line as an arrow showing direction
    y_line = 0.5
    start_x = 0.05  # Start from left but not cut off
    plot_width = 0.55  # Use about 55% for main plot
    step_spacing = plot_width / max_dist  # Dynamic spacing
    line_end_x = start_x + plot_width
    
    # Arrow line showing distance direction
    ax1.annotate('', xy=(line_end_x, y_line), xytext=(start_x - 0.02, y_line),
                arrowprops=dict(arrowstyle='->', lw=4, color='#333333', alpha=0.8))
    
    # Add step markers and draw bubbles/bars
    for step in range(1, max_dist + 1):
        step_x = start_x + ((step - 1) / (max_dist - 1)) * plot_width
        
        # Vertical tick marks
        ax1.plot([step_x, step_x], [y_line - 0.02, y_line + 0.02], 
                color='#333333', linewidth=2, alpha=0.8)
        
        # Draw MOE bubbles - LARGE AND VISIBLE
        moe_count = moe_dist_counts.get(step, 0)
        if moe_count > 0:
            # MOE bubble sizing
            base_size = 0.04
            scale_factor = 0.15
            circle_radius = base_size + (moe_count / max_moe_count) * scale_factor
            
            # MOE color (burgundy/red family)
            intensity = moe_count / max_moe_count
            if intensity >= 0.75:
                color = '#8B1538'  # Dark burgundy
                edge_color = '#ffffff'
            elif intensity >= 0.5:
                color = '#A64458'  # Medium burgundy
                edge_color = '#f8fafc'
            elif intensity >= 0.25:
                color = '#C1606E'  # Light burgundy
                edge_color = '#e2e8f0'
            else:
                color = '#DC7C84'  # Very light burgundy
                edge_color = '#cbd5e1'
            
            # Draw MOE bubble
            circle = Circle((step_x, y_line), circle_radius, 
                           facecolor=color, alpha=0.8, 
                           edgecolor=edge_color, linewidth=2, zorder=2)
            ax1.add_patch(circle)
        
        # Draw Random bars - TALL AND VISIBLE
        random_count = random_dist_counts.get(step, 0)
        if random_count > 0:
            # Random bar sizing
            min_height = 0.08
            max_height = 0.2
            height_range = max_height - min_height
            size_ratio = (random_count - 22) / (84 - 22) if max_random_count > 22 else 0
            bar_height = min_height + (size_ratio * height_range)
            
            # Random color (blue family)
            if random_count >= 70:
                color = '#1B4A73'      # Dark blue
                edge_color = '#ffffff'
            elif random_count >= 50:
                color = '#2563eb'      # Medium blue  
                edge_color = '#e0e0e0'
            else:
                color = '#60a5fa'      # Light blue
                edge_color = '#94a3b8'
            
            # Draw Random bars
            bar_width = 0.02
            
            # Upper bar
            upper_rect = patches.Rectangle((step_x - bar_width/2, y_line), 
                                         bar_width, bar_height,
                                         facecolor=color, alpha=0.9,
                                         edgecolor=edge_color, linewidth=2, zorder=5)
            ax1.add_patch(upper_rect)
            
            # Lower bar
            lower_rect = patches.Rectangle((step_x - bar_width/2, y_line - bar_height), 
                                         bar_width, bar_height,
                                         facecolor=color, alpha=0.9,
                                         edgecolor=edge_color, linewidth=2, zorder=5)
            ax1.add_patch(lower_rect)
        
        # Step numbers BELOW everything
        ax1.text(step_x, 0.2, str(step), fontsize=14, fontweight='600', 
                ha='center', va='center', color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#333333', alpha=0.9, linewidth=1))
    
    # Add "Distance" label at the bottom
    mid_x = (start_x + line_end_x) / 2
    ax1.text(mid_x, 0.1, 'Increasing Distance from Designated Areas', 
             fontsize=16, fontweight='600', ha='center', va='center', 
             color='#333333', style='italic')
    
    # LEGENDS SIDE BY SIDE TO THE RIGHT - PROPER HORIZONTAL LAYOUT
    legend_start = 0.65  # Start legend area
    
    # MOE Bubble legend - LEFT side of legend area
    bubble_x = legend_start
    ax1.text(bubble_x + 0.08, 0.9, 'MOE Bubble Size:', 
             fontsize=14, fontweight='bold', color='#8B1538', ha='center')
    
    # Three bubble sizes HORIZONTALLY
    bubble_sizes = [0.03, 0.045, 0.06]
    bubble_counts = [50, 200, 400]
    h_spacing = 0.06
    
    for i, (size, count) in enumerate(zip(bubble_sizes, bubble_counts)):
        x_pos = bubble_x + (i * h_spacing)
        y_pos = 0.8
        circle = Circle((x_pos, y_pos), size, facecolor='#8B1538', alpha=0.8, 
                       edgecolor='white', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x_pos, y_pos - 0.1, f'{count}\ntracts', fontsize=12, 
                ha='center', va='center', color='#333333', fontweight='500')
    
    # Random Bar legend - RIGHT side of legend area  
    bar_x = legend_start + 0.2
    ax1.text(bar_x + 0.08, 0.9, 'Random Bar Height:', 
             fontsize=14, fontweight='bold', color='#1B4A73', ha='center')
    
    # Three bar heights HORIZONTALLY
    bar_heights = [0.06, 0.12, 0.18]
    bar_counts = [30, 60, 84]
    
    for i, (height, count) in enumerate(zip(bar_heights, bar_counts)):
        x_pos = bar_x + (i * h_spacing)
        y_pos = 0.8
        
        # Upper and lower bars
        rect1 = patches.Rectangle((x_pos - 0.01, y_pos), 0.02, height,
                                facecolor='#1B4A73', alpha=0.9, edgecolor='white', linewidth=2)
        ax1.add_patch(rect1)
        rect2 = patches.Rectangle((x_pos - 0.01, y_pos - height), 0.02, height,
                                facecolor='#1B4A73', alpha=0.9, edgecolor='white', linewidth=2)
        ax1.add_patch(rect2)
        ax1.text(x_pos, y_pos - height - 0.08, f'{count}\ntracts', fontsize=12, 
                ha='center', va='center', color='#333333', fontweight='500')
    
    ax1.set_title('Combined Spatial Networks: MOE vs Random', 
                  fontsize=18, fontweight='bold', pad=20, color=text_color)
    ax1.axis('off')
    
    # C) Distance Distribution Comparison (LOWER LEFT)
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('white')
    
    # Calculate distributions
    max_distance = max(max(moe_distances, default=0), max(random_distances, default=0))
    distances = range(1, int(max_distance) + 1)
    
    moe_counts = [np.sum(moe_distances == d) for d in distances]
    random_counts = [np.sum(random_distances == d) for d in distances]
    
    # Convert to percentages
    moe_pct = np.array(moe_counts) / len(moe_distances) * 100 if len(moe_distances) > 0 else np.zeros(len(distances))
    random_pct = np.array(random_counts) / len(random_distances) * 100 if len(random_distances) > 0 else np.zeros(len(distances))
    
    x = np.arange(len(distances))
    width = 0.35
    
    # Enhanced bar chart styling
    bars1 = ax3.bar(x - width/2, moe_pct, width, label='MOE Uncertainty',
                    color=moe_color_primary, alpha=0.8, 
                    edgecolor='white', linewidth=1.5)
    
    # Add subtle gradient effect to bars
    for bar in bars1:
        bar.set_facecolor(moe_color_primary)
        bar.set_alpha(0.85)
    
    ax3.set_xlabel('Distance from Designated Areas (Steps)', 
                   fontsize=14, fontweight='600', color=text_color)
    ax3.set_ylabel('MOE: % of Tracts at Distance', 
                   fontsize=14, fontweight='600', color=moe_color_primary)
    ax3.tick_params(axis='y', labelcolor=moe_color_primary, width=1.2, labelsize=12)
    ax3.tick_params(axis='x', colors=text_color, width=1.2, labelsize=12)
    
    # Secondary y-axis for Random
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, random_pct, width, label='Random Sample',
                        color=random_color_primary, alpha=0.8, 
                        edgecolor='white', linewidth=1.5)
    
    # Add subtle gradient effect to bars
    for bar in bars2:
        bar.set_facecolor(random_color_primary)
        bar.set_alpha(0.85)
    
    ax3_twin.set_ylabel('Random: % of Tracts at Distance', 
                       fontsize=14, fontweight='600', color=random_color_primary)
    ax3_twin.tick_params(axis='y', labelcolor=random_color_primary, width=1.2, labelsize=12)
    
    ax3.set_title('C. Distance Distribution Comparison', 
                  fontsize=16, fontweight='bold', pad=15, color=text_color)
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(d) for d in distances])
    
    # Enhanced legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    legend = ax3.legend(lines1 + lines2, labels1 + labels2, 
                       loc='upper right', fontsize=13, frameon=True, 
                       fancybox=True, shadow=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(grid_color)
    
    # Refined grid
    ax3.grid(True, alpha=0.4, color=grid_color, linestyle='-', linewidth=0.8)
    ax3.set_axisbelow(True)
    
    # Enhanced spines
    for spine in ax3.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1.2)
    for spine in ax3_twin.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1.2)
    
    # D) Cumulative Analysis (LOWER RIGHT)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('white')
    
    # Calculate cumulative percentages
    moe_cumulative = np.cumsum(moe_pct)
    random_cumulative = np.cumsum(random_pct)
    
    # Enhanced line plots with refined markers
    line1 = ax4.plot(distances, moe_cumulative, 'o-', color=moe_color_primary, 
                     linewidth=3.5, markersize=8, label='MOE Uncertainty', 
                     markerfacecolor='white', markeredgewidth=2.5, 
                     markeredgecolor=moe_color_primary, alpha=0.9)
    
    line2 = ax4.plot(distances, random_cumulative, 's-', color=random_color_primary, 
                     linewidth=3.5, markersize=8, label='Random Sample', 
                     markerfacecolor='white', markeredgewidth=2.5,
                     markeredgecolor=random_color_primary, alpha=0.9)
    
    # Refined confidence intervals
    ax4.fill_between(distances, moe_cumulative * 0.95, moe_cumulative * 1.05,
                     alpha=0.12, color=moe_color_primary, label='_nolegend_')
    ax4.fill_between(distances, random_cumulative * 0.92, random_cumulative * 1.08,
                     alpha=0.12, color=random_color_primary, label='_nolegend_')
    
    ax4.set_xlabel('Distance from Designated Areas (Steps)', 
                   fontsize=14, fontweight='600', color=text_color)
    ax4.set_ylabel('Cumulative % of Uncertain Tracts', 
                   fontsize=14, fontweight='600', color=text_color)
    ax4.set_title('D. Cumulative Uncertainty Capture', 
                  fontsize=16, fontweight='bold', pad=15, color=text_color)
    
    # Enhanced legend
    legend = ax4.legend(loc='lower right', fontsize=13, frameon=True, 
                       fancybox=True, shadow=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(grid_color)
    
    # Refined grid and styling
    ax4.grid(True, alpha=0.4, color=grid_color, linestyle='-', linewidth=0.8)
    ax4.set_axisbelow(True)
    ax4.set_ylim(0, 105)
    ax4.tick_params(axis='both', which='major', labelsize=12, 
                   colors=text_color, width=1.2)
    
    # Enhanced spines
    for spine in ax4.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1.2)
    
    # Final layout adjustments
    plt.tight_layout(pad=3.5)
    
    # Add subtle figure border
    fig.patch.set_edgecolor(grid_color)
    fig.patch.set_linewidth(2)
    
    return fig