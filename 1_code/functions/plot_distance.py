import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yaml import warnings
from io_utils import read_parquet
import matplotlib.patches as patches
from matplotlib.patches import Circle
from collections import Counter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import time
import os
import io
from matplotlib.backends.backend_svg import FigureCanvasSVG

def create_spatial_network_plot(data_path='gitignore/one_percent_distance/moe_random_distance.parquet'):

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 22,
        'axes.linewidth': 1.5,
        'lines.linewidth': 3.0,
        'patch.linewidth': 1.2,
    })

    folder, filename = os.path.split(data_path)
    df = read_parquet(folder, filename)


    moe_tracts = df[(df['count_design'] >= 1) & (df['nom_desig'] != 1) & df['moe_distance'].notna()]
    moe_distances = moe_tracts['moe_distance'].values

    random_tracts = df[df['random_distance'].notna()]
    random_distances = random_tracts['random_distance'].values

    moe_color_primary = "#a23e48"
    random_color_primary = '#264653'
    text_color = '#4A4A4A'
    edge_color = "#7f8c8d"


    moe_dist_counts = Counter(moe_distances)
    max_moe_count = max(moe_dist_counts.values()) if moe_dist_counts else 1
    max_dist = min(12, max(max(moe_distances, default=0), max(random_distances, default=0)))

    random_dist_counts = Counter(random_distances)
    max_random_count = max(random_dist_counts.values()) if random_dist_counts else 1

    # Balanced width with proper left padding
    fig, ax = plt.subplots(figsize=(11.0, 4.5), facecolor='white')
    ax.set_facecolor('white')

    # Balanced plotting space 
    ax.set_xlim(-0.25, 2.5)
    ax.set_ylim(-0.18, 1.05)
    ax.set_aspect('auto')

    # Layout settings - balanced positioning
    y_line = 0.5
    start_x = 0.2  # Balanced position
    plot_width = 1.5  # Slightly increased to prevent number overlap
    line_end_x = start_x + plot_width

    # Distance direction arrow
    ax.annotate('', xy=(line_end_x, y_line), xytext=(start_x - 0.02, y_line),
                arrowprops=dict(arrowstyle='->', lw=6, color=text_color, alpha=0.8))

    # Draw step lines, MOE bubbles, and random bars
    for step in range(1, max_dist + 1):
        step_x = start_x + ((step - 1) / (max_dist - 1)) * plot_width

        # Tick lines
        ax.plot([step_x, step_x], [y_line - 0.05, y_line + 0.05],
                color='#333333', linewidth=5, alpha=0.8)

        # MOE bubble
        moe_count = moe_dist_counts.get(step, 0)
        if moe_count > 0:
            visual_unit = 0.02  # 1 tract shown as radius 0.05 units
            circle_radius = np.sqrt(moe_count) * visual_unit  # absolute count-based

            circle = Circle((step_x, y_line), circle_radius,
                            facecolor=moe_color_primary, alpha=0.85,
                            edgecolor=edge_color, linewidth=1, zorder=3)
            ax.add_patch(circle)

        random_count = random_dist_counts.get(step, 0)

        if random_count > 0:
            visual_unit_bar = 0.001  
            bar_height = random_count * visual_unit_bar

            bar_width = 0.035
            for direction in [-1, 1]:
                rect = patches.Rectangle(
                    (step_x - bar_width / 2,
                    y_line if direction == 1 else y_line - bar_height),
                    bar_width, bar_height,
                    facecolor=random_color_primary, alpha=0.9,
                    edgecolor=edge_color, linewidth=1, zorder=5
                )
                ax.add_patch(rect)

            
        ax.text(step_x, -0.0, f"{step}", fontsize=13,
                ha='center', va='top', color=text_color)


    # Bottom label - removed italic styling
    mid_x = (start_x + line_end_x) / 2
    ax.text(mid_x, -0.11, 'Network Steps to Closest Selected Tract',
            fontsize=17, ha='center', va='center',
            color=text_color)


    legend_ax = fig.add_axes([0.79, 0.35, 0.2, 0.6])
    legend_ax.axis('off')

    # Reference counts for legend bubbles and bars (real values from your data)
    moe_reference_counts = [10, 100]  # Adjust if different in your dataset
    random_reference_counts = [10, 100 ]

    visual_unit = 0.02  # match main plot

    # X positions
    bubble_x = 0.25
    bar_x = 0.75
    text_offset = 0.08
    spacing = 0.30

    # Plot MOE legend bubbles
    for i, val in enumerate(moe_reference_counts[::-1]):
        y = 0.3 + i * spacing
        radius = np.sqrt(val) * visual_unit
        circ = Circle((bubble_x, y), radius,
                      color=moe_color_primary, alpha=0.85,
                      edgecolor=edge_color, lw=1.2)
        legend_ax.add_patch(circ)
        legend_ax.text(bubble_x + text_offset + 0.15, y, f'{val}', va='center', ha='left',
                       fontsize=11, fontweight='bold', color='black')

    # Plot Random legend bars # Must match main plot

    for i, val in enumerate(random_reference_counts[::-1]):
        y = 0.3 + i * spacing
        bar_height = val * visual_unit_bar  # absolute count-based height
        bar_width = 0.05
        bar = patches.Rectangle((bar_x - bar_width / 2, y - bar_height / 2),
                                bar_width, bar_height,
                                facecolor=random_color_primary, alpha=0.9,
                                edgecolor=edge_color, lw=1.2)
        legend_ax.add_patch(bar)

    # Legend labels
    legend_ax.text(bubble_x, -0.05, 'MOE\nUncertainty', ha='center', va='top',
                   fontsize=14, color=text_color)
    legend_ax.text(bar_x, -0.05, 'Random\nUncertainty', ha='center', va='top',
                   fontsize=14, color=text_color)

    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08)

    fig.suptitle("Number of Tracts and Network Steps to Closest Selected Tract", fontsize=18, color=text_color)

    return fig


def create_distribution_plot(data_path='gitignore/one_percent_distance/moe_random_distance.parquet'):
    """
    Create the distribution comparison plot
    """
    # Set style specifically for bar chart - changed to sans-serif
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.0,
    })

    # Read the data
    folder, filename = os.path.split(data_path)
    df = read_parquet(folder, filename)


    # Prepare data
    moe_tracts = df[(df['count_design'] >= 1) & (df['nom_desig'] != 1) & df['moe_distance'].notna()]
    moe_distances = moe_tracts['moe_distance'].values

    random_tracts = df[df['random_distance'].notna()]
    random_distances = random_tracts['random_distance'].values

    # Define colors - matching the spatial network plot
    moe_color_primary = '#8B1538'
    random_color_primary = '#1B4A73'
    grid_color = '#E5E5E5'
    text_color = '#2C2C2C'

    moe_color_primary = "#a23e48"
    random_color_primary = '#264653'
    text_color = '#4A4A4A'
    edge_color = "#7f8c8d"

    # Create figure optimized for half page width
    fig, ax = plt.subplots(figsize=(5.4, 4.0), facecolor='white')  # Half page width
    ax.set_facecolor('white')

    # Calculate distributions
    max_distance = max(max(moe_distances, default=0), max(random_distances, default=0))
    distances = range(1, int(max_distance) + 1)

    moe_counts = [np.sum(moe_distances == d) for d in distances]
    random_counts = [np.sum(random_distances == d) for d in distances]

    # Convert to percentages
    moe_pct = np.array(moe_counts) / len(moe_distances) * 100 if len(moe_distances) > 0 else np.zeros(len(distances))
    random_pct = np.array(random_counts) / len(random_distances) * 100 if len(random_distances) > 0 else np.zeros(len(distances))

    x = np.arange(len(distances))
    width = 0.4  # Increased from 0.35 to make bars thicker

    # Bar charts with optimized styling - thin light gray edges
    bars1 = ax.bar(x - width/2, moe_pct, width, label='MOE Uncertainty',
                   color=moe_color_primary, alpha=0.85,
                   edgecolor='#f8f9fa', linewidth=1)

    ax.set_xlabel('Network Steps to Closest Selected Tract',
                  fontsize=13, color=text_color)
    ax.set_ylabel('Margin of Error %of Tracts',
                  fontsize=13, color=moe_color_primary)
    ax.tick_params(axis='y', labelcolor=moe_color_primary, width=1.2, labelsize=11)
    ax.tick_params(axis='x', colors=text_color, width=1.2, labelsize=11)

    # Secondary y-axis for Random
    ax_twin = ax.twinx()
    bars2 = ax_twin.bar(x + width/2, random_pct, width, label='Random Sample',
                        color=random_color_primary, alpha=0.85,
                        edgecolor='#f8f9fa', linewidth=1)

    ax_twin.set_ylabel('Random %of Tracts',
                       fontsize=13, color=random_color_primary)
    ax_twin.tick_params(axis='y', labelcolor=random_color_primary, width=1.2, labelsize=11)

    ax.set_title('Step Distribution Comparison',
                 fontsize=18, pad=20, color=text_color)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in distances])
    ax.set_xlim(-0.5, 14.5)

    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    legend = ax.legend(lines1 + lines2, labels1 + labels2,
                       loc='upper right', fontsize=12, frameon=True,
                       fancybox=True, shadow=False, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(grid_color)

    ax.grid(True, alpha=0.3, color=grid_color, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)

    # Enhanced spines
    for spine in ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1.2)
    for spine in ax_twin.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1.2)

    plt.tight_layout()
    return fig

def create_cumulative_plot(data_path='gitignore/one_percent_distance/moe_random_distance.parquet'):

    # Set style specifically for line plot - changed to sans-serif
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.6,
        'lines.linewidth': 3.5,
    })

    # Read the data
    folder, filename = os.path.split(data_path)
    df = read_parquet(folder, filename)

    # Prepare data
    moe_tracts = df[(df['count_design'] >= 1) & (df['nom_desig'] != 1) & df['moe_distance'].notna()]
    moe_distances = moe_tracts['moe_distance'].values

    random_tracts = df[df['random_distance'].notna()]
    random_distances = random_tracts['random_distance'].values

    # Define colors - matching the other plots
    moe_color_primary = '#8B1538'
    random_color_primary = '#1B4A73'
    grid_color = '#E5E5E5'
    text_color = '#2C2C2C'

    moe_color_primary = "#a23e48"
    random_color_primary = '#264653'
    text_color = '#4A4A4A'
    edge_color = "#7f8c8d"

    # Create figure optimized for half page width
    fig, ax = plt.subplots(figsize=(5.4, 4.0), facecolor='white')  # Half page width
    ax.set_facecolor('white')

    # Calculate distributions and cumulative
    max_distance = max(max(moe_distances, default=0), max(random_distances, default=0))
    distances = range(1, int(max_distance) + 1)

    moe_counts = [np.sum(moe_distances == d) for d in distances]
    random_counts = [np.sum(random_distances == d) for d in distances]

    # Convert to percentages
    moe_pct = np.array(moe_counts) / len(moe_distances) * 100 if len(moe_distances) > 0 else np.zeros(len(distances))
    random_pct = np.array(random_counts) / len(random_distances) * 100 if len(random_distances) > 0 else np.zeros(len(distances))

    # Calculate cumulative percentages
    moe_cumulative = np.cumsum(moe_pct)
    random_cumulative = np.cumsum(random_pct)

    # Line plots with optimized styling
    line1 = ax.plot(distances, moe_cumulative, 'o-', color=moe_color_primary,
                    linewidth=3.5, markersize=8, label='MOE Uncertainty',
                    markerfacecolor='white', markeredgewidth=2.5,
                    markeredgecolor=moe_color_primary, alpha=0.9)

    line2 = ax.plot(distances, random_cumulative, 's-', color=random_color_primary,
                    linewidth=3.5, markersize=8, label='Random Sample',
                    markerfacecolor='white', markeredgewidth=2.5,
                    markeredgecolor=random_color_primary, alpha=0.9)

    # Subtle confidence intervals
    ax.fill_between(distances, moe_cumulative * 0.98, moe_cumulative * 1.02,
                    alpha=0.12, color=moe_color_primary, label='_nolegend_')
    ax.fill_between(distances, random_cumulative * 0.96, random_cumulative * 1.04,
                    alpha=0.12, color=random_color_primary, label='_nolegend_')

    ax.set_xlabel('Network Steps to Closest Selected Tract',
                  fontsize=14, color=text_color)
    ax.set_ylabel('Cumulative % of Uncertain Tracts',
                  fontsize=13, color=text_color)
    ax.set_title('Cumulative Uncertainty Capture',
                 fontsize=18, pad=20, color=text_color)

    # Legend
    legend = ax.legend(loc='lower right', fontsize=12, frameon=True,
                       fancybox=True, shadow=False, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(grid_color)

    ax.grid(True, alpha=0.3, color=grid_color, linestyle='-', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 105)

    ax.tick_params(axis='both', which='major', labelsize=11,
                   colors=text_color, width=1.2)

    # Enhanced spines
    for spine in ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1.2)

    ax.set_xlim(1, 15)
    plt.tight_layout()
    return fig


def create_panel_plot(data_path='gitignore/one_percent_distance/moe_random_distance.parquet'):
    """
    Create a panel plot using SVGs of the individual plots
    """
    # Create output directory
    output_dir = '../2_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create the individual figures
    fig1 = create_spatial_network_plot(data_path)
    fig2 = create_distribution_plot(data_path)
    fig3 = create_cumulative_plot(data_path)
    
    # Create SVG strings for each figure
    def fig_to_svg(fig):
        output = io.StringIO()
        canvas = FigureCanvasSVG(fig)
        canvas.print_svg(output)
        svg_string = output.getvalue()
        output.close()
        return svg_string
    
    svg1 = fig_to_svg(fig1)
    svg2 = fig_to_svg(fig2)
    svg3 = fig_to_svg(fig3)
    
    # Close individual figures
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    
    # Create HTML layout combining the SVGs
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .panel-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: auto auto;
                gap: 20px;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
                font-family: Arial, sans-serif;
            }}
            .panel-a {{
                grid-column: 1 / -1;
                position: relative;
            }}
            .panel-b {{
                grid-column: 1;
                position: relative;
            }}
            .panel-c {{
                grid-column: 2;
                position: relative;
            }}
            .panel-label {{
                position: absolute;
                top: 10px;
                left: 10px;
                font-size: 18px;
                font-weight: bold;
                background: white;
                padding: 5px 8px;
                border-radius: 3px;
                z-index: 100;
            }}
            .svg-container {{
                width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <div class="panel-container">
            <div class="panel-a">
                <div class="panel-label">A</div>
                <div class="svg-container">{svg1}</div>
            </div>
            <div class="panel-b">
                <div class="panel-label">B</div>
                <div class="svg-container">{svg2}</div>
            </div>
            <div class="panel-c">
                <div class="panel-label">C</div>
                <div class="svg-container">{svg3}</div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save as HTML file in the specified directory
    html_path = os.path.join(output_dir, 'panel_plot.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Panel plot saved as '{html_path}'")
    print("You can open this file in a web browser or convert to PDF/PNG as needed")
        
    return html_content

# Alternative: Save individual SVGs for manual combination
def save_individual_svgs():
    """
    Save individual plots as SVG files for manual combination
    """
    # Create output directory
    output_dir = '../0_data/gitignore/distance_plot'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create and save each figure
    fig1 = create_spatial_network_plot()
    svg_path1 = os.path.join(output_dir, 'spatial_network_plot.svg')
    fig1.savefig(svg_path1, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig1)
    
    fig2 = create_distribution_plot()
    svg_path2 = os.path.join(output_dir, 'distribution_plot.svg')
    fig2.savefig(svg_path2, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig2)
    
    fig3 = create_cumulative_plot()
    svg_path3 = os.path.join(output_dir, 'cumulative_plot.svg')
    fig3.savefig(svg_path3, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig3)
    
    print("Individual SVG files saved:")
    print(f"- {svg_path1}")
    print(f"- {svg_path2}") 
    print(f"- {svg_path3}")
    print("These can be combined in any vector graphics software (Inkscape, Illustrator, etc.)")


def export_html_to_tiff(html_path, tiff_path, width=1600, height=1200, delay=2, dpi=600):
    """
    Renders an HTML file in headless Chrome and saves it as a high-resolution TIFF.
    """
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument(f"--window-size={width},{height}")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--force-device-scale-factor=2")  # improves retina scaling

    # Initialize browser
    driver = webdriver.Chrome(options=chrome_options)

    try:
        abs_path = "file://" + os.path.abspath(html_path)
        driver.get(abs_path)

        time.sleep(delay)  # Wait for rendering

        screenshot_path = os.path.splitext(tiff_path)[0] + "_temp.png"
        driver.save_screenshot(screenshot_path)

        # Open and convert to TIFF
        img = Image.open(screenshot_path).convert("RGB")
        img.save(tiff_path, dpi=(dpi, dpi), format='TIFF')
        print(f"Saved TIFF at: {tiff_path}")

        os.remove(screenshot_path)  # Cleanup

    finally:
        driver.quit()