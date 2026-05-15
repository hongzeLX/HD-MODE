import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams, gridspec
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings('ignore')

def set_modern_style():
    rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 11,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 9,
        
        'figure.figsize': (16, 12), 
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.direction': 'out', 
        'ytick.direction': 'out',
        'xtick.top': False,
        'ytick.right': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

set_modern_style()

df = pd.read_csv('results/experiment_metrics_log.csv')

error_cols = ['Train_Error_U', 'Test_Error_U_OOD', 'Generator_Error_Phi']
for col in error_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') * 100 

EQUATIONS = ['Fokker_Planck', 'KPZ', 'Allen_Cahn', 'Fisher_KPP', 'Zeldovich']
EQ_SHORT = ['F-P', 'KPZ','A-C', 'F-KPP', 'Zeldo']
DIMENSIONS = [20, 50, 100]
NOISE_LEVELS = [1, 5, 10, 20]
METRICS = ['Train_Error_U', 'Test_Error_U_OOD', 'Generator_Error_Phi']
METRIC_LABELS = ['Training Error', 'Test Error (OOD)', 'Generator Error']

DIM_COLORS = {
    20: '#FFA5A4',   
    50: '#93F0F4',   
    100: '#9998FF'   
}

def format_label(val):
    if val >= 10: return f"{val:.1f}"
    elif val >= 1: return f"{val:.2f}"
    else: return f"{val:.3f}"

row_ylim_max = {}
for metric in METRICS:
    max_val = df[metric].max()
    if pd.isna(max_val) or max_val == 0: 
        max_val = 1.0
    row_ylim_max[metric] = max_val * 1.25  

fig = plt.figure()

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.08) 
bar_width = 0.25
x_positions = np.arange(len(EQUATIONS))

for row_idx, metric in enumerate(METRICS):
    current_row_ylim = row_ylim_max[metric] 
    
    for col_idx, noise in enumerate(NOISE_LEVELS):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        subset = df[(df['Noise_Level'].astype(str).str.replace('%', '').astype(float) == noise)]
        
        ax.set_ylim(0, current_row_ylim)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
        
        for eq_idx, equation in enumerate(EQUATIONS):
            eq_data = subset[subset['Equation'] == equation]
            for dim_idx, dim in enumerate(DIMENSIONS):
                dim_data = eq_data[eq_data['Dimension'] == dim]
                
                if not dim_data.empty and not pd.isna(dim_data[metric].values[0]):
                    val = dim_data[metric].values[0]
                    x_pos = x_positions[eq_idx] + (dim_idx - 1) * bar_width
                    
                    ax.bar(x_pos, val, width=bar_width, color=DIM_COLORS[dim],
                           edgecolor='black', linewidth=0.7, zorder=3)
                    
                    ax.text(x_pos, val + (current_row_ylim * 0.015), format_label(val),
                            ha='center', va='bottom', rotation=90, 
                            fontsize=6, color='#222222', zorder=4)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(EQ_SHORT, rotation=0, ha='center', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3, zorder=1)
        ax.set_axisbelow(True)
        
        if col_idx == 0:
            ax.set_ylabel('Error (%)', fontsize=16, labelpad=8)

            ax.annotate(METRIC_LABELS[row_idx], xy=(-0.28, 0.5), xycoords='axes fraction',
                        fontsize=18, fontweight='bold', rotation=90,
                        ha='center', va='center')
        else:
            ax.tick_params(labelleft=False)
            
        panel_label = chr(97 + row_idx * 4 + col_idx)
        x_offset = -0.15 if col_idx == 0 else -0.05 
        ax.text(x_offset, 1.08, f'{panel_label}', transform=ax.transAxes, 
                fontsize=24, fontweight='bold')
        
        if row_idx == 0:
            ax.set_title(f'Noise: {noise}%', fontsize=20, pad=15, fontweight='bold')

legend_elements = [mpatches.Patch(facecolor=DIM_COLORS[d], edgecolor='black', 
                   linewidth=0.7, label=f'{d} Dimensions') for d in DIMENSIONS]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.03),
           ncol=3, frameon=False, fontsize=16)
os.makedirs("results", exist_ok=True)
out_path = "results/Error_Analysis.pdf"
plt.savefig(out_path, facecolor='white', dpi=300, bbox_inches='tight')
plt.show()