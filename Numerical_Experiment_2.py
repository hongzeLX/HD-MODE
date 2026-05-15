import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.metrics import r2_score

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.linewidth': 1.0,
    'figure.facecolor': 'white',
    'pdf.fonttype': 42
})

def build_correlation_heatmaps(data_dir="results/plot_data"):

    if not os.path.exists(data_dir):
        print("Error: Data directory not found.")
        return

    raw_data = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    for file in files:
        parts = file.replace('.npz', '').split('_')
        noise_val = int(parts[-1])
        dim_val = int(parts[-3].replace('D',''))
        eq_name = "_".join(parts[1:-3])
        data = np.load(os.path.join(data_dir, file))
        u_true, u_pred = data['u_ex'].flatten(), data['u_pr'].flatten()
        raw_data.append({
            "Equation": eq_name, "Dimension": dim_val, "Noise": noise_val, 
            "R2": r2_score(u_true, u_pred), "r": np.corrcoef(u_true, u_pred)[0, 1]
        })

    df = pd.DataFrame(raw_data)
    target_order = ["Fokker_Planck", "KPZ", "Allen_Cahn", "Fisher_KPP", "Zeldovich"]
    equations = [e for e in target_order if e in df['Equation'].unique()]

    global_r2_min = max(0.85, np.floor(df['R2'].min() * 50) / 50) 
    global_r_min = max(0.90, np.floor(df['r'].min() * 50) / 50)   
    print(f"[*] Global R2 range: {global_r2_min} to 1.00")
    print(f"[*] Global r range: {global_r_min} to 1.00")

    fig = plt.figure(figsize=(15, 16))
    
    outer_gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    def draw_panel(outer_idx, metric_key, panel_label, metric_title, v_min):

        inner_gs = GridSpecFromSubplotSpec(2, 7, subplot_spec=outer_gs[outer_idx], 
                                           width_ratios=[1,1,1,1,1,1, 0.2], 
                                           hspace=0.35, wspace=0.7)
        
        ax_bg = fig.add_subplot(outer_gs[outer_idx])
        ax_bg.axis('off')
        ax_bg.text(-0.02, 1.12, panel_label, transform=ax_bg.transAxes, fontsize=22, fontweight='bold')
        ax_bg.text(0.03, 1.11, metric_title, transform=ax_bg.transAxes, fontsize=22, fontweight='bold')

        last_im = None 
        
        for i, eq in enumerate(equations):
            if i < 2: 
                ax_pos = inner_gs[0, 1+i*2 : 3+i*2]
            else: 
                ax_pos = inner_gs[1, (i-2)*2 : (i-2)*2 + 2]
            
            ax = fig.add_subplot(ax_pos)
            eq_df = df[df['Equation'] == eq]
            pivot = eq_df.pivot(index="Noise", columns="Dimension", values=metric_key)
            
            im = sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlGnBu", 
                            vmin=v_min, vmax=1.0, cbar=False,
                            annot_kws={"size": 8}, linewidths=0.5)
            last_im = im
            
            ax.set_title(eq.replace('_', ' '), fontsize=11, fontweight='semibold', pad=10)
            ax.set_xlabel("Dimension ($D$)", fontsize=9)
            ax.set_ylabel("Noise (%)", fontsize=9)

        cax = fig.add_subplot(inner_gs[:, 6]) 
        plt.colorbar(last_im.get_children()[0], cax=cax, label=metric_key)
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 + pos.height*0.15, pos.width * 0.8, pos.height * 0.7])

    draw_panel(0, "R2", "a", "Coefficient of Determination ($R^2$ Score)", global_r2_min)
    draw_panel(1, "r", "b", "Pearson Correlation Coefficient ($r$)", global_r_min)

    os.makedirs("results", exist_ok=True)
    out_path = "results/Standardized_Heatmaps.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[*] Standardized heatmap saved to: {out_path}")

if __name__ == "__main__":
    build_correlation_heatmaps()