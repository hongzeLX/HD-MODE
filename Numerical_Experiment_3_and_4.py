import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'savefig.facecolor': 'white',
    'axes.linewidth': 1.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'scatter.edgecolors': 'none'
})
COLOR_TRUE = '#777777'
EQUATION_COLORS = {
    'Fokker_Planck': '#BBDF6F', 
    'Allen_Cahn':    '#AAD1CC',  
    'Fisher_KPP':    '#C0A3ED',  
    'Zeldovich':     '#F6B593',  
    'KPZ':           '#81CEAE'  
}

UNIFIED_CMAP = 'coolwarm'
NOISE_MAP_2x2 = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)} 

def extract_pure_phi(eq_name, data):
    u_subset = data['x_axis']
    if eq_name == 'Fokker_Planck':
        u_full, S_full, t_full = data['u_ex'], data['S'], data['t']
        return u_full + 2.0 * (S_full - t_full) * u_full - 2.0 * (S_full - t_full)**2 * u_full
    elif eq_name == 'Allen_Cahn': return u_subset - u_subset**3
    elif eq_name == 'Fisher_KPP': return u_subset - u_subset**2
    elif eq_name == 'Zeldovich': return u_subset**2 - u_subset**3
    elif eq_name == 'KPZ': return 0.5 * data['x_axis']
    else: return data['phi_tr']

def setup_mega_layout(fig):
    gs_main = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1.2, 1.0], hspace=0.3, top=0.96, bottom=0.06, left=0.06, right=0.94)
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], wspace=0.25)
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], wspace=0.25)
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[2], wspace=0.20)
    return gs_main, [gs_row1[0], gs_row1[1], gs_row2[0], gs_row2[1]], gs_row3

def draw_panel_label(fig, gs_area, letter, title, color):
    ax_title = fig.add_subplot(gs_area)
    ax_title.axis('off')
    ax_title.text(-0.12, 1.08, letter, transform=ax_title.transAxes, fontsize=34, fontweight='bold', va='bottom', ha='right', color='black')
    ax_title.text(-0.05, 1.08, title, transform=ax_title.transAxes, fontsize=32, fontweight='bold', va='bottom', ha='left', color=color)
    return ax_title

def create_mega_physics_response(equations, dimensions, noises):
    print("[*] Generating Mega Figure 1: Physics Response Dashboard...")
    fig = plt.figure(figsize=(24, 30))
    gs_main, top_grids, gs_bottom = setup_mega_layout(fig)
    
    for i, eq_name in enumerate(equations[:4]):
        gs_inner = gridspec.GridSpecFromSubplotSpec(len(noises), len(dimensions), subplot_spec=top_grids[i], hspace=0.15, wspace=0.15)
        draw_panel_label(fig, top_grids[i], chr(ord('a') + i), f"{eq_name.replace('_', ' ')}: Physics Response", 'black')

        for r, noise in enumerate(noises):
            for c, D in enumerate(dimensions):
                ax = fig.add_subplot(gs_inner[r, c])
                data_path = f"results/plot_data/data_{eq_name}_{D}D_noise_{int(noise*100)}.npz"
                if not os.path.exists(data_path):
                    ax.set_xticks([]); ax.set_yticks([]); continue
                
                data = np.load(data_path)
                pure_phi_tr = extract_pure_phi(eq_name, data)

                if eq_name != 'Fokker_Planck':
                    sort_idx = np.argsort(data['x_axis'])
                    ax.plot(data['x_axis'][sort_idx], pure_phi_tr[sort_idx], color=COLOR_TRUE,
                           linestyle='--', lw=1.5, alpha=0.8, zorder=3)
                else:
                    ax.scatter(data['u_ex'], pure_phi_tr, c=COLOR_TRUE, s=1, alpha=0.3, zorder=3)

                ax.scatter(data['x_axis'], data['phi_pr'], color=EQUATION_COLORS[eq_name], s=8, alpha=0.8, zorder=1)
                ax.grid(True, linestyle=':', alpha=0.4)
                
                if r == 0: ax.set_title(f"{D}D", fontsize=16, fontweight='bold', pad=15)
                if c == 0: ax.set_ylabel(f"Noise {noise*100:.0f}%", fontsize=14, fontweight='bold', labelpad=8)
                else: ax.set_yticklabels([])
                if r == len(noises) - 1: ax.set_xlabel(r"$U$", fontsize=14, fontweight='bold')
                else: ax.set_xticklabels([])

    eq_name = equations[4]
    ax_title = fig.add_subplot(gs_main[2])
    ax_title.axis('off')
    ax_title.text(-0.05, 1.15, 'e', transform=ax_title.transAxes, fontsize=34, fontweight='bold', va='bottom', ha='right', color='black')
    ax_title.text(-0.02, 1.15, f"KPZ: Physics Response", transform=ax_title.transAxes, fontsize=32, fontweight='bold', va='bottom', ha='left', color='black')
   
    for col_idx, D in enumerate(dimensions):
        gs_inner_e = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_bottom[col_idx], hspace=0.15, wspace=0.15)
        for n_idx, noise in enumerate(noises):
            r, c = NOISE_MAP_2x2[n_idx]
            ax = fig.add_subplot(gs_inner_e[r, c])
            
            data_path = f"results/plot_data/data_{eq_name}_{D}D_noise_{int(noise*100)}.npz"
            if not os.path.exists(data_path):
                ax.set_xticks([]); ax.set_yticks([]); continue
            
            data = np.load(data_path)
            pure_phi_tr = extract_pure_phi(eq_name, data)
            
            sort_idx = np.argsort(data['x_axis'])
            ax.plot(data['x_axis'][sort_idx], pure_phi_tr[sort_idx], 'k--', lw=1.5, alpha=0.8, zorder=3)
            ax.scatter(data['x_axis'], data['phi_pr'], color=EQUATION_COLORS[eq_name], s=8, alpha=0.8, zorder=1)
            ax.grid(True, linestyle=':', alpha=0.4)
            
            ax.text(0.05, 0.95, f"Noise: {noise*100:.0f}%", transform=ax.transAxes, ha='left', va='top', 
                    fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            if c == 0: ax.set_ylabel(r"Output $\phi$", fontsize=14)
            else: ax.set_yticklabels([])
            if r == 1: ax.set_xlabel(r"$\|\nabla u\|^2$", fontsize=14)
            else: ax.set_xticklabels([])
            
            if r == 0 and c == 0:
                ax.text(1.05, 1.15, f"{D} Dimensions Projection", transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=20, fontweight='bold')

    out_path = "results/Physics_Response.pdf"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight', format='pdf', transparent=True)
    plt.close()
    print(f"[*] Mega Figure 1 saved to: {out_path}")


def create_mega_manifold_collapse(equations, dimensions, noises):
    print("[*] Generating Mega Figure 2: Manifold Collapse Dashboard...")
    fig = plt.figure(figsize=(24, 30))
    gs_main, top_grids, gs_bottom = setup_mega_layout(fig)
    
    for i, eq_name in enumerate(equations[:4]):
        gs_inner = gridspec.GridSpecFromSubplotSpec(len(noises), len(dimensions), subplot_spec=top_grids[i], hspace=0.15, wspace=0.15)
        draw_panel_label(fig, top_grids[i], chr(ord('a') + i), f"{eq_name.replace('_', ' ')}: Manifold Collapse", 'black')

        for r, noise in enumerate(noises):
            for c, D in enumerate(dimensions):
                ax = fig.add_subplot(gs_inner[r, c])
                data_path = f"results/plot_data/data_{eq_name}_{D}D_noise_{int(noise*100)}.npz"
                if not os.path.exists(data_path):
                    ax.set_xticks([]); ax.set_yticks([]); continue
                
                data = np.load(data_path)
                sc = ax.scatter(data['S'], data['u_pr'], c=data['t'], cmap=UNIFIED_CMAP, s=4, alpha=0.7, edgecolors='none')
                ax.grid(True, linestyle=':', alpha=0.4)
                
                if r == 0: ax.set_title(f"{D}D", fontsize=16, fontweight='bold', pad=15)
                if c == 0: ax.set_ylabel(f"Noise {noise*100:.0f}%", fontsize=14, fontweight='bold', labelpad=8)
                else: ax.set_yticklabels([])
                if r == len(noises) - 1: ax.set_xlabel(r"Spatial Proj. $\bar{X}$", fontsize=14, fontweight='bold')
                else: ax.set_xticklabels([])

    eq_name = equations[4]
    ax_title = fig.add_subplot(gs_main[2])
    ax_title.axis('off')
    ax_title.text(-0.05, 1.15, 'e', transform=ax_title.transAxes, fontsize=34, fontweight='bold', va='bottom', ha='right', color='black')
    ax_title.text(-0.02, 1.15, f"KPZ: Manifold Collapse", transform=ax_title.transAxes, fontsize=32, fontweight='bold', va='bottom', ha='left', color='black')
   
    for col_idx, D in enumerate(dimensions):
        gs_inner_e = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_bottom[col_idx], hspace=0.15, wspace=0.1)
        for n_idx, noise in enumerate(noises):
            r, c = NOISE_MAP_2x2[n_idx]
            ax = fig.add_subplot(gs_inner_e[r, c])
            
            data_path = f"results/plot_data/data_{eq_name}_{D}D_noise_{int(noise*100)}.npz"
            if not os.path.exists(data_path):
                ax.set_xticks([]); ax.set_yticks([]); continue
            
            data = np.load(data_path)
            sc = ax.scatter(data['S'], data['u_pr'], c=data['t'], cmap=UNIFIED_CMAP, s=4, alpha=0.7, edgecolors='none')
            ax.grid(True, linestyle=':', alpha=0.4)
            
            ax.text(0.95, 0.95, f"Noise: {noise*100:.0f}%", transform=ax.transAxes, ha='right', va='top', 
                    fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            if c == 0: ax.set_ylabel("State $U$", fontsize=14)
            else: ax.set_yticklabels([])
            if r == 1: ax.set_xlabel(rf"Spatial Proj. $\bar{{X}}$", fontsize=14)
            else: ax.set_xticklabels([])
            
            if r == 0 and c == 0:
                ax.text(1.05, 1.15, f"{D} Dimensions Projection", transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=20, fontweight='bold')

    cbar_ax = fig.add_axes([0.96, 0.1, 0.012, 0.8])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=UNIFIED_CMAP), cax=cbar_ax)
    cbar.set_label("Time ($t$)", fontsize=20, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=14)

    out_path = "results/Manifold_Collapse.pdf"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight', format='pdf', transparent=True)
    plt.close()
    print(f"[*] Mega Figure 2 saved to: {out_path}")

if __name__ == "__main__":
    equations = ['Fokker_Planck', 'Allen_Cahn', 'Fisher_KPP', 'Zeldovich', 'KPZ']
    dimensions = [20, 50, 100]
    noises = [0.01, 0.05, 0.1, 0.2]
    
    data_dir = "results/plot_data"
    if not os.path.exists(data_dir):
        print(f"[Error] Data directory not found: {data_dir}")
    else:
        create_mega_physics_response(equations, dimensions, noises)
        create_mega_manifold_collapse(equations, dimensions, noises)