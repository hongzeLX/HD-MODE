import matplotlib; matplotlib.use('Agg')
import torch
import numpy as np
import sys
import os
import csv
import psutil

from data import RealWorldPDEDataset
from shotgun import ShotgunInverseModel

DEFAULT_GPU_ID = 0 
if len(sys.argv) > 1: GPU_ID = int(sys.argv[1])
else: GPU_ID = DEFAULT_GPU_ID
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
print(f"[*] Running on device: {device}")

def run_experiment(eq_type, D, noise_level=0.005):
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        _dummy = torch.zeros(1).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        
    print(f"\n" + "="*80)
    print(f"=== [Task] Synergy Discovery: {eq_type} in {D}D | Noise: {noise_level*100:.1f}% ===")
    print("="*80)
    
    if D <= 20: num_train_samples = 4000
    elif D <= 50: num_train_samples = 8000
    else: num_train_samples = 12000
        
    dataset = RealWorldPDEDataset(eq_type=eq_type, D=D)
    
    t_train, x_train, u_train = dataset.generate_random_data(
        num_samples=num_train_samples, is_test=False, noise_level=noise_level
    )
    t_train, x_train, u_train = t_train.to(device), x_train.to(device), u_train.to(device)

    num_test_samples = 2000
    t_test, x_test, u_test_ex = dataset.generate_random_data(
        num_samples=num_test_samples, is_test=True, noise_level=0.0
    )
    t_test, x_test, u_test_ex = t_test.to(device), x_test.to(device), u_test_ex.to(device)


    mu_f = lambda x: torch.zeros_like(x)
    sigma_f = lambda x: torch.ones((x.shape[0], 1), device=device) * 1.0

    model = ShotgunInverseModel(D, mu_f, sigma_f, device=device, dt_local=1e-4, M_local=64)
    batch_size = 1000
    num_samples = len(t_train)

    # STAGE 1

    print(f"\n[{D}D Stage 1] Pre-training U (Data Manifold Burn-in)...")
    model.net_phi.requires_grad_(False)
    optimizer_u = torch.optim.Adam(model.net_u.parameters(), lr=1e-3)
    
    for ep in range(1500):
        indices = torch.randperm(num_samples)
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            idx = indices[i:i+batch_size]
            t_b, x_b, u_b = t_train[idx], x_train[idx], u_train[idx]
            
            optimizer_u.zero_grad()
            loss = model.compute_data_loss(t_b, x_b, u_b)
            loss.backward()
            optimizer_u.step()
            epoch_loss += loss.item() * len(idx) / num_samples
        if ep % 500 == 0:
            print(f"  Epoch {ep:4d} | Data L2 Loss: {epoch_loss:.2e}")

    # STAGE 2

    print(f"\n[{D}D Stage 2] Extracting Phi (Omniscient Neural Approximator)...")
    model.net_u.requires_grad_(False)
    model.net_phi.requires_grad_(True)
    optimizer_phi = torch.optim.Adam(model.net_phi.parameters(), lr=2e-3)
    
    for ep in range(1500):
        indices = torch.randperm(num_samples)
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            idx = indices[i:i+batch_size]
            t_b, x_b = t_train[idx], x_train[idx]
            
            optimizer_phi.zero_grad()
            loss = model.compute_physics_extraction_loss(t_b, x_b)
            loss.backward()
            optimizer_phi.step()
            epoch_loss += loss.item() * len(idx) / num_samples
        if ep % 500 == 0:
            print(f"  Epoch {ep:4d} | Physics Ext. Loss: {epoch_loss:.2e}")

    # STAGE 3

    print(f"\n[{D}D Stage 3] Unrestricted Synergistic Fine-tuning...")
    model.net_u.requires_grad_(True)
    model.net_phi.requires_grad_(True)
    
    optimizer_joint = torch.optim.Adam([
        {'params': model.net_u.parameters(), 'lr': 1e-4},
        {'params': model.net_phi.parameters(), 'lr': 1e-4}
    ])

    for ep in range(4000):
        running_l_data = 0.0
        running_l_phys = 0.0
        indices = torch.randperm(num_samples)
        
        optimizer_joint.zero_grad() 
        for i in range(0, num_samples, batch_size):
            idx = indices[i:i+batch_size]
            t_b, x_b, u_b = t_train[idx], x_train[idx], u_train[idx]
            
            l_data, l_phys = model.compute_synergistic_loss(t_b, x_b, u_b)
            weight = len(idx) / num_samples
            total_loss = (1000.0 * l_data + 1.0 * l_phys) * weight
            total_loss.backward()
            
            running_l_data += l_data.item() * weight
            running_l_phys += l_phys.item() * weight
            
        torch.nn.utils.clip_grad_norm_(model.net_u.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.net_phi.parameters(), 1.0)
        optimizer_joint.step()

        if ep % 500 == 0: 
             print(f"  Ep {ep:4d} | Data L2: {running_l_data:.2e} | Obs Phys: {running_l_phys:.2e}")


    print(f"\n[Evaluation] Quantifying rigorous errors and hardware usage...")
    model.net_u.eval()
    model.net_phi.eval()
    
    with torch.no_grad():
        u_train_pred = model.net_u(t_train, x_train)
        l2_err_train = (torch.norm(u_train_pred - u_train) / torch.norm(u_train)).item()

    t_test.requires_grad_(True)
    x_test.requires_grad_(True)
    
    u_test_pred = model.net_u(t_test, x_test)
    l2_err_test = (torch.norm(u_test_pred - u_test_ex) / torch.norm(u_test_ex)).item()
    
    z_test_pred = torch.autograd.grad(u_test_pred.sum(), x_test, create_graph=False)[0]
    
    u_test_ex_with_grad = dataset.u_exact(t_test, x_test) 
    z_test_ex = torch.autograd.grad(u_test_ex_with_grad.sum(), x_test, create_graph=False)[0]
    
    with torch.no_grad():

        phi_pred = model.net_phi(t_test, x_test, u_test_pred.detach(), z_test_pred.detach())
        
        phi_exact = dataset.phi_exact(t_test, x_test, u_test_ex.detach(), z_test_ex.detach())
        
        l2_err_phi = (torch.norm(phi_pred - phi_exact) / torch.norm(phi_exact)).item()
    

    if device.type == 'cuda':
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_gpu_mem_mb = 0.0
        
    process = psutil.Process(os.getpid())
    cpu_mem_mb = process.memory_info().rss / (1024 * 1024)
        
    print(f"\n[{D}D Final Results] {eq_type} Equation")
    print(f"--> Train Err (Noisy Data)         : {l2_err_train:.4e}")
    print(f"--> State (U) Gen Err (OOD Sensors): {l2_err_test:.4e}")
    print(rf"--> Physics Generator (\phi) Err   : {l2_err_phi:.4e}")
    print(f"--> Peak GPU Memory Usage          : {peak_gpu_mem_mb:.2f} MB")
    print(f"--> Current CPU RAM Usage          : {cpu_mem_mb:.2f} MB")

    S_test = torch.sum(x_test.detach(), dim=-1).cpu().numpy().flatten() / np.sqrt(D)
    
    if eq_type == 'KPZ':
        x_axis_val = torch.sum(z_test_pred.detach()**2, dim=-1).cpu().numpy().flatten()
    else:
        x_axis_val = u_test_ex.detach().cpu().numpy().flatten()

    idx = np.random.choice(len(u_test_ex), min(1000, len(u_test_ex)), replace=False)

    plot_data_dict = {
        'u_ex': u_test_ex.detach().cpu().numpy().flatten(),
        'u_pr': u_test_pred.detach().cpu().numpy().flatten(),
        'phi_tr': phi_exact.cpu().numpy().flatten()[idx],
        'phi_pr': phi_pred.cpu().numpy().flatten()[idx],
        'x_axis': x_axis_val[idx],
        'S': S_test,
        't': t_test.detach().cpu().numpy().flatten()
    }
    
    save_dir = "results/plot_data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"data_{eq_type}_{D}D_noise_{int(noise_level*100)}.npz")
    np.savez_compressed(save_path, **plot_data_dict)
    print(f"[*] Plot data successfully serialized and saved to {save_path}")
    
    return l2_err_train, l2_err_test, l2_err_phi, peak_gpu_mem_mb, cpu_mem_mb


if __name__ == "__main__":
    pde_types = ['Fokker_Planck', 'KPZ', 'Allen_Cahn', 'Fisher_KPP', 'Zeldovich']
    dimensions = [20, 50, 100]
    noises = [0.01, 0.05, 0.1, 0.2] 

    os.makedirs("results", exist_ok=True)
    csv_filename = "results/experiment_metrics_log.csv"
    
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Equation", "Dimension", "Noise_Level", 
                "Train_Error_U", "Test_Error_U_OOD", "Generator_Error_Phi",
                "Peak_GPU_Mem(MB)", "CPU_RAM(MB)"         
            ])

    for pde in pde_types:
        print(f"\n{'#'*80}\n### [Mega-Experiment] Starting Pipeline for Equation: {pde} ###\n{'#'*80}")
        
        for D in dimensions:
            for noise in noises:
                
                data_path = f"results/plot_data/data_{pde}_{D}D_noise_{int(noise*100)}.npz"
                if os.path.exists(data_path):
                    print(f"[Skip] Found existing data for {pde} {D}D Noise {noise*100}%. Skipping training.")
                    continue

                train_err, test_err, phi_err, gpu_mem, cpu_mem = run_experiment(pde, D=D, noise_level=noise)
                
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        pde, D, f"{noise * 100:.1f}%", 
                        f"{train_err:.6e}", f"{test_err:.6e}", f"{phi_err:.6e}",
                        f"{gpu_mem:.2f}", f"{cpu_mem:.2f}"
                    ])

