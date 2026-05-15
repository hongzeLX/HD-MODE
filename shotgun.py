import torch
import torch.nn as nn
from einops import repeat
import numpy as np

class Net_U(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D + 1, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish(),
            nn.Linear(256, 1)
        )
    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=-1))

class GeneratorNN(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.D = D
        self.net = nn.Sequential(
            nn.Linear(2 * D + 2, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish(),
            nn.Linear(256, 1)
        )
    def forward(self, t, x, u, z):
        return self.net(torch.cat([t, x, u, z], dim=-1))

class ShotgunInverseModel:
    def __init__(self, D, mu_f, sigma_f, device, dt_local=1e-4, M_local=64):
        self.D, self.mu_f, self.sigma_f = D, mu_f, sigma_f
        self.device = device
        self.dt1, self.M1 = dt_local, M_local
        self.net_u = Net_U(D).to(device)
        self.net_phi = GeneratorNN(D).to(device)

    def _get_shotgun_operator(self, t, x, u_val):

        Batch = x.shape[0]
        mu, sigma = self.mu_f(x), self.sigma_f(x)
        dWt = torch.randn((Batch, self.M1, self.D), device=self.device) * np.sqrt(self.dt1)
        
        x_mean = repeat(x + mu * self.dt1, 'b d -> b m d', m=self.M1)
        
        t_prev = repeat(t - self.dt1, 'b 1 -> b m 1', m=self.M1)
        
        if sigma.shape[-1] == self.D:
            diffusion = sigma.reshape(Batch, 1, self.D) * dWt
        else:
            diffusion = sigma.reshape(Batch, 1, 1) * dWt
            
        x_p, x_m = x_mean + diffusion, x_mean - diffusion
        
        u_p = self.net_u(t_prev.reshape(-1, 1), x_p.reshape(-1, self.D)).reshape(Batch, self.M1, 1)
        u_m = self.net_u(t_prev.reshape(-1, 1), x_m.reshape(-1, self.D)).reshape(Batch, self.M1, 1)
        
        u_sym_mean = 0.5 * u_p.mean(dim=1) + 0.5 * u_m.mean(dim=1)
        
        return (u_val - u_sym_mean) / self.dt1

    def compute_data_loss(self, t, x, u):
        """ Stage 1 Loss """
        return torch.mean((self.net_u(t, x) - u)**2)

    def compute_physics_extraction_loss(self, t, x):
        """ Stage 2 Loss """
        x = x.clone().detach().requires_grad_(True)
        u_val = self.net_u(t, x)
        z_val = torch.autograd.grad(u_val.sum(), x, create_graph=True)[0]
        z_val_safe = torch.clamp(z_val, min=-5.0, max=5.0)
        with torch.no_grad():
            op_target = self._get_shotgun_operator(t, x, u_val)
            
        phi_pred = self.net_phi(t, x, u_val.detach(), z_val_safe.detach())
        
        return torch.mean((op_target - phi_pred)**2)

    def compute_synergistic_loss(self, t, x, u_data):
        """ Stage 3 Loss """
        x = x.clone().detach().requires_grad_(True)
        u_val = self.net_u(t, x)
        z_val = torch.autograd.grad(u_val.sum(), x, create_graph=True)[0]

        z_val_safe = torch.clamp(z_val, min=-5.0, max=5.0) 
        loss_data = torch.mean((u_val - u_data)**2)
        
        op_val = self._get_shotgun_operator(t, x, u_val)
        phi_pred = self.net_phi(t, x, u_val, z_val_safe)
        
        loss_phys = torch.mean((op_val - phi_pred)**2)
        
        return loss_data, loss_phys