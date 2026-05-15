import torch
import numpy as np

class RealWorldPDEDataset:
    def __init__(self, eq_type='Allen_Cahn', D=20, num_samples=5000):
        self.eq_type = eq_type
        self.D = D
        self.num_samples = num_samples
        self.t_min, self.t_max = 0.0, 1.0
        

        # ============== 1. Allen-Cahn ==============
        if eq_type == 'Allen_Cahn':

            self.x_min, self.x_max = -3.0, 3.0

            self.u_exact = lambda t, x: torch.tanh(torch.sum(x, dim=-1, keepdim=True) / np.sqrt(self.D))
            self.phi_exact = lambda t, x, u, z: u - u**3

        # ============== 2. Zeldovich ==============
        elif eq_type == 'Zeldovich':

            mu_x = 0.25 / np.sqrt(self.D)
            radius = 4.5
            self.x_min, self.x_max = mu_x - radius, mu_x + radius
            
            self.u_exact = lambda t, x: 1.0 / (1.0 + torch.exp(torch.sum(x, dim=-1, keepdim=True) / np.sqrt(self.D) - 0.5 * t))
            self.phi_exact = lambda t, x, u, z: u**2 - u**3

        # ============== 3. Fisher_KPP ==============
        elif eq_type == 'Fisher_KPP':

            mu_x = (5.0 / 12.0) * np.sqrt(3.0 / self.D)
            radius = 3.5
            self.x_min, self.x_max = mu_x - radius, mu_x + radius
            
            self.u_exact = lambda t, x: 1.0 / (1.0 + torch.exp(torch.sum(x, dim=-1, keepdim=True) / np.sqrt(3.0 * self.D) - (5.0 / 6.0) * t))**2
            self.phi_exact = lambda t, x, u, z: u - u**2

        # ============== 4. KPZ ==============
        elif eq_type == 'KPZ':

            mu_x = -0.25 / np.sqrt(self.D)
            radius = 2.5
            self.x_min, self.x_max = mu_x - radius, mu_x + radius
            
            self.u_exact = lambda t, x: torch.log(1.0 + torch.exp(torch.sum(x, dim=-1, keepdim=True) / np.sqrt(self.D) + 0.5 * t))

            self.phi_exact = lambda t, x, u, z: 0.5 * torch.sum(z**2, dim=-1, keepdim=True)
            
        # ============== 5. Fokker-Planck ==============
        elif eq_type == 'Fokker_Planck':

            mu_x = 0.5 / np.sqrt(self.D)
            radius = 2.5 
            self.x_min, self.x_max = mu_x - radius, mu_x + radius

            S_func = lambda x: torch.sum(x, dim=-1, keepdim=True) / np.sqrt(self.D)
            
            self.u_exact = lambda t, x: torch.exp( -(S_func(x) - t)**2 )
            
            self.phi_exact = lambda t, x, u, z: u + 2.0 * (S_func(x) - t) * u - 2.0 * (S_func(x) - t)**2 * u

        else:
            raise ValueError(f"Unknown PDE type: {eq_type}")
        
    def generate_random_data(self, num_samples=4000, is_test=False, noise_level=0.0):

        seed = 42 if not is_test else 999
        torch.manual_seed(seed)

        t_flat = torch.rand(num_samples, 1) * (self.t_max - self.t_min) + self.t_min
        x_flat = torch.rand(num_samples, self.D) * (self.x_max - self.x_min) + self.x_min

        u_flat = self.u_exact(t_flat, x_flat)

        if not is_test and noise_level > 0.0:
            noise = torch.randn_like(u_flat) * noise_level * torch.std(u_flat)
            u_flat += noise

        return t_flat, x_flat, u_flat