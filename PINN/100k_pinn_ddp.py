import h5py
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os
import gc
file_path = "./train_data.h5"
with h5py.File(file_path, "r") as f:
    for i in list(f.keys()):
        globals()[i] = f[i][:]
        if globals()[i].ndim == 1:
            globals()[i] = globals()[i].reshape(-1,1)
l = [0,0.01,0,0.01,0,0.01,0.02,0.02,0,0.01,0.02,0.03,0.03,0,0.01,0.02,0.03,0.04,0.04,0,0.01,0.02,0.03,0.04,0.05,0.05,0.02,0.03,0.04,0.05,0.06,0.06,0.03,0.04,0.05,0.06,0.07,0.07,0.04,0.05,0.06,0.07,0.08,0.08,0.05,0.06,0.07,0.08,0.09,0.09,0.06,0.07,0.08,0.09,0.1,0.1,0.07,0.08,0.09,0.1,0.11,0.11,0.08,0.09,0.1,0.11,0.12,0.12,0.09,0.1,0.11,0.12,0.13,0.13,0.1,0.11,0.12,0.13,0.14,0.14,0.11,0.12,0.13,0.14,0.15,0.15,0.12,0.13,0.14,0.15,0.16,0.16,0.13,0.14,0.15,0.16,0.17,0.17,0.14,0.15,0.16,0.17,0.18,0.18,0.15,0.16,0.17,0.18,0.19,0.19,0.16,0.17,0.18,0.19,0.2,0.2,0.17,0.18,0.19,0.2,0.21,0.21,0.18,0.19,0.2,0.21,0.22,0.22,0.19,0.2,0.21,0.22,0.23,0.23,0.2,0.21,0.22,0.23,0.24,0.24,0.21,0.22,0.23,0.24,0.25,0.25,0.22,0.23,0.24,0.25,0.26,0.26,0.23,0.24,0.25,0.26,0.27,0.27,0.24,0.25,0.26,0.27,0.28,0.28,0.25,0.26,0.27,0.28,0.29,0.29,0.26,0.27,0.28,0.29,0.3,0.3,0.27,0.28,0.29,0.3,0.31,0.31,0.28,0.29,0.3,0.31,0.32,0.32,0.29,0.3,0.31,0.32,0.33,0.33,0.3,0.31,0.32,0.33,0.34,0.34,0.31,0.32,0.33,0.34,0.35,0.35,0.32,0.33,0.34,0.35,0.36,0.36,0.33,0.34,0.35,0.36,0.37,0.37,0.34,0.35,0.36,0.37,0.38,0.38,0.35,0.36,0.37,0.38,0.39,0.39,0.36,0.37,0.38,0.39,0.4,0.4,0.37,0.38,0.39,0.4,0.41,0.41,0.38,0.39,0.4,0.41,0.42,0.42,0.39,0.4,0.41,0.42,0.43,0.43,0.4,0.41,0.42,0.43,0.44,0.44,0.41,0.42,0.43,0.44,0.45,0.45,0.42,0.43,0.44,0.45,0.46,0.46,0.43,0.44,0.45,0.46,0.47,0.47,0.44,0.45,0.46,0.47,0.48,0.48,0.45,0.46,0.47,0.48,0.49,0.49,0.46,0.47,0.48,0.49,0.5,0.5,0.47,0.48,0.49,0.5,0.51,0.51,0.48,0.49,0.5,0.51,0.52,0.52,0.49,0.5,0.51,0.52,0.53,0.53,0.5,0.51,0.52,0.53,0.54,0.54,0.51,0.52,0.53,0.54,0.55,0.55,0.52,0.53,0.54,0.55,0.56,0.56,0.53,0.54,0.55,0.56,0.57,0.57,0.54,0.55,0.56,0.57,0.58,0.58,0.55,0.56,0.57,0.58,0.59,0.59,0.56,0.57,0.58,0.59,0.6,0.6,0.57,0.58,0.59,0.6,0.61,0.61,0.58,0.59,0.6,0.61,0.62,0.62,0.59,0.6,0.61,0.62,0.63,0.63,0.6,0.61,0.62,0.63,0.64,0.64,0.61,0.62,0.63,0.64,0.65,0.65,0.62,0.63,0.64,0.65,0.66,0.66,0.63,0.64,0.65,0.66,0.67,0.67,0.64,0.65,0.66,0.67,0.68,0.68,0.65,0.66,0.67,0.68,0.69,0.69,0.66,0.67,0.68,0.69,0.7,0.7,0.67,0.68,0.69,0.7,0.71,0.71,0.68,0.69,0.7,0.71,0.72,0.72,0.69,0.7,0.71,0.72,0.73,0.73,0.7,0.71,0.72,0.73,0.74,0.74,0.71,0.72,0.73,0.74,0.75,0.75,0.72,0.73,0.74,0.75,0.76,0.76,0.73,0.74,0.75,0.76,0.77,0.77,0.74,0.75,0.76,0.77,0.78,0.78,0.75,0.76,0.77,0.78,0.79,0.79,0.76,0.77,0.78,0.79,0.8,0.8,0.77,0.78,0.79,0.8,0.81,0.81,0.78,0.79,0.8,0.81,0.82,0.82,0.79,0.8,0.81,0.82,0.83,0.83,0.8,0.81,0.82,0.83,0.84,0.84,0.81,0.82,0.83,0.84,0.85,0.85,0.82,0.83,0.84,0.85,0.86,0.86,0.83,0.84,0.85,0.86,0.87,0.87,0.84,0.85,0.86,0.87,0.88,0.88,0.85,0.86,0.87,0.88,0.89,0.89,0.86,0.87,0.88,0.89,0.9,0.9,0.87,0.88,0.89,0.9,0.91,0.91,0.88,0.89,0.9,0.91,0.92,0.92,0.89,0.9,0.91,0.92,0.93,0.93,0.9,0.91,0.92,0.93,0.94,0.94,0.91,0.92,0.93,0.94,0.95,0.95,0.92,0.93,0.94,0.95,0.96,0.96,0.93,0.94,0.95,0.96,0.97,0.97,0.94,0.95,0.96,0.97,0.98,0.98,0.95,0.96,0.97,0.98,0.99,0.99,0.96,0.97,0.98,0.99,1,1,0.97,0.98,0.99,1,0.98,0.99,1,0.99,1,1]
r = [0.7037,0.7037,0.76296,0.76296,0.82222,0.82222,0.7037,0.76296,0.88148,0.88148,0.82222,0.7037,0.76296,0.94074,0.94074,0.88148,0.82222,0.7037,0.76296,1,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,0.7037,0.76296,1,0.94074,0.88148,0.82222,1,0.94074,0.88148,1,0.94074,1]
l = np.array(l)
r = np.array(r)
unique_l = np.unique(l)
unique_r = np.unique(r)

grid = np.zeros((101, 6), dtype=int)

for i in range(606):
    l_index = np.where(unique_l == l[i])[0][0]
    r_index = np.where(unique_r == r[i])[0][0]
    grid[l_index, r_index] = i

new_par = np.zeros((par.shape[0], par.shape[1], 101, 6))
new_sol = np.zeros((sol.shape[0], sol.shape[1], 101, 6))
for i in range(101):
    for j in range(6):
        new_par[:,:,i,j] = par[:,:,grid[i,j]]
        new_sol[:,:,i,j] = sol[:,:,grid[i,j]]
par, sol = new_par, new_sol
del new_par, new_sol
gc.collect()
input = np.concatenate((ka, kd, qmax, sa, sd, sq, col_disp, fil_diff, par_diff, bul0),axis=1)
# min_vals = input.min(axis=0) 
# max_vals = input.max(axis=0)
# np.save("min_vals.npy", min_vals)
# np.save("max_vals.npy", max_vals)
# new_input_scaled = (input - min_vals) / (max_vals - min_vals)
# input = new_input_scaled
t_s, v_s = round(input.shape[0]*0.8), round(input.shape[0]*0.9)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Notice: t, l, r in dimensionless form t:[0,] l:[0,1] r:[0.7037,1]
from torch.autograd import grad

def partial_derivative(output, inputs, dim):
    grads = grad(output.sum(), inputs, create_graph=True)[0]
    return grads[:, dim]


def second_partial_derivative(output, inputs, dim):
    first = partial_derivative(output, inputs, dim)
    second = grad(first.sum(), inputs, create_graph=True)[0]
    return second[:, dim]

class PINN(nn.Module):
    def __init__(self,H):
        super(PINN, self).__init__()
        # input features t, l, r, adsorption, desorption, capacity, sa, sd, sq, Dax, kf, Dp, not input for model but use to compute loss (c0, phi，c0_current)
        self.IL = nn.Linear(12,H)
        self.linear1 = nn.Linear(H,H)
        self.linear2 = nn.Linear(H,H//2)
        # output features cl, cp, cs
        self.OL = nn.Linear(H//2,3)
        # self.dropout = nn.Dropout(p=0.1)
        # self.activation = nn.ReLU()
        
    def forward(self,x):
        x = self.IL(x)
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        # x = self.dropout(x)  
        x = self.OL(x)
        return x
    
class PINNLoss(nn.Module):
    def __init__(self, device, weight_factors = [1.0, 1e-10, 1e-5, 1e-5]):
        super(PINNLoss, self).__init__()
        self.device = device
        self.weight_factors = weight_factors
        self.mse_loss = nn.MSELoss()  # Using built-in MSE as part of custom loss

    def forward(self, predictions, targets, inputs):
        # extract parameters
        r = inputs[:,2]
        ads = inputs[:,3]
        dep = inputs[:,4]
        cap = inputs[:,5]
        sa = inputs[:,6]
        sd = inputs[:,7]
        sq = inputs[:,8]
        Dax = inputs[:,9]
        kf = inputs[:,10]
        Dp = inputs[:,11]
        c0 = inputs[:,12]
        cprp = predictions[:,1]/c0  #dimensionless
        phi = inputs[:,13]
        c0_current = inputs[:,14]
        
        # column parameters
        l = 0.03  # m
        ec = 0.5
        ep = 0.4
        vl = 0.0028294  # m/s
        Rout = 1.35e-6 # m
        l = torch.tensor(l, dtype=torch.float32).to(self.device)
        ec = torch.tensor(ec, dtype=torch.float32).to(self.device)
        ep = torch.tensor(ep, dtype=torch.float32).to(self.device)
        vl = torch.tensor(vl, dtype=torch.float32).to(self.device)
        Rout = torch.tensor(Rout, dtype=torch.float32).to(self.device)
        
        # calculate dimensionless numbers 
        cl = predictions[:,0]/c0
        cp = predictions[:,1]/c0
        cs = predictions[:,2]/c0
        # mse loss
        data_loss = self.mse_loss(predictions/c0.view(-1,1), targets) 
        Pe = vl*l/Dax
        St = 3*l*kf/(vl*Rout)
        Bi = Rout*kf/(ep*Dp)
        ka_eff = ads * torch.exp(-sa*phi)
        kd_eff = dep * torch.exp(sd*phi)
        q_eff = cap * torch.exp(-sq*phi)
        
        #  compute derivation
        # cl
        cl_t = partial_derivative(cl, inputs, 0)
        cl_z = partial_derivative(cl, inputs, 1)
        cl_z_2 = second_partial_derivative(cl, inputs, 1)
        # cp
        cp_t = partial_derivative(cp, inputs, 0)
        cp_r = partial_derivative(cp, inputs, 2)
        cp_r_2 = second_partial_derivative(cp, inputs, 2)
        # cs
        cs_t = partial_derivative(cs, inputs, 0)
        
        # three governing equations loss
        eqn1 = -1/Pe * cl_z_2 + cl_z + cl_t + (1-ec)/ec * St * (cl-cprp)
        eqn2 = ep*cp_t + (1-ep)*cs_t - St/3/Bi * (2/r * cp_r + cp_r_2)
        eqn3 = vl/l * cs_t - ka_eff * cp * (q_eff - cs * c0) + kd_eff * cs
        
        # Boundary condition
        BC_1_1 = 1/Pe * cl_z - c0_current + cl
        BC_1_2 = cl_z
        BC_2_1 = cp_r
        BC_2_2 = cp_r - Bi * (cl-cp)
        
        # Initial condition
        IC_1 = cl
        IC_2 = cp
        IC_3 = cs
        
        # compute loss
        phy_loss = torch.zeros(predictions.shape[0]).to(self.device)
        all_zone = torch.ones(predictions.shape[0]).to(self.device)
        a = torch.where(inputs[:,0] == 0, 1.0, 0.0) # initial condition
        b = torch.where(inputs[:,1] == 0, 1.0, 0.0) # BC_1_1
        c = torch.where(inputs[:,1] == 1, 1.0, 0.0) # BC_1_2
        d = torch.where(inputs[:,2] == 0.7037, 1.0, 0.0) # BC_2_1
        e = torch.where(inputs[:,2] == 1, 1.0, 0.0) # BC_2_2
        # print(eqn1, eqn2, eqn3, IC_1, IC_2, IC_3, BC_1_1, BC_1_2, BC_2_1, BC_2_2)
        phy_loss = self.weight_factors[1]*all_zone * (eqn1**2 + eqn2**2 + eqn3**2) +  self.weight_factors[2]*a * (IC_1**2 + IC_2**2 + IC_3**2) +  self.weight_factors[3]*b * (BC_1_1**2) +  self.weight_factors[3]*c * (BC_1_2**2) +  self.weight_factors[3]*d * (BC_2_1**2) +  self.weight_factors[3]*e * (BC_2_2**2)
        
        return self.weight_factors[0] * data_loss + phy_loss.mean()
    # + self.weight_factors[1] * phy_loss.mean()

from tqdm import tqdm


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


is_distributed, rank, world_size, local_rank = setup_distributed()
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

model = PINN(6000).to(device)
if is_distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

criterion = PINNLoss(device=device, weight_factors=[1.0, 1e-10, 1e-5, 1e-5])
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.98)

c_best = 1
train_limit = min(10000, input.shape[0])
rr_list = [0.7037,0.76296,0.82222,0.88148,0.94074,1]

for epoch in tqdm(range(50), disable=(rank != 0)):
    model.train()
    log_interval = 100
    global_step = 0
    for leng in tqdm(range(rank, train_limit, world_size), disable=(rank != 0)):
        train_data_in = []
        train_data_out = []
        end_con_value = float(end_con[leng, 0])
        end_time_value = float(end_time[leng, 0])
        slope = (end_con_value - 0.05) / end_time_value
        for tt in range(bul.shape[1]):
            for ll in range(bul.shape[2]):
                for rr in range(par.shape[3]):
                    train_data = []
                    # dimensionless time, length, radius
                    train_data.extend([tt / 10.6, ll / 100, rr_list[rr]])
                    train_data.extend(input[leng].tolist())
                    phi = slope * tt + 0.05
                    if phi > end_con_value:
                        phi = end_con_value
                    if tt > 0:
                        co_current = 0
                    else:
                        co_current = 1
                    train_data.extend([phi, co_current])
                    train_data_in.append(train_data)
                    train_data_out.append([bul[leng, tt, ll], par[leng, tt, ll, rr], sol[leng, tt, ll, rr]])
        xo = torch.tensor(np.array(train_data_in), dtype=torch.float32).to(device)
        y = torch.tensor(train_data_out, dtype=torch.float32).to(device)
        dataset = TensorDataset(xo, y)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        for xo_batch, y_batch in dataloader:
            xo_batch = xo_batch.detach().clone().requires_grad_(True)
            x_batch = xo_batch[:, :-3]
            optimizer.zero_grad()
            p = model(x_batch)
            loss = criterion(p, y_batch, xo_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            if rank == 0 and global_step % log_interval == 0:
                print(
                    f"Epoch {epoch + 1} Step {global_step} "
                    f"training_loss={loss.item() :.6f}"
                )
                

    model.eval()
    local_mse_sum = 0.0
    local_mse_count = 0
    with torch.no_grad():
        for leng in range(rank, 3, world_size):
            valid_data_in = []
            valid_data_out = []
            idx = leng
            end_con_value = float(end_con[idx, 0])
            end_time_value = float(end_time[idx, 0])
            slope = (end_con_value - 0.05) / end_time_value
            for tt in range(bul.shape[1]):
                for ll in range(bul.shape[2]):
                    for rr in range(par.shape[3]):
                        valid_data = []
                        valid_data.extend([tt / 10.6, ll / 100, rr_list[rr]])
                        valid_data.extend(input[idx].tolist())
                        phi = slope * tt + 0.05
                        if phi > end_con_value:
                            phi = end_con_value
                        if tt > 0:
                            co_current = 0
                        else:
                            co_current = 1
                        valid_data.extend([phi, co_current])
                        valid_data_in.append(valid_data)
                        valid_data_out.append([bul[idx, tt, ll], par[idx, tt, ll, rr], sol[idx, tt, ll, rr]])
            xo = torch.tensor(np.array(valid_data_in), dtype=torch.float32).to(device)
            pv = model(xo[:, :-3])
            target = torch.tensor(valid_data_out, dtype=torch.float32).to(device)[:, 0]
            local_mse_count += 1

    if is_distributed:
        val_stats = torch.tensor([local_mse_sum, local_mse_count], dtype=torch.float32, device=device)
        dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
        mse_bul = (val_stats[0] / val_stats[1]).item() if val_stats[1] > 0 else float("inf")
    else:
        mse_bul = local_mse_sum / local_mse_count if local_mse_count > 0 else float("inf")

    if rank == 0:
        if mse_bul < c_best:
            model_to_save = model.module if is_distributed else model
            torch.save(model_to_save.state_dict(), "100k_pinn_test.pt")
            c_best = mse_bul
            print(f"Best val_mse updated: {c_best:.6f}")

cleanup_distributed(is_distributed)
