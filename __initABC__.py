"""
__initABC__.py: 

- takes 4 minutes for 1000 steps
- takes 35 minutes fro 10000 steps

"""

import numpy as np

from R_dynamics import *
from N_dynamics import *
from visualize  import *
from update import *

# initialize R0
n_r = 3
n_s = 3
n   = 40 

g   = np.array([0.5,0.5,0.5]) 
m   = np.array([0.,0.0,0.0])
w   = np.array([0.1,0.1,1])

# initialize species grid: random
N0 = np.zeros((n,n,n_s))
for i in range(n):
    for j in range(n):
        idx = np.random.randint(3)
        N0[i,j,idx]=1

# find well mixed equilibrium
ext = np.array([10.,0.1,0.1])
N_wm = np.sum(N0, axis=2)


# initialize R0 based on eq. R
R0  = np.zeros((n, n, n_r))
R0[:,:,0]=10
R0[:,:,1]=0
R0[:,:,2]=0

# define parameters
param = {
    # model parameters
    'R0' : R0.copy(),                                  # initial conc. nxnxn_r [monod constants]
    'N0' : N0.copy(),                                  # initial species grid nxnxn_s
    'w'  : w,                                          # energy conversion     [energy/mass]
    'l'  : np.array([0.6,0.6,0.9]),                    # leakage               [adim]
    'tau': np.array([1,np.inf,np.inf]),                # reinsertion rate inv. [time] 
    'g'  : g,                                          # growth conv. factors  [1/energy]
    'm'  : m,                                          # maintainance requ.    [energy/time]
    'ext': ext,                                        # esternal R condition  [Monod constants]
    
    # discretization parameters
    'n'  : n,                                          # grid points in each dim
    'L'  : 40,                                         # grid true size        [length]
    'D'  : 1e3,                                        # diffusion constant    [area/time] 
    'acc': 1e-8                                        # accepted maximum relative error
}

# make matrices
up_mat   = np.array([[1,0.,1],[1,0.,1],[0.,1,0.]])
met_mat  = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
sign_mat = np.array([[1.,1.,1],[1.,1.,1],[1,1,1]])
mat_ess  = np.array([[0.,0.,1],[0.,0.,1],[0.,1,0.]])
spec_met = np.array([[0.,0.,0.],[0.,1,0.],[0.,0.,1.]])
print(up_mat)
print(met_mat)
print(sign_mat)
print(mat_ess)
print(spec_met)

mat = {
    'uptake'  : up_mat,
    'met'     : met_mat,
    'sign'    : sign_mat,
    'ess'     : mat_ess,
    'spec_met': spec_met
}

# visualize matrices
vispreferences(mat)
makenet(met_mat)

#-------------------------------------------------------------------------------------------------------------
# SIMULATION

guess = np.random.uniform(9, 10, size=(n,n,n_r))

# run 1000 steps 
steps, R_fin, N_fin, rates = simulate(10, f, guess, N0, param, mat)

# plot final R and N grids
R_ongrid(R_fin)
N_ongrid(N_fin)
R_ongrid_3D((R_fin))
G_ongrid(rates[-1],N_fin)

#np.savez('matrices_1.npz', *steps)
#abundances(steps)

# produce animation
#animation_grid(steps)