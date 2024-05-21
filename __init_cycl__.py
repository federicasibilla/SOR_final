"""
__init_cycl__.py: 

                
"""
import numpy as np

import cProfile
import pstats

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

from R_dynamics import *
from N_dynamics import *
from visualize  import *
from update import *
from SOR_vanilla import *

# initialize R0
n_r = 4
n_s = 3
n   = 40

R0  = np.zeros((n, n, n_r))
# saturate resource c everywhere
R0[:,:,0]=10
R0[:,:,1]= R0[:,:,2]=R0[:,:,3]=0
g   = np.array([0.5,0.5,0.5]) 
m   = np.array([0.,0.,0.])

initial_guess = np.zeros((n, n, n_r))*10

# initialize species grid: random
N0 = np.zeros((n,n,n_s))
for i in range(n):
    for j in range(n):
        idx = np.random.randint(3)
        N0[i,j,idx]=1


# define parameters
param = {
    # model parameters
    'R0' : R0.copy(),                                  # initial conc. nxnxn_r [monod constants]
    'w'  : np.ones((n_r))*20,                          # energy conversion     [energy/mass]
    'l'  : np.array([1,0.8,0.7,0.6]),                  # leakage               [adim]
    'tau': np.array([np.inf,np.inf,np.inf,np.inf]),    # reinsertion rate inv. [time] 
    'g'  : g,                                          # growth conv. factors  [1/energy]
    'm'  : m,                                          # maintainance requ.    [energy/time]
    'ext': np.array([10.,0.,0.,0.]),
    
    # sor algorithm parameters
    'n'  : n,                                          # grid points in each dim
    'sor': 1.85,                                       # relaxation parameter
    'L'  : 40,                                         # grid true size        [length]
    'D'  : 1e2,                                        # diffusion constant    [area/time] 
    'acc': 1e-3,                                       # maximum accepted stopping criterion   
    'ref': 0                                           # number of grid refinements to perform 
}

# make matrices
up_mat   = np.array([[1,0.,0.,1],[0.,1,0.,0.],[0.,0.,1,0.]])
met_mat  = np.array([[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]])
sign_mat = np.array([[1.,1.,1,1],[1.,1.,1,1],[1,1,1,1]])
mat_ess  = np.array([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
spec_met = np.array([[0.,1.,0.,0.],[0.,0.,1,0.],[0.,0.,0.,1]])
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

vispreferences(mat)
makenet(met_mat)


frames, current_R, current_N, g_rates = simulate(5000, f, initial_guess, N0, param, mat)

R_ongrid(current_R)
G_ongrid(g_rates,encode(frames[-2], np.array([0,1,2])))
N_ongrid(current_N)
R_ongrid_3D(current_R)

abundances(frames)

