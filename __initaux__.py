"""
__initaux__.py: 

                
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
n_r = 2
n_s = 2
n   = 40

R0  = np.zeros((n, n, n_r))
# saturate resource c everywhere
R0[:,:,0]=10
R0[:,:,1]=0
g   = np.array([0.5,0.5]) 
m   = np.array([0.,0.])

initial_guess = np.random.uniform(9, 10, size=(n,n,n_r))

# initialize species grid: random
#N = np.random.randint(2, size=(n, n,n_s))
#N[N[:,:,0]==1,1]=0
#N[N[:,:,0]==0,1]=1
N = np.zeros((n,n,n_s))
N[:,20:,0]=1
N[:,:20,1]=1


# define parameters
param = {
    # model parameters
    'R0' : R0.copy(),                                  # initial conc. nxnxn_r [monod constants]
    'w'  : np.ones((n_r))*20,                          # energy conversion     [energy/mass]
    'l'  : np.ones((n_r))-0.4,                         # leakage               [adim]
    'tau': np.array([1,np.inf]),                       # reinsertion rate inv. [time] 
    'g'  : g,                                          # growth conv. factors  [1/energy]
    'm'  : m,                                          # maintainance requ.    [energy/time]
    'ext': np.array([10.,0.]),
    
    # sor algorithm parameters
    'n'  : n,                                          # grid points in each dim
    'sor': 1.85,                                       # relaxation parameter
    'L'  : 40,                                         # grid true size        [length]
    'D'  : 1e2,                                        # diffusion constant    [area/time] 
    'acc': 1e-3,                                      # maximum accepted stopping criterion   
    'ref': 0                                           # number of grid refinements to perform 
}

# make matrices
up_mat   = np.array([[1,0.],[1.,1.]])
met_mat  = np.array([[0.,0.],[1.,0.]])
sign_mat = np.array([[1.,1.],[1.,1.]])
mat_ess  = np.array([[0.,0.],[0.,1]])
spec_met = np.array([[0.,1.],[0.,0.]])


mat = {
    'uptake'  : up_mat,
    'met'     : met_mat,
    'sign'    : sign_mat,
    'ess'     : mat_ess,
    'spec_met': spec_met
}


frames, current_R, current_N, g_rates = simulate(5000, f, initial_guess, N, param, mat)

R_ongrid(current_R)
G_ongrid(g_rates,encode(frames[-2], np.array([0,1])))
N_ongrid(current_N)
R_ongrid_3D(current_R)

abundances(frames)

