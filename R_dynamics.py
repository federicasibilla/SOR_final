"""
R_dynamics.py: python file to contain the equation for the resources dynamics

"""

import numpy as np

#----------------------------------------------------------------------------------------------
# f vectorial function to calculate the sources, given the nxnxn_r concentrations matrix

def f(R,N,param,mat):

    species  = np.argmax(N,axis=2)

    # identify auxotrophies on the grid
    mask = np.zeros((R.shape[0],R.shape[1],R.shape[2]))
    mask[mat['ess'][species]!=0] = 1

    # calculate MM at each site and mask for essential resources
    upp      = R/(R+1)
    up_ess  = np.where(mask==0, 1, upp)

    # find limiting nutrient and calculate correspoinding mu modulation
    lim = np.argmin(up_ess, axis=2)
    mu_lim  = np.min(up_ess,axis=2)

    # create modulation mask
    mu  = np.zeros((R.shape[0],R.shape[1],R.shape[2]))
    mu += mu_lim[:, :, np.newaxis]
    mu[np.arange(R.shape[0])[:, None], np.arange(R.shape[1]),lim] = 1

    # modulate uptake and insert uptake rates (this is also depletion)
    up = upp*mu*mat['uptake'][species]

    # calculate production
    inn = np.dot(up*param['l']*param['w'],mat['met'].T)*mat['spec_met'][species]/param['w']

    # external
    ext = 1/param['tau']*(param['R0']-R)
    
    return inn-up, up