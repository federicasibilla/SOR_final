"""
SOR_vanilla.py: python file containing the SOR implementation with efficient red-black ordering

"""

import numpy as np

from time import time

#-----------------------------------------------------------------------------------------------
# SOR algorithm to iteratively solve a non linear Poisson problem of the form ∇^2*R = f(R)

def SOR(N, param, mat, source, initial_guess):

    """
    N:             state vector for species grid nxnxn_s
    param:         parameters of the simulation (dictionary)
    mat:           matrices of the simulation (dictionary)
    source:        source function (ATT: *args are R,N,param,mat; outputs nxn source matrix)
    initial_guess: initial guess for the solution nxnxn_r

    returns nxnxn_r vector of equilibrium concentration

    """

    # parameters needed for SOR implementation
    n     = param['n']                                    # number of original grid points
    R0    = param['R0']                                   # replenishment concentration nxnxn_r
    L     = param['L']                                    # length of the side in micrometers
    D     = param['D']                                    # diffusion constant
    stop  = param['acc']                                  # convergence criterion
    bound = param['ext']                                  # n_r vector with boundary conditions

    # finite difference length
    h     = L/n
    # number of resources
    n_r   = R0.shape[2]

    # updates initialization
    delta_list = [1]
    delta = np.zeros((n,n,n_r))

    # R initialization, source initialization
    current_best   = initial_guess                       

    # create frame for fixed BC
    BC = np.tile(bound, (current_best.shape[0]+2, current_best.shape[1]+2, 1))
    BC[1:n+1,1:n+1,:] = current_best
    best_BC = BC

    # start timing
    t0 = time()

    # SOR algorithm, convergence condition based on update relative to current absolute value
    while ((delta_list[-1]>best_BC*stop).any()):

        current_source, up = source(best_BC[1:n+1,1:n+1,:],N,param,mat)

        # prepare grid, give red and black colors        
        i, j = np.mgrid[1:n+1, 1:n+1]
        checkerboard  = (i + j) % 2
        red_squares   = checkerboard == 0
        black_squares = checkerboard == 1
        i_red = i[red_squares]
        j_red = j[red_squares]
        i_black = i[black_squares]
        j_black = j[black_squares]

        # red dots update
        delta[i_red-1,j_red-1,:] = 0.25*(best_BC[i_red,j_red+1,:]+best_BC[i_red,j_red-1,:]+best_BC[i_red+1,j_red,:]+best_BC[i_red-1,j_red,:]+(h**2/D)*current_source[i_red-1,j_red-1,:])-best_BC[i_red,j_red,:]
        best_BC[i_red,j_red] += delta[i_red-1,j_red-1]

        # black dots update
        delta[i_black-1,j_black-1,:] = 0.25*(best_BC[i_black,j_black+1,:]+best_BC[i_black,j_black-1,:]+best_BC[i_black+1,j_black,:]+best_BC[i_black-1,j_black,:]+(h**2/D)*current_source[i_black-1,j_black-1,:])-best_BC[i_black,j_black,:] 
        best_BC[i_black,j_black] += delta[i_black-1,j_black-1]

        # extract biggest update
        delta_list.append(np.max(np.abs(delta)))

        print("N_iter %d delta_max %e\r" % (len(delta_list)-1, delta_list[-1]), end='')

        # check for very small deltas
        if (np.abs(delta_list[-1])<1e-10):
            break

    # end timing and print time
    t1 = time()
    print('\n Time taken to solve for equilibrium: ', round((t1-t0)/60,4), ' minutes')

    # extract uptake at equilibrium condition
    _, up = source(best_BC[1:n+1,1:n+1,:],N,param,mat)

    return best_BC[1:n+1,1:n+1,:], up 

