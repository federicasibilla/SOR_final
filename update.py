"""
update.py:

"""

import numpy as np
from N_dynamics import *

from SOR_vanilla import *

def simulate(steps, source, initial_guess, initial_N, param, mat):

    """
    steps:         number of steps we want to run the simulation for
    source:        source function (ATT: *args are R,N,param,mat; outputs nxn source matrix)
    initial_guess: initial guess for the first equilibrium concentration
    initial_N:     initial composition of species grid
    param:         parameters dictionary
    mat:           matrices dictionary

    returns

    """

    # start timing simulation
    t0 = time()

    # extract list of all possible species
    all_species = list(range(len(param['g'])))

    # list to store steps of the population grid
    frames = [decode(initial_N)]

    # first iteration
    print('Solving iteration zero, finding equilibrium from initial guess')
    # computing equilibrium concentration at ztep zero
    current_R, up = SOR(initial_N, param, mat, source, initial_guess)
    # computing growth rates on all the grid
    g_rates   = growth_rates(initial_N,param,mat,up)
    # performing DB dynamics
    decoded_N,check,most_present = death_birth(decode(initial_N),g_rates)
    current_N = encode(decoded_N, all_species)
    frames.append(decoded_N)

    for i in range(steps):

        print("Step %i" % (i+1))

        # compute new equilibrium, initial guess is previous equilibrium
        current_R, up = SOR(current_N, param, mat, source, current_R)

        # compute growth rates
        g_rates   = growth_rates(current_N,param,mat,up)
        # performe DB dynamics
        decoded_N,check,most_present = death_birth(decode(current_N),g_rates)
        # check that there is more than one species
        if check == 'vittoria':
            print('winner species is: ', most_present)
            break
        frames.append(decoded_N)
        current_N = encode(decoded_N, all_species)

    # end timing
    t1 = time()
    print(f'\n Time taken to solve for {steps} steps: ', round((t1-t0)/60,4), ' minutes \n')

    return frames, current_R, current_N, g_rates


        

