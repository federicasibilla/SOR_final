"""
visualization.py: file containing the functions to make plots 

CONTAINS:
    - R_ongrid: takes the matrix nxnxn_R with concentrations and returns the plot of such distributions 
                in 2D, one surface for each nutrient on the grid it is defined on
    - R_ongrid_3D: takes the matrix nxnxn_R with concentrations and returns the plot of such distributions 
                in 3D, one surface for each nutrient on the grid it is defined on
    - N_ongrid: takes the matrix nxnxn_s with ones and zeros representing which species is present
                on each sites and returns the plot, with one color for each species
    - G_ongrid: takes the nxn matrix of growth rates and returns the plot on grid
    - makenet:  function to draw the metabolic network, takes the metabolic matrix as input
    - vispreferences: function to visualize the uptake preferences
    - animation_grid: function to animate the grid simulation

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import networkx as nx
import seaborn as sns

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from networkx.drawing.nx_agraph import to_agraph
from matplotlib.animation import FuncAnimation,ArtistAnimation
from matplotlib.animation import PillowWriter
sns.set(style='whitegrid')

#---------------------------------------------------------------------------------
# define R_ongrid(R) to visualize the equilibrium concentrations for each resource

def R_ongrid(R):
    """
    R: the matrix n x n x n_r with concentrations

    plots the equilibrium concentrations for each nutrient

    """

    # create the grid
    x = np.arange(R.shape[0])
    y = np.arange(R.shape[0])
    X, Y = np.meshgrid(x, y)

    # R matrix as function of x and y plot (one plot per nutrient)
    n_r = R.shape[2]
    fig, axs = plt.subplots(1, n_r, figsize=(18, 6))

    if n_r == 1:
        im = axs.imshow(R[:, :, 0], cmap='ocean')
        fig.colorbar(im)
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.set_title('Resource {}'.format(1))

    else:

        for i in range(n_r):
            ax = axs[i]
            im = ax.imshow(R[:, :, i], cmap='ocean')
            fig.colorbar(im, ax=ax, label='Concentration')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Resource {}'.format(i+1))

    plt.savefig('R.png')
    plt.close()

    return

#--------------------------------------------------------------------------------
# same but 3D

def R_ongrid_3D(R):
    """
    R: the matrix n x n x n_r with concentrations

    plots the equilibrium concentrations for each nutrient

    """

    # R matrix as function of x and y plot (one plot per nutrient)
    n_r = R.shape[2]
    fig = plt.figure(figsize=( n_r*4,10))

    if n_r!=1:
        for r in range(n_r):
            ax = fig.add_subplot(1, n_r, r+1, projection='3d')
            x = np.linspace(0, 1, R.shape[0])
            y = np.linspace(0, 1, R.shape[1])
            X, Y = np.meshgrid(x, y)
            ax.plot_surface(X, Y, R[:,:,r])
            ax.set_title(f'Resource {r+1}')

    else:
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(0, 1, R.shape[0])
        y = np.linspace(0, 1, R.shape[1])
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, R[:,:,0])
        ax.set_title(f'Resource {1}')

    plt.savefig('R_3D.png')
    plt.close()

    return

#---------------------------------------------------------------------------------
# define N_ongrid(R) to visualize the disposition of species 

def N_ongrid(N):
    """
    N: the nxnxn_s matrix containing nxn elements with length n_s composed by all zeros and
       one corresponding to the species present in the grid point (1 species per grid point)

    plots the grid with current species disposition

    """
    # define colors for species distinction
    cmap = plt.cm.get_cmap('bwr', N.shape[2])  
    norm = mc.Normalize(vmin=0, vmax=N.shape[2]-1)

    # plot gird
    colors = cmap(norm(np.argmax(N, axis=2)))
    plt.figure(figsize=(8, 8))

    plt.imshow(colors, interpolation='nearest')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm,ticks=np.arange(N.shape[2]),label='Species')

    plt.savefig('N0.png')
    plt.close()

    return

#---------------------------------------------------------------------------------
# define G_ongrid(G) function to visualize growth rates

def G_ongrid(G,N):
    """
    G: growth rates vector
    N: hot encoded species state

    returns grid with color gradient corresponding to growth rates

    """

    fig = plt.figure(figsize=(N.shape[2]*10,10))

    for species in range(N.shape[2]):

        ax = fig.add_subplot(1, N.shape[2], species+1)
        cmap = plt.cm.get_cmap('summer') 

        # Apply the mask to the G matrix
        G_masked = np.where(N[:,:,species] == 1, G, np.nan)

        # Plot the masked G matrix
        im = ax.imshow(G_masked, cmap=cmap)

        # Add a colorbar with the label 'Growth rate'
        fig.colorbar(im, ax=ax, label='Growth rate')

        ax.set_title(f'Species {species+1}')

    # Save the figure with all subplots
    plt.savefig('G_all_species.png')

    plt.close()

    return

#---------------------------------------------------------------------------------
# define makenet(met_matrix) to visualize the metabolic processes network, with
# resources as nodes and allocation magnitude as edges thikness

def makenet(met_matrix):
    """
    met_matrix: metabolic matrix, with resources as rows and columns and allocation rates as
                entries

    returns the graph of metabolic allocations

    """
    G = nx.DiGraph()

    for i in range(met_matrix.shape[0]):
        for j in range(met_matrix.shape[1]):
            G.add_edge(f"Res{j+1}", f"Res{i+1}", weight=met_matrix[i, j])

    # draw graph
    agraph = to_agraph(G)
    agraph.layout(prog='dot', args='-GK=0.5 -Gsep=3 -Ncolor=lightblue -Nstyle=filled -Npenwidth=2 -Ecolor=gray -Nnodesep=0.1')
    for edge in agraph.edges():
        weight = G[edge[0]][edge[1]]['weight']
        agraph.get_edge(*edge).attr['penwidth'] = weight * 5
    img = agraph.draw(format='png')
    with open('met_net.png', 'wb') as f:
        f.write(img)

    return

#---------------------------------------------------------------------------------
# defining vispreferences(up_mat) function to visualize the uptake preferences 
# of the different species

def vispreferences(mat):
    """
    up_mat: uptake matrix of the different species and resources

    returns a graph to visualize uptake preferences 

    """
    up_mat = mat['uptake']*mat['sign']

    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, up_mat.shape[1]))  

    legend = 0
    for i in range(up_mat.shape[0]):
        offset = 0 
        offset_neg = 0
    
        for j in range(up_mat.shape[1]):
            lunghezza_segmento = up_mat[i, j]  
            if (lunghezza_segmento>=0):
                if legend<up_mat.shape[1]:
                    plt.bar(i, lunghezza_segmento, bottom=offset, width=0.8, color=colors[j], label=f'Res {j+1}')
                    offset += lunghezza_segmento
                    legend +=1
                else:
                    plt.bar(i, lunghezza_segmento, bottom=offset, width=0.8, color=colors[j])
                    offset += lunghezza_segmento
            else:
                if legend<up_mat.shape[1]:
                    plt.bar(i, lunghezza_segmento, bottom=offset_neg, width=0.8, color=colors[j], label=f'Res {j+1}')
                    offset_neg += lunghezza_segmento
                    legend +=1
                else:
                    plt.bar(i, lunghezza_segmento, bottom=offset_neg, width=0.8, color=colors[j])
                    offset_neg += lunghezza_segmento



    plt.xlabel('Species')
    plt.ylabel('Uptake')
    plt.title('Consumer preferences')
    plt.legend()
    plt.grid(True) 

    plt.savefig('uptake_pref.png')
    plt.close()

    return

#---------------------------------------------------------------------------------------------------
# function to visualize the animation of population grid 

def animation_grid(steps):
    """
    steps: list of grid steps decoded

    returns nothing, saves the animation

    """
    
    fig = plt.figure()
    ims = []

    for i, step in enumerate(steps):
        im = plt.imshow(step, cmap='bwr', animated=True)
        plt.axis('off')  
        plt.grid(False)
        ims.append([im])

    ani = ArtistAnimation(fig, ims, interval=5, repeat=False)

    ani.save('animation.mp4', writer='ffmpeg')

    return

#---------------------------------------------------------------------------------------------------------
# function to visualize aboundances and determin if steady state is reached

def abundances(steps):
    """
    steps: list of matrices where each element represents a point in time
    
    returns abundance_series: time series matrix of abundances
    """

    # Initialize an empty list to store abundance time series
    abundance_series = []

    # Get the number of unique integers present in the matrices
    num_unique_integers = len(np.unique(steps[0]))

    # Iterate over each time step (matrix)
    for step in steps:
        # Initialize an empty list to store abundances for this time step
        step_abundances = []

        # Count the occurrences of each integer in the matrix
        for i in range(num_unique_integers):
            abundance = np.count_nonzero(step == i)
            step_abundances.append(abundance)

        # Append the abundances for this time step to the abundance series
        abundance_series.append(step_abundances)

    # Convert the list of lists to a numpy array
    abundance_series = np.array(abundance_series)

    # Plot the abundance time series
    plt.figure(figsize=(8, 6))
    for i in range(num_unique_integers):
        plt.plot(abundance_series[:, i], label=f'Species {i+1}')

    plt.xlabel('Time Step')
    plt.ylabel('Abundance')
    plt.title('Abundance Time Series')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('abundances.png')

    return abundance_series

    