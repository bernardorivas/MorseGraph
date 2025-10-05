# PlotMorseSets.py
# Marcio Gameiro
# MIT LICENSE
# 2023-07-15

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import CMGDB

def PlotMorseSets(morse_sets, morse_nodes=None, proj_dims=None, cmap=None, clist=None,
                  fig_w=8, fig_h=8, xlim=None, ylim=None, axis_labels=True,
                  xlabel='$x$', ylabel='$y$', fontsize=15, fig_fname=None, dpi=300):
    # Check if morse_sets is a Morse graph, file name, or list
    if type(morse_sets) == CMGDB._cmgdb.MorseGraph: # Morse graph
        morse_graph = morse_sets
        num_morse_sets = morse_graph.num_vertices()
        if morse_nodes == None:
            morse_nodes = range(num_morse_sets)
        morse_sets = [box + [node] for node in morse_nodes for box in morse_graph.morse_set_boxes(node)]
    elif type(morse_sets) == str: # String representing file name
        morse_fname = morse_sets
        morse_sets = CMGDB.LoadMorseSetFile(morse_fname)
        num_morse_sets = None
    else: # List of Morse sets
        num_morse_sets = None
    # Plot Morse set boxes as a scatter plot
    PlotBoxesScatter(morse_sets, num_morse_sets=num_morse_sets, morse_nodes=morse_nodes,
                     proj_dims=proj_dims, cmap=cmap, clist=clist, fig_w=fig_w, fig_h=fig_h,
                     xlim=xlim, ylim=ylim, axis_labels=axis_labels, xlabel=xlabel,
                     ylabel=ylabel, fontsize=fontsize, fig_fname=fig_fname, dpi=dpi)

def PlotBoxesScatter(morse_sets, num_morse_sets=None, morse_nodes=None, proj_dims=None, cmap=None,
                     clist=None, fig_w=8, fig_h=8, xlim=None, ylim=None, axis_labels=True,
                     xlabel='$x$', ylabel='$y$', fontsize=15, fig_fname=None, dpi=300):
    # Default colormap
    default_cmap = matplotlib.cm.tab20
    rect = morse_sets[0]
    assert len(rect) % 2 == 1, "Wrong dimension in Morse sets data"
    dim = int((len(rect) - 1) / 2)
    if dim == 1:
        # Add extra fake dimension to plot
        x0_min, x0_max, node = morse_sets[0]
        x1_min = 0
        x1_max = x0_max - x0_min
        morse_sets = [[x0_min, x1_min, x0_max, x1_max, node] for x0_min, x0_max, node in morse_sets]
        dim = 2
    if num_morse_sets == None:
        num_morse_sets = max([int(rect[-1]) for rect in morse_sets]) + 1
    if morse_nodes == None:
        morse_nodes = range(num_morse_sets)
    # Set colormap to use
    if clist and cmap == None:
        cmap = matplotlib.colors.ListedColormap(clist[:num_morse_sets], name='clist')
    if cmap == None:
        cmap = default_cmap
    if proj_dims == None:
        d1 = 0
        d2 = 1
    else:
        d1 = proj_dims[0]
        d2 = proj_dims[1]
    assert max(d1, d2) < dim, "Wrong projection dimensions"
    # Get min and max x and y values
    if xlim == None:
        x_min = min([rect[d1] for rect in morse_sets if int(rect[-1]) in morse_nodes])
        x_max = max([rect[dim + d1] for rect in morse_sets if int(rect[-1]) in morse_nodes])
        if x_max - x_min < 0.1:
            x_min -= 0.05
            x_max += 0.05
    else:
        x_min = xlim[0]
        x_max = xlim[1]
    if ylim == None:
        y_min = min([rect[d2] for rect in morse_sets if int(rect[-1]) in morse_nodes])
        y_max = max([rect[dim + d2] for rect in morse_sets if int(rect[-1]) in morse_nodes])
        if y_max - y_min < 0.1:
            y_min -= 0.05
            y_max += 0.05
    else:
        y_min = ylim[0]
        y_max = ylim[1]
    x_axis_width = x_max - x_min
    y_axis_height = y_max - y_min
    # Next plot Morse sets
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    # Set axis limits
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    # Normalization for color map
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_morse_sets-1)
    # The scatter plot in Matplotlib uses a marker size in points**2 units.
    # The relationship between points and pixels is 1 point = dpi/72 pixels,
    # hence multiplying by 72 / fig.dpi gives the number of pixels in the plot.
    # For example using size s=10**2 in a scatter plot will plot the markers
    # with a size of 100 points^2, which changes size according to the dpi.
    # On the other hand using size s=(10*72.0/fig.dpi)**2 will plot markers
    # of size 100 pixels^2, which will be the same size regardless of the dpi.
    # First we get unit size in data units, that is, the size in pixels that
    # correspond to one unit of data (one unit in each axis). The value needs
    # to be squared later for the scatter plot.
    s0_x = (ax.get_window_extent().width / x_axis_width) * (72.0 / fig.dpi)
    s0_y = (ax.get_window_extent().height / y_axis_height) * (72.0 / fig.dpi)
    # Alternative way to set marker size in data units
    # M = ax.transData.get_matrix()
    # x_scale = M[0,0]
    # y_scale = M[1,1]
    for morse_node in morse_nodes:
        morse_set = [rect for rect in morse_sets if int(rect[-1]) == morse_node]
        # Use morse_node as color index for consistency if not plotting all
        clr = matplotlib.colors.to_hex(cmap(cmap_norm(morse_node)), keep_alpha=True)
        X = []; Y = []; S = []
        for rect in morse_set:
            p1 = [rect[d1], rect[d2]]                   # Lower point
            p2 = [rect[dim + d1], rect[dim + d2]]       # Upper point
            p = list((np.array(p1) + np.array(p2)) / 2) # Center point
            s = list((np.array(p2) - np.array(p1)))     # Rect size
            s_x = (s0_x * s[0]) ** 2 # Scatter x-axis size
            s_y = (s0_y * s[1]) ** 2 # Scatter y-axis size
            # Alternative way to set marker size in data units
            # s_x = (x_scale * s[0]) ** 2 # Scatter x-axis size
            # s_y = (y_scale * s[1]) ** 2 # Scatter y-axis size
            X.append(p[0])
            Y.append(p[1])
            # Use max of both sizes
            S.append(max(s_x, s_y))
        ax.scatter(X, Y, s=S, marker='s', c=clr)
        # ax.scatter(X, Y, s=S, marker='s', c=clr, edgecolors=None)
        # ax.scatter(X, Y, s=S, marker='s', c=clr, alpha=0.5)
    # Add axis labels
    if axis_labels:
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
    # Set tick labels size
    ax.tick_params(labelsize=fontsize)
    # ax.set_aspect('equal')
    # plt.grid(True)
    if fig_fname:
        fig.savefig(fig_fname, dpi=dpi, bbox_inches='tight')
    plt.show()

def _box_to_cuboid_faces(box):
    """
    Convert a 3D box to 6 faces for Poly3DCollection rendering.

    :param box: List/array of [x_min, y_min, z_min, x_max, y_max, z_max, node]
    :return: List of 6 faces, each face is a list of 4 (x,y,z) vertices
    """
    x_min, y_min, z_min = box[0], box[1], box[2]
    x_max, y_max, z_max = box[3], box[4], box[5]

    # Define 8 vertices of the cuboid
    vertices = [
        [x_min, y_min, z_min],  # 0: bottom-front-left
        [x_max, y_min, z_min],  # 1: bottom-front-right
        [x_max, y_max, z_min],  # 2: bottom-back-right
        [x_min, y_max, z_min],  # 3: bottom-back-left
        [x_min, y_min, z_max],  # 4: top-front-left
        [x_max, y_min, z_max],  # 5: top-front-right
        [x_max, y_max, z_max],  # 6: top-back-right
        [x_min, y_max, z_max],  # 7: top-back-left
    ]

    # Define 6 faces (each face has 4 vertices in counter-clockwise order)
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom (z=min)
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top (z=max)
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front (y=min)
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back (y=max)
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left (x=min)
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right (x=max)
    ]

    return faces

def PlotMorseSets3D(morse_sets, morse_nodes=None, cmap=None, clist=None,
                    fig_w=10, fig_h=10, alpha=0.3, elev=30, azim=45,
                    xlim=None, ylim=None, zlim=None, axis_labels=True,
                    xlabel='$x$', ylabel='$y$', zlabel='$z$',
                    fontsize=15, fig_fname=None, dpi=300):
    """
    Plot 3D Morse sets using cuboid visualization with Poly3DCollection.

    :param morse_sets: Morse graph object, filename, or list of boxes
    :param morse_nodes: List of Morse node indices to plot (default: all)
    :param cmap: Matplotlib colormap for Morse sets
    :param clist: List of colors to create a custom colormap
    :param fig_w: Figure width in inches
    :param fig_h: Figure height in inches
    :param alpha: Transparency of cuboids (0.0-1.0, lower shows interior structure)
    :param elev: Elevation angle for 3D view (degrees)
    :param azim: Azimuth angle for 3D view (degrees)
    :param xlim: X-axis limits [x_min, x_max]
    :param ylim: Y-axis limits [y_min, y_max]
    :param zlim: Z-axis limits [z_min, z_max]
    :param axis_labels: Whether to show axis labels
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param zlabel: Z-axis label
    :param fontsize: Font size for labels
    :param fig_fname: Filename to save figure (optional)
    :param dpi: DPI for saved figure
    """
    # Check if morse_sets is a Morse graph, file name, or list
    if type(morse_sets) == CMGDB._cmgdb.MorseGraph:  # Morse graph
        morse_graph = morse_sets
        num_morse_sets = morse_graph.num_vertices()
        if morse_nodes == None:
            morse_nodes = range(num_morse_sets)
        morse_sets = [box + [node] for node in morse_nodes for box in morse_graph.morse_set_boxes(node)]
    elif type(morse_sets) == str:  # String representing file name
        morse_fname = morse_sets
        morse_sets = CMGDB.LoadMorseSetFile(morse_fname)
        num_morse_sets = None
    else:  # List of Morse sets
        num_morse_sets = None

    # Verify dimension is 3
    rect = morse_sets[0]
    assert len(rect) % 2 == 1, "Wrong dimension in Morse sets data"
    dim = int((len(rect) - 1) / 2)
    if dim != 3:
        raise ValueError(f"PlotMorseSets3D requires 3D data, got {dim}D. Use PlotMorseSets for 2D data.")

    # Determine number of Morse sets
    if num_morse_sets == None:
        num_morse_sets = max([int(rect[-1]) for rect in morse_sets]) + 1
    if morse_nodes == None:
        morse_nodes = range(num_morse_sets)

    # Set colormap to use
    default_cmap = matplotlib.cm.tab20
    if clist and cmap == None:
        cmap = matplotlib.colors.ListedColormap(clist[:num_morse_sets], name='clist')
    if cmap == None:
        cmap = default_cmap

    # Normalization for color map
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_morse_sets-1)

    # Create 3D figure
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Get axis limits if not provided
    if xlim == None:
        x_min = min([rect[0] for rect in morse_sets if int(rect[-1]) in morse_nodes])
        x_max = max([rect[3] for rect in morse_sets if int(rect[-1]) in morse_nodes])
    else:
        x_min, x_max = xlim

    if ylim == None:
        y_min = min([rect[1] for rect in morse_sets if int(rect[-1]) in morse_nodes])
        y_max = max([rect[4] for rect in morse_sets if int(rect[-1]) in morse_nodes])
    else:
        y_min, y_max = ylim

    if zlim == None:
        z_min = min([rect[2] for rect in morse_sets if int(rect[-1]) in morse_nodes])
        z_max = max([rect[5] for rect in morse_sets if int(rect[-1]) in morse_nodes])
    else:
        z_min, z_max = zlim

    # Plot each Morse set
    for morse_node in morse_nodes:
        morse_set = [rect for rect in morse_sets if int(rect[-1]) == morse_node]
        # Use morse_node as color index for consistency
        clr = matplotlib.colors.to_hex(cmap(cmap_norm(morse_node)), keep_alpha=True)

        # Collect all faces for this Morse set
        faces = []
        for rect in morse_set:
            faces.extend(_box_to_cuboid_faces(rect))

        if faces:
            # Create 3D polygon collection for this Morse set
            poly3d = Poly3DCollection(faces, facecolors=clr, alpha=alpha,
                                     edgecolors='none', linewidths=0)
            ax.add_collection3d(poly3d)

    # Set axis limits
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Add axis labels
    if axis_labels:
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_zlabel(zlabel, fontsize=fontsize)

    # Set tick labels size
    ax.tick_params(labelsize=fontsize)

    # Save figure if filename provided
    if fig_fname:
        fig.savefig(fig_fname, dpi=dpi, bbox_inches='tight')

    plt.show()
