import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_2D_occupancy(u):
    """
    plot 2d occupancy
    """
    u = u.flatten()
    grid_resolution = np.round(u[:].shape[0] ** (1./2)).astype(int)
    levels = np.linspace(-1e-5, 1+1e-5, 101)
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list('occupancy', [(0.0,'#92c5de'),(0.45, '#ffffff'), (0.5, '#000000'), (0.55, '#ffffff'), (1.0,'#ef8a62')], N=256) # red blue
    plot = plt.contourf(u.reshape(grid_resolution,grid_resolution).T, levels = levels, cmap=colormap, origin="lower")
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.axis('off')

def sdf_circle(x, r, center):
    """
    output the SDF value of a circle in 2D

    Inputs
    x: nx2 array of locations
    r: radius of the circle
    center: center point of the circle

    Outputs
    array of signed distance values at x
    """
    dx = x - center
    return np.sqrt(np.sum((dx)**2, axis = 1)) - r

def sdf_square(p, size, translation): 
    """
    output the signed distance value of a square in 2D

    Inputs
    p: nx2 array of locations

    Outputs
    array of signed distance values at x
    """
    Tp = p - translation[None,:]
    d = np.abs(Tp) - size
    return np.sqrt(np.sum(np.maximum(d,0.0)**2,1)) + np.minimum(np.maximum(d[:,0],d[:,1]), 0.0)

def sigmoid(x):
    positive = x >= 0
    negative = ~positive
    result = np.empty_like(x)
    result[positive] = 1 / (1 + np.exp(-x[positive]))
    exp_x_neg = np.exp(x[negative])
    result[negative] = exp_x_neg / (exp_x_neg + 1)
    return result

def sdf_to_occupancy(x, sharpness):
    return sigmoid(-x * sharpness)

def fuzzy_boolean(c, x, y):
    """
    Fuzzy boolean operator

    Inputs:
        c: (4,) numpy array of boolean parameters (0 <= c[i] <= 1 and sum(c) = 1)
        x: (n,) numpy array of input occupancy between 0 and 1
        y: (n,) numpy array of input occupancy between 0 and 1
    Output:
        (4,) numpy array of occupancy resulting from the boolean operation
    
    Notes:
        INTERSECTION = np.array([1,0,0,0])
        UNION = np.array([0,1,0,0])
        LEFT_DIFFERENCE_RIGHT = np.array([0,0,1,0])
        RIGHT_DIFFERENCE_LEFT = np.array([0,0,0,1])
    """
    assert( np.all(c >= 0) )
    assert( np.all(c <= 1) )
    assert( np.isclose(np.sum(c), 1.0) ) 
    assert( np.all(x >= 0) )
    assert( np.all(x <= 1) )
    assert( np.all(y >= 0) )
    assert( np.all(y <= 1) )
    return (c[1] + c[2]) * x + (c[1] + c[3]) * y + (c[0] - c[1] - c[2] - c[3]) * x*y 

# sample 2d points on a grid
resolution = 128 
idx = np.linspace(-1,1,num=resolution)
x, y = np.meshgrid(idx, idx)
V = np.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), 1)

# get input sdfs
sharpness0 = 50.0
sdf0 = sdf_square(V, 0.5,  np.array([0.25,0.25]))
o0 = sdf_to_occupancy(sdf0, sharpness0)

sharpness1 = 50.0
sdf1 = sdf_circle(V, 0.55, np.array([-0.25,-0.25]))
o1 = sdf_to_occupancy(sdf1, sharpness1)

# plot input sdfs
plt.figure(0)
plt.clf()
plot_2D_occupancy(o0)
plt.savefig("occupancy0.png")

plt.figure(1)
plt.clf()
plot_2D_occupancy(o1)
plt.savefig("occupancy1.png")

# save fuzzy logic results
INTERSECTION = np.array([1,0,0,0])
UNION = np.array([0,1,0,0])
LEFT_DIFFERENCE_RIGHT = np.array([0,0,1,0])
RIGHT_DIFFERENCE_LEFT = np.array([0,0,0,1])

plt.figure(2)
plt.clf()
plot_2D_occupancy(fuzzy_boolean(INTERSECTION, o0, o1))
plt.savefig("intersection.png")

plt.figure(3)
plt.clf()
plot_2D_occupancy(fuzzy_boolean(UNION, o0, o1))
plt.savefig("union.png")

plt.figure(4)
plt.clf()
plot_2D_occupancy(fuzzy_boolean(LEFT_DIFFERENCE_RIGHT, o0, o1))
plt.savefig("left_difference_right.png")

plt.figure(5)
plt.clf()
plot_2D_occupancy(fuzzy_boolean(RIGHT_DIFFERENCE_LEFT, o0, o1))
plt.savefig("right_difference_left.png")

