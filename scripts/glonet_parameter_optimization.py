# Use these parameters to choose which devices to use.
use_GPU = False

# Import device utils.
import sys
import os
sys.path.append('../src/')
sys.path.append('../rcwa_tf/src/')
import utils

# Configure GPUs.
if (use_GPU): utils.config_gpu_memory_usage()

# Measure GPU memory usage.
if (use_GPU):
    gpu_memory_init = utils.gpu_memory_info()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import solver
import solver_metasurface
import importlib
importlib.reload(solver_metasurface)

# Initialize parameters.
user_params = {}

# Tunable parameters.
# These are the values used if hyperparameter grid search is disabled.
user_params['pixelsX'] = int(sys.argv[1])
user_params['N'] = int(sys.argv[2])
user_params['sigmoid_update'] = float(sys.argv[3])
user_params['learning_rate'] = float(sys.argv[4])
user_params['initial_height'] = int(sys.argv[5])

user_params['parameter_string'] = 'N' + str(user_params['N']) \
    + '-sigmoid_update' + str(user_params['sigmoid_update']) \
    + '-learning_rate' + str(user_params['learning_rate']) \
    + '-initial_height' + str(user_params['initial_height'])

# Source parameters.
user_params['wavelengths'] = [120.0]
user_params['thetas'] = [0.0]
user_params['phis'] = [0.0]
user_params['pte'] = [1.0]
user_params['ptm'] = [0.0]

# Device parmeters.
# //RAYMOND// user_params['pixelsY'] = 100
user_params['pixelsY'] = user_params['pixelsX']
# user_params['pixelsY'] = 1
user_params['erd'] = 11.9
user_params['ers'] = user_params['erd']
user_params['L'] = [50.0, 50.0, 50.0, 50.0, 50.0, 950.0]
user_params['Lx'] = 5000.0 / user_params['pixelsX']
# //RAYMOND//user_params['Ly'] = 2500.0 / user_params['pixelsY']
user_params['Ly'] = user_params['Lx']
user_params['f'] = 0.0 # Focal distance (nm)
user_params['initial_k'] = np.zeros(shape = (len(user_params['L']) - 1, user_params['pixelsX'], user_params['pixelsY']))+.5

# Solver parameters.
user_params['PQ'] = [3,3]
user_params['upsample'] = 11

# Problem parameters.
user_params['focal_spot_radius'] = 1
user_params['sigmoid_coeff'] = 0.1
user_params['enable_random_init'] = False
user_params['enable_debug'] = False
user_params['enable_print'] = True
user_params['enable_timing'] = True

# Logging parameters.
user_params['enable_logging'] = True
user_params['log_filename_prefix'] = './results/nearfield-' + str(user_params['pixelsX']) + 'x' + str(user_params['pixelsY']) + '-'
user_params['log_filename_extension'] = '.txt'
# Let h be again a set of floats, but now normalize the floats and have them correspond to cross-rectangle permittivity
def loss_function(k, params):
    
    # Generate permittivity and permeability distributions.
    # Change to generate_scaled_metasurface
    ER_t, UR_t = solver_metasurface.generate_scaled_metasurface(k, params)

    # Simulate the system.
    outputs = solver.simulate_allsteps(ER_t, UR_t, params)
    
    # First loss term: maximize sum of electric field magnitude within some radius of the desired focal point.
    rx = int(user_params['Lx'] * user_params['pixelsX'])
    ry = int(user_params['Ly'] * user_params['pixelsY'])
    field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]

    focal_plane = solver.propagate(params['input'] * field, params['propagator'], params['upsample'])
    index = (params['pixelsX'] * params['upsample']) // 2
    l1 = tf.reduce_sum(tf.abs(focal_plane[0, index-rx:index+rx, index-ry:index+ry]))

    # Final loss: (negative) field intensity at focal point + field intensity elsewhere.
    # return -params['w_l1']*l1
    return -l1

# Set loss function.
user_params['loss_function'] = loss_function
    
# Optimize.
k, loss, params, focal_plane = solver_metasurface.optimize_device(user_params)

# print(k,loss,params,focal_plane)
print(loss)