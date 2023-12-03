from sys import path
path.append('../src/')
path.append('../rcwa_tf/src/')
import tensorflow as tf
import solver
import solver_metasurface
import numpy as np
from copy import deepcopy
p = {}
p['enable_print'] = True
p['pixelsX'] = 100
p['pixelsY'] = 1
p['N'] = 200
p['sigmoid_update'] = 10
p['learning_rate'] = 0.01
p['parameter_string'] = 'N' + str(p['N']) \
    + '-sigmoid_update' + str(p['sigmoid_update']) \
    + '-learning_rate' + str(p['learning_rate'])
step = 30
wavelengths = np.arange(400, 600 + step, step = step)/1000.0
passthrough_band_indices = [4,5,6,7]
determinator = np.array([-1/(len(wavelengths) - len(passthrough_band_indices))] * len(wavelengths))
determinator[passthrough_band_indices] = 1/len(passthrough_band_indices)
determinator = tf.constant(determinator, dtype = tf.float64)
# wavelengths = [120.0]
p['thetas'] = np.zeros(len(wavelengths), dtype = 'float32')
p['phis'] = deepcopy(p['thetas'])
p['pte'] = deepcopy(p['thetas']) + 1
p['ptm'] = deepcopy(p['thetas'])
p['Lx'] = p['Ly'] = 100 # Nanometers
p['erd'] = p['ers'] = 3.4
# Check all dimensions when done, also check reflectivity coefficients
p['L'] = np.zeros(shape = (100,)) + 5.0 #THIS IS THE NUMBER OF LAYERS, WTF IS GOING ON????????
# p['L'] = deepcopy(p['thetas']) + 50.0
p['f'] = 0.0
p['initial_k'] = np.zeros(shape = (len(p['L']) - 1, p['pixelsX'], p['pixelsY'])) + 1
for i in range(len(p['initial_k'])):
    for j in range(len(p['initial_k'][i])):
        for k in range(len(p['initial_k'][i][j])):
            if (i+j+k)%2:
                p['initial_k'][i][j][k] += 0.5
p['PQ'] = [3,3]
p['upsample'] = 2 # Density of measurements
p['sigmoid_coeff'] = 0.1
p['enable_random_init'] = True
p['enable_debug'] = False
p['enable_print'] = True
p['enable_timing'] = True
p['enable_logging'] = True
rx = int(p['Lx'] * p['pixelsX'])
ry = int(p['Ly'] * p['pixelsY'])
def loss_function(k, params):
    ER_t, UR_t = solver_metasurface.generate_scaled_metasurface(k, params)
    outputs = solver.simulate_allsteps(ER_t, UR_t, params)
    field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
    focal_plane = solver.propagate(params['input'] * field, params['propagator'], params['upsample'])
    indexX = (params['pixelsX'] * params['upsample']) // 2
    indexY = (params['pixelsY'] * params['upsample']) // 2
    p = tf.reduce_sum(-tf.abs(focal_plane[:, indexX - rx : indexX + rx, indexY - ry : indexY + ry]), [-1, -2]) # Deleting gradients? Find gradients with track=True and backpropagate throughout network
    # print(p.shape)
    # return tf.reduce_sum(p)
    print(p)
    return tf.tensordot(determinator, tf.cast(p, dtype = tf.float64), 1)
p['loss_function'] = loss_function
p['wavelengths'] = wavelengths
print("Initialized parameters, starting optimization routine")
k, loss, params, focal_plane = solver_metasurface.optimize_device(p)
print(loss)
print(k)



