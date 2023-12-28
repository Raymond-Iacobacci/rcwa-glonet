'''
    From solver_pt:
    - Wavelengths are in micrometers
    - Theta and Phi are in degrees
    - Manually set permeability of device to ≠1 value in solver_pt
    - Ny and Nx are artificially set to similar values inside solver_pt
    - Upsample is artificially set to 1? Done in initialize_params, changing it
    - L is layer thickness in nanometers--total comes out to half a micrometer
'''
from sys import path
path.append('../src/')
path.append('../rcwa_pt/src/')
import torch, solver_pt, solver_metasurface_pt, numpy as np, utils
use_GPU = torch.cuda.is_available()
if use_GPU:
    print("Nvidia drivers correctly installed, using GPUs\n\n")
device = torch.device('cuda:0' if use_GPU else 'cpu') # TODO: this assumes that we're working with a single GPU, please fix, also add in device as passthrough var for all funcs
from copy import deepcopy
if use_GPU:
    utils.config_gpu_memory_usage()
p = {}
p['enable_print'] = True
p['pixelsX'] = 100
p['pixelsY'] = 1
p['N'] = 1000
p['sigmoid_update'] = 10
p['learning_rate'] = 0.01
p['parameter_string'] = 'N' + str(p['N']) + '-sigmoid_update' + str(p['sigmoid_update']) + '-learning_rate' + str(p['learning_rate'])
step = 10
wavelengths = np.arange(400, 600 + step, step = step)/1000.0
passthrough_band_indices = [9,10,11]
determinator = np.array([-1/(len(wavelengths) - len(passthrough_band_indices))] * len(wavelengths))
determinator[passthrough_band_indices] = 1/len(passthrough_band_indices)
determinator = torch.tensor(determinator, dtype = torch.float32) # TODO: check if 'dtype' makes the difference when initializing tensors to run on the GPU architecture
p['thetas'] = np.zeros(len(wavelengths), dtype = 'float32')
p['phis'] = deepcopy(p['thetas'])
p['pte'] = deepcopy(p['thetas']) + 1
p['ptm'] = deepcopy(p['thetas'])
p['Lx'] = p['Ly'] = 100 # Nanometers, Ly left in because we might expand in that direction eventually
p['erd'] = 3.4 # Edit from ers
p['L'] = np.zeros(shape = (100,)) + 5.0
p['f'] = 1e9
p['initial_k'] = np.zeros(shape = (p['L'].shape[0], p['pixelsX'], p['pixelsY'])) + 2.7
p['PQ'] = [3, 3]
p['upsample'] = 2
p['sigmoid_coeff'] = 0.1
p['enable_random_init'] = p['enable_debug'] = p['enable_print'] = p['enable_timing'] = p['enable_logging'] = True
def loss_function(k, params):
    ER_t, UR_t = solver_metasurface_pt.generate_metasurface(k, params)
    print("Generated metasurface")
    outputs = solver_pt.simulate(ER_t, UR_t, params)
    print("Started simulation")
    field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0] #TODO: understand why we're taking the 4 in the answer even in the working solution
    print("Building propagator")
    focal_plane = solver_pt.propagate(params['input'] * field, params['propagator'], params['upsample'])
    p = torch.sum(-torch.abs(focal_plane), dim = (-1, -2)) # Deleting gradients? Find gradients with track=True and backpropagate throughout network
    return torch.tensordot(determinator, p.double(), dims = 1)
p['loss_function'] = loss_function
p['wavelengths'] = wavelengths
k, loss, params, focal_plane = solver_metasurface_pt._optimize_device(p)
print(loss)