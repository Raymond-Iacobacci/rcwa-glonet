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
p['pixelsX'] = 20
p['pixelsY'] = 20
p['N'] = 1000
p['sigmoid_update'] = 1
p['learning_rate'] = 0.099
p['parameter_string'] = 'N' + str(p['N']) + '-sigmoid_update' + str(p['sigmoid_update']) + '-learning_rate' + str(p['learning_rate'])
step = 10
wavelengths = np.arange(400, 600 + step, step = step)/1000.0
passthrough_wavelength_index = 10
determinator = -np.concatenate((np.arange(0, passthrough_wavelength_index + 1, dtype = np.float64), np.arange(passthrough_wavelength_index - 1, -1, -1, dtype = np.float64)))
determinator[passthrough_wavelength_index] = 50
determinator -= np.mean(determinator) # No bias traps
determinator /= determinator[passthrough_wavelength_index] # Normal scaling
determinator = torch.tensor(determinator, dtype = torch.float32) # TODO: check if 'dtype' makes the difference when initializing tensors to run on the GPU architecture
p['thetas'] = np.zeros(len(wavelengths), dtype = 'float32')
p['phis'] = deepcopy(p['thetas'])
p['pte'] = deepcopy(p['thetas']) + 1
p['ptm'] = deepcopy(p['thetas'])
p['Lx'] = p['Ly'] = .1 # TODO Micrometers?, Ly left in because we might expand in that direction eventually
p['erd'] = 3.4 # Edit from ers
p['L'] = np.zeros(shape = (3,)) + 5.0
p['f'] = 0
p['initial_k'] = np.zeros(shape = (p['L'].shape[0], p['pixelsX'], p['pixelsY'])) + 2.7
p['PQ'] = [3, 3]
p['upsample'] = 2
p['sigmoid_coeff'] = 0.1
p['enable_random_init'] = p['enable_debug'] = p['enable_print'] = p['enable_timing'] = p['enable_logging'] = True
copy_dt_string = '26:01:2024-13:49'
p['restore_from'] = copy_dt_string
from datetime import datetime
init_now = datetime.now()
init_dt_string = init_now.strftime('%d:%m:%Y-%H:%M')
p['time'] = init_dt_string
def loss_function(k, params):
    ER_t, UR_t = solver_metasurface_pt.generate_metasurface(k, params)
    binarization_loss = torch.sum((ER_t - 1.0) * (3.4 - ER_t))/(2.7 ** 2)/(20 * 20 * 3)/(21 * 16 * 16)
    outputs = solver_pt.simulate(ER_t, UR_t, params)
    field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0] #TODO: understand why we're taking the 4 in the answer even in the working solution
    focal_plane = solver_pt.propagate(params['input'] * field, params['propagator'], params['upsample'])
    p = torch.sum(torch.abs(focal_plane), dim = (-1, -2)) # Deleting gradients? Find gradients with track=True and backpropagate throughout network
    with open(f'loss_{init_dt_string}/wavelength_loss.txt', 'a+') as f:
        f.write(str(p.cpu().detach().numpy()))
        f.write('\n')
    print(f"Binarization: {np.real(binarization_loss.cpu().detach().numpy())}")
    return torch.tensordot(determinator.type(p.type()), p, dims = 1) # Need a binarization coefficient
p['loss_function'] = loss_function
p['wavelengths'] = wavelengths
k, loss, params, focal_plane = solver_metasurface_pt.optimize_device(p)
print(loss)
