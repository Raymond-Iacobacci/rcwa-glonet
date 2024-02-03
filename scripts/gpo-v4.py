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
p['sigmoid_update'] = 19
p['learning_rate'] = 1
p['parameter_string'] = 'N' + str(p['N']) + '-sigmoid_update' + str(p['sigmoid_update']) + '-learning_rate' + str(p['learning_rate'])
step = 10
wavelengths = np.arange(400, 600 + step, step = step)/1000.0
passthrough_wavelength_index = 10
determinator = -np.concatenate((np.arange(1, passthrough_wavelength_index + 2, dtype = np.float64), np.arange(passthrough_wavelength_index, 0, -1, dtype = np.float64)))
determinator[passthrough_wavelength_index] = 50
determinator -= np.mean(determinator) # No bias traps
determinator /= determinator[passthrough_wavelength_index] # Normal scaling, scaling ranges between 1 and -1
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
p['sigmoid_coeff'] = 1
p['enable_random_init'] = p['enable_debug'] = p['enable_print'] = p['enable_timing'] = p['enable_logging'] = True
copy_dt_string = ''
p['restore_from'] = copy_dt_string
from datetime import datetime
init_now = datetime.now()
init_dt_string = init_now.strftime('%m:%d:%H:%M')
p['time'] = init_dt_string
def loss_function(k, params, image_number = None, epoch_number = None):
    ER_t, UR_t = solver_metasurface_pt.generate_metasurface(k, params)
    image_pixels = ER_t[0, :, :, :, 0, 0]
    binarization_loss = torch.sum((image_pixels - 1.0) * (3.4 - image_pixels)/(1.7 ** 2)/(20 * 20 * 3))
    outputs = solver_pt.simulate(ER_t, UR_t, params)
    if not torch.sum(params['phi']):
        torch.save(image_pixels, f'../.log_{init_dt_string}/images/{image_number}_{epoch_number}.pt')
        print(f"Binarization: {np.real(binarization_loss.cpu().detach().numpy())}")
    field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0] #TODO: understand why we're taking the 4 in the answer even in the working solution
    focal_plane = solver_pt.propagate(params['input'] * field, params['propagator'], params['upsample'])
    # print(f"this shape: {focal_plane.shape}")
    # print(focal_plane[0, :, 0])
    # import time
    # time.sleep(500)
    reflected_focal_plane = torch.flip(focal_plane, [1])
    symmetric_focal_plane = focal_plane + reflected_focal_plane
    p = torch.sum(torch.abs(symmetric_focal_plane), dim = (-1, -2)) # Deleting gradients? Find gradients with track=True and backpropagate throughout network
    with open(f'../log_{init_dt_string}/loss/wavelength_loss.txt', 'a+') as f:
        f.write(str(p.cpu().detach().numpy()))
        f.write('\n')
    return torch.tensordot(determinator.type(p.type()), p, dims = 1) # Need a binarization coefficient # Negate this when starting a new run
p['loss_function'] = loss_function
p['wavelengths'] = wavelengths
k, loss, params, focal_plane = solver_metasurface_pt.optimize_device(p)
print(loss)
