'''
solver_metasurface_pt.py

Functions implementing optimization algorithm for COPILOT metalens devices,
using PyTorch differentiable implementation of RCWA.

The important user-facing functions here are -

optimize_device: use to optimize a single device for some given parameters.

hyperparameter_gridsearch: use to optimize many devices for a given grid of
   algorithm hyperparameters.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.lines as lines
from matplotlib import colors
import itertools
import json
# import gc
import net

import solver_pt
import rcwa_utils_pt

def generate_metasurface(k, params):
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR_t = params['urd'] * torch.ones(materials_shape)
    UR_t = UR_t.type(torch.complex64)
    k = torch.clamp(k, min = 1, max = params['erd'])
    k = k[None, :, :, :, None, None]
    ER_t = torch.tile(k, (batchSize, 1, 1, 1, Nx, Ny))
    ER_t = ER_t.type(torch.complex64)
    return ER_t, UR_t


def generate_layered_metasurface(h, params):
    '''
    Generates permittivity/permeability for a multilayer metasurface design,
    based on a height representation of the metasurface.

    Args:
        h: A `torch.Tensor` of shape `(pixelsX, pixelsY)` specifying the
        metasurface height at each unit cell. Each entry in this tensor should
        be a float in [0,params['Nlay']-1].
        
        params: A `dict` containing simulation and optimization settings.

    Returns:
        ER_t: A `torch.Tensor` of shape 
        `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)` specifying the relative
        permittivity distribution of the unit cell.

        UR_t: A `torch.Tensor` of shape
        `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)` specifying the relative
        permeability distribution of the unit cell.

    '''

    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']

    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR_t = params['urd'] * torch.ones(materials_shape)
    
    # Limit optimization range.
    h = torch.clamp(h, min = 0, max = Nlay-1) 
    
    # Convert height representation of to stacked representation.
    z = diff_height_to_stacked(h, params)
    
    # Repeat entries in z so that it has the shape
    # (batchSize, pixelsX, pixelsY, 1, Nx, Ny).
    z = z[None, :, :, :, None, None]
    z = torch.tile(z, (batchSize, 1, 1, 1, Nx, Ny))

    # Build substrate layer and concatenate along the layers dimension.
    layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(layer_shape, dtype = torch.float32)
    ER_t = torch.cat([z, ER_substrate], dim = 3)

    # Cast to complex for subsequent calculations.
    ER_t = ER_t.type(torch.complex64)
    UR_t = UR_t.type(torch.complex64)
    return ER_t, UR_t


def diff_height_to_stacked(h, params):
    '''
    Performs a differentiable transformation from the continuous height
    representation of a metasurface pixel to a continous stacked 
    representation. This is achieved via differential thresholding based on a
    sigmoid function.

    The HEIGHT REPRESENTATION of a metasurface is a tensor containing floats
    representing the material height at each metasurface pixel.

    The STACKED REPRESENTAION of a metasurface is a 3D tensor specifying a
    float relative permittivity of the device at each pixel and on each
    layer. This does not include the substrate layer.

    As params['sigmoid_coeff'] is increased, the thresholding becomes more
    strict, until eventually the metasurface is restricted to be 'admissable' -
    that is, each position in the stacked representation may take on only one
    of the two allowed values.
    
    Args:
        h: A `torch.Tensor` of shape `(pixelsX, pixelsY)` and type float
            containing heights of each pixel.
        
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        z: A `torch.Tensor` of shape `(pixelsX, pixelsY, Nlay-1)` and type float
            containing relative permittivity of the device at each pixel and on
            each non-substrate layer.

    '''
    
    Nlay = params['Nlay']
    z = torch.stack( [diff_threshold(h, thresh=Nlay-1-i,
                                  coeff=params['sigmoid_coeff'],
                                  offset=Nlay-2-i,
                                  output_scaling = [params['eps_min'],params['eps_max']]) for i in range(Nlay-1) ] )
    return torch.permute(z, [1,2,0])


def diff_threshold(x, coeff=1, thresh=1, offset=0, output_scaling=[0,1]):
    '''
    Performs a differentiable thresholding operation on the input, based on a
    sigmoid funciton.
    
    Args:
        x: Float input to be thresholded. Can be a single number or a tensor
            of any dimension.
        
        coeff: Float coefficient determining steepness of thresholding.
        
        thresh: Float thresholding cutoff, i.e. where in x the step should
            occur.
        
        offset: Float minimum value assumed to occur in x. This value is
            subtracted from x first before the operation is applied, such that
            the sigmoid cutoff occurs halfway between in_offset and thresh.
            
        output_scaling: Float list of length 2 specifying limits to which
            output should be renormalized.
        
        Both offsets should be < thresh, and coeff should be >= 0.
    
    Returns:
        x: Thresholded input.

    '''
    
    x_new = torch.sigmoid(coeff * (x - (offset + (thresh - offset)/2)) )
    x_new = output_scaling[0] + (output_scaling[1] - output_scaling[0]) * x_new
    return x_new


def get_substrate_layer(params):
    '''
    Generates a tensor representing the substrate layer of the device.
    
    Args:
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        ER_substrate: A `torch.Tensor` of shape
            `(batchSize, pixelsX, pixelsY, 1, Nx, Ny)' specifying the relative
            permittivity distribution of the unit cell in the substrate layer.

    '''
    
    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    
    # Build and return substrate layer.
    layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(layer_shape, dtype = torch.float32)
    
    return ER_substrate

def init_metasurface(params, initial_heights = None):
    optimization_shape = (100, 100)
    if initial_heights is None:
        # print("OUT") #TODO: deal with initial_heights vs initial_k confusion
        assert params['enable_random_init']
        return torch.rand(25, dtype = torch.float32) * 2 - 1
    else:
        assert [int(x) for x in initial_heights.size()] == optimization_shape and not params['enable_random_init']
        return initial_heights.float()


def init_layered_metasurface(params, initial_height=0):
    '''
    Generates an initial guess for optimization of a multilayer metasurface.

    The provided guess is a height representation of the metasurface.

    If params['random_init'] == 0, returns a initial guess with zero height
    at all pixels. Otherwise, returns an initial guess with all pixels
    at the height specified by initial_height.
    
    Args:
        params: A `dict` containing simulation and optimization settings.

        initial_height: A float in the range [0, Nlay-1] specifying the an
            initial guess for the height at each pixel.
        
    Returns:
        init: A `np.array of shape `(pixelsX, pixelsY)` specifying an initial
        guess for the height of the metasurface at each pixel.
    
    '''
    
    if params['enable_random_init']:
        init = torch.rand((params['pixelsX'], params['pixelsY']))
        return init * (params['Nlay'] - 1)
    else:
        return torch.ones((params['pixelsX'], params['pixelsY'])) * initial_height


def display_layered_metasurface(ER_t, params):
    '''
    Displays stacked representation of a metasurface.

    Args:
        ER_t: A `torch.Tensor` of shape
        `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)` specifying the relative
        permittivity distribution of the unit cell.
        
        params: A `dict` containing simulation and optimization settings.

    Returns: None

    '''
    
    # Display the permittiviy profile.
    norm = colors.Normalize(vmin=params['eps_min'], vmax=params['eps_max'])
    images=[]
    fig, axes = plt.subplots(params['Nlay'], 1, figsize=(12,12))
    for l in range(params['Nlay']):
        img = torch.permute(torch.squeeze(ER_t[0,:,:,l,:,:]),(0,2,1,3))
        img = torch.real(torch.reshape(img, (params['pixelsX']*params['Nx'],params['pixelsY']*params['Ny'])))
        images.append(axes[l].matshow(img.detach().cpu().numpy(), interpolation='nearest'))
        axes[l].get_xaxis().set_visible(False)
        axes[l].get_yaxis().set_visible(False)
        images[l].set_norm(norm)
        
    plt.show()
    

def display_multiple_metasurfaces(h_list, params):
    '''
    Displays stacked representations of a list of devices.

    Args:
        h_list: A list of `torch.Tensor`s of shape `(pixelsX, pixelsY)` specifying the height representation
            of multiple metasurfaces.
        
        params: A `dict` containing simulation and optimization settings.

    Returns: None

    '''
    
    # Set up the plot.
    norm = colors.Normalize(vmin=params['eps_min'], vmax=params['eps_max'])
    images=[]
    fig, axes = plt.subplots(params['Nlay'], len(h_list), figsize=(12,12))
    
    # For each device...
    for d in range(len(h_list)):
    
        # Get stacked representation of the surface.
        ER_t, UR_t = generate_layered_metasurface(torch.Tensor(h_list[d]), params)

        # Then for each layer...
        for l in range(params['Nlay']):
            img = torch.permute(torch.squeeze(ER_t[0,:,:,l,:,:]),(0,2,1,3))
            img = torch.real(torch.reshape(img, (params['pixelsX']*params['Nx'],params['pixelsY']*params['Ny'])))
            images.append(axes[l,d].matshow(img.detach().cpu().numpy(), interpolation='nearest'))
            axes[l,d].get_xaxis().set_visible(False)
            axes[l,d].get_yaxis().set_visible(False)
            images[-1].set_norm(norm)
        
        # Add line dividing devices.
        #if d > 0:
        #    div_x = d/(len(h_list))
        #    fig.lines.append(plt.Line2D([div_x, div_x], [0, 1], transform=fig.transFigure, color='black'))
        
    plt.show()


def evaluate_solution(focal_plane, params):
    '''
    Generates an evaluation score of a metasurface solution which can be used
    to compare a solution to others.
    
    Args:
        focal_plane: A `torch.Tensor` of shape
            `(batchSize, pixelsX * upsample, pixelsY * upsample)` describing 
            electric field intensity on the focal plane.
    
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        eval_score: Float evaluation score in range [0, inf).

    '''
    
    r = params['focal_spot_radius']
    index = (params['pixelsX'] * params['upsample']) // 2

    eval_score = torch.sum(torch.abs(focal_plane[0, index-r:index+r, index-r:index+r]) )

    return float(eval_score)

def optimize_device(user_params):
    params = solver_pt.initialize_params(wavelengths=user_params['wavelengths'],
                                  thetas=user_params['thetas'],
                                  phis=user_params['phis'],
                                  pte=user_params['pte'],
                                  ptm=user_params['ptm'],
                                  pixelsX=user_params['pixelsX'],
                                  pixelsY=user_params['pixelsY'],
                                  erd=user_params['erd'],
                                  PQ=user_params['PQ'],
                                  Lx=user_params['Lx'],
                                  Ly=user_params['Ly'],
                                  L=user_params['L'],
                                  Nx=16,
                                  eps_min=1.0,
                                  eps_max=user_params['erd'])
    params['N'] = user_params['N']
    params['sigmoid_coeff'] = user_params['sigmoid_coeff']
    params['sigmoid_update'] = user_params['sigmoid_update']
    params['learning_rate'] = user_params['learning_rate']
    params['enable_random_init'] = user_params['enable_random_init']
    params['enable_debug'] = user_params['enable_debug']
    params['enable_print'] = user_params['enable_print']
    params['enable_logging'] = user_params['enable_logging']
    params['initial_k'] = user_params['initial_k']
    params['err'] = False
    params['f'] = user_params['f'] * 1E-9
    params['upsample'] = user_params['upsample']
    params['propagator'] = solver_pt.make_propagator(params, params['f'])
    params['input'] = solver_pt.define_input_fields(params)
    params['loss_function'] = user_params['loss_function']
    params['restore_from'] = user_params['restore_from']
    n_images = 20
    n_images = 1
    angles = torch.linspace(-30, 0, steps = 7)
    k_array = [torch.autograd.Variable(init_metasurface(params), requires_grad = True) for i in range(n_images)]
    generator = net.Generator()
    start_epoch = 0
    if params['restore_from']:
        print(f'''Loading from prior state: {params['restore_from']}''')
        generator.load_state_dict(torch.load(f'''../.log_{params['restore_from']}/models/{params['restore_from']}.pt''')) # Need to input loading for images, but logging has not finished yet
        k_array = [torch.load(f'''../.log_{params['restore_from']}/image_backup/{image_num}.pt''') for image_num in range(n_images)]
        start_epoch = 0
        params['sigmoid_coeff'] += start_epoch * params['sigmoid_update'] / params['N']
    opt = torch.optim.Adam(generator.parameters(), lr = params['learning_rate']) # Not tracking images yet, will start soon
    # TODO: check the learning rate, likely decrease since it seems to move around?
    N = params['N']
    loss = []
    init_dt_string = user_params['time']
    from os import system
    system(f'mkdir ../.log_{init_dt_string}/')
    system(f'mkdir ../.log_{init_dt_string}/loss')
    system(f'mkdir ../.log_{init_dt_string}/storage')
    system(f'mkdir ../.log_{init_dt_string}/values')
    system(f'mkdir ../.log_{init_dt_string}/storage_spread')
    system(f'mkdir ../.log_{init_dt_string}/image_backup')
    system(f'mkdir ../.log_{init_dt_string}/images')
    system(f'mkdir ../.log_{init_dt_string}/models')
    bias = 5
    n_ref = 2
    big_refresh = 15
    big_refresh_size = 100
    little_refresh = 3
    little_refresh_variance = 0.05
    l_val = torch.tensor(0.)
    for epoch in range(start_epoch, N):
        if params['enable_print']: print(str(epoch) + ', ', end = '')
        opt.zero_grad()
        epoch_loss = []
        for image_num in range(n_images):
            torch.save(k_array[image_num], f'../.log_{init_dt_string}/image_backup/{image_num}.pt')
            values = torch.clamp(generator(k_array[image_num], params['sigmoid_coeff']) * 0.5 * 1.05 + 0.5, min=0, max=1)
            image_loss = []
            for angle in angles:
                params['theta'] = torch.zeros(params['theta'].shape)
                params['theta'] += angle
                print(f"Epoch: {epoch}, iteration: {image_num}, angle: {angle}")
                l = params['loss_function'](values * (params['erd'] - 1.0) + 1, params, image_number = image_num, epoch_number = epoch)
                print(f'this is the loss: {l}')
                if angle == 0:
                    with open(f'../.log_{init_dt_string}/values/values_{image_num}.txt', 'a+') as f:
                        f.write(str(values))
                        f.write('\n')
                torch.save(l, f'../.log_{init_dt_string}/storage/storage_{image_num}_{angle}.pt') # This is still good practice because of the relative length of time of the simulation process
                image_loss.append(l.cpu().detach().numpy())
                with open(f'../.log_{init_dt_string}/loss/angle_loss.txt', 'a+') as f:
                    f.write(str(epoch))
                    f.write(', ')
                    f.write(str(image_num))
                    f.write(', ')
                    f.write(str(angle.cpu().detach().numpy()))
                    f.write(' | ')
                    f.write(str(l.cpu().detach().numpy()))
                    f.write('\n')
                del l
            epoch_loss.append(np.mean(image_loss))
            print(f"This is the loss for the whole image: {np.mean(image_loss)}")
            with open(f'../.log_{init_dt_string}/loss/image_loss.txt', 'a+') as f:
                f.write(str(epoch))
                f.write(', ')
                f.write(str(image_num))
                f.write(' | ')
                f.write(str(np.mean(image_loss)))
                f.write('\n')

        if epoch % little_refresh == 0 and epoch:
            lottery_values = (np.array(epoch_loss) - np.min(epoch_loss)) / (np.max(epoch_loss) - np.min(epoch_loss))
            winning_lottery_values = np.e ** (-bias * np.array(lottery_values)) # Perform the transformation
            winning_lottery_values /= np.sum(winning_lottery_values) # Normalize the result
            winning_lottery_indices = range(len(lottery_values))
            winners_indices = np.random.choice(winning_lottery_indices, size = n_ref, replace = False, p = winning_lottery_values)

            losing_lottery_values = np.e ** (bias * np.array(lottery_values))
            losing_lottery_indices = [i for i in range(len(lottery_values)) if i not in winners_indices]
            losing_lottery_values /= np.sum(losing_lottery_values[losing_lottery_indices])
            losers_indices = np.random.choice(losing_lottery_indices, size = n_ref, replace = False, p = losing_lottery_values[losing_lottery_indices])

            for lottery_i in range(n_ref):
                k_array[losers_indices[lottery_i]] = k_array[winners_indices[lottery_i]] + torch.randn(size = k_array[winners_indices[lottery_i]].shape) * np.sqrt(little_refresh_variance)
                k_array[losers_indices[lottery_i]] = torch.clamp(k_array[losers_indices[lottery_i]], min = -1, max = 1)
            print(f"Eliminating {losers_indices}, propagating {winners_indices}")

        l_val = torch.tensor(0.)
        for image_num in range(n_images):
            for angle in angles:
                loss_on_ram = torch.load(f'../.log_{init_dt_string}/storage/storage_{image_num}_{angle}.pt')
                l_val.add_(loss_on_ram)
                del loss_on_ram
                system(f'rm ../.log_{init_dt_string}/storage/storage_{image_num}_{angle}.pt')
        l_result = l_val / (n_images * len(angles))
        l_result.backward()
        opt.step()
        params['sigmoid_coeff'] += (params['sigmoid_update'] / params['N'])
        loss.append(np.mean(epoch_loss))
        print(f"This is the loss for the whole epoch: {np.mean(epoch_loss)}")
        with open(f'../.log_{init_dt_string}/loss/epoch_loss.txt', 'a+') as f:
            f.write(str(epoch))
            f.write(', ')
            f.write(str(np.mean(epoch_loss)))
            f.write('\n')
        torch.save(generator.state_dict(), f'../.log_{init_dt_string}/models/{init_dt_string}.pt')
        if epoch % big_refresh == 0 and epoch and epoch != N:
            explore_k_array = [torch.autograd.Variable(init_metasurface(params), requires_grad = True) for i in range(big_refresh_size)]
            new_image_loss = []
            for image_num in range(big_refresh_size):
                new_angle_loss = []
                for angle in angles:
                    params['phi'] = torch.zeros(params['phi'].shape)
                    params['phi'] += angle
                    print(f"Epoch: {epoch}, iteration: {image_num}, angle: {angle}")
                    values = torch.clamp(generator(explore_k_array[image_num], params['sigmoid_coeff']) * 0.5 * 1.05 + 0.5, min=0, max=1)
                    l = params['loss_function'](values * (params['erd'] - 1.0) + 1, params)
                    torch.save(l, f'../.log_{init_dt_string}/storage_spread/storage_{image_num}_{angle}.pt')
                    new_angle_loss.append(l.cpu().detach().numpy())
                    del l
                new_image_loss.append(np.mean(new_angle_loss))
            expand_indices = np.argsort(new_image_loss)[:n_images]
            k_array_indices = np.argsort(epoch_loss)
            print(expand_indices)
            for i in range(len(expand_indices)):
                k_array[k_array_indices[n_images - 1 - i]] = explore_k_array[expand_indices[i]]
            print(f"This is the result from the random expansion: {epoch_loss}")
    system(f'rmdir ../.log_{init_dt_string}/storage')
    print(f"This is the loss over time: {loss}") # TODO: deal with remaining generator

def _optimize_device(user_params):
    '''
    Produces an optimized layered metasurface design for some given device and
    optimization parameters.

    Args:
        user_params: A `dict` containing simulation and optimization settings.
            As opposed to dicts named simply 'params' elsewhere in this code,
            'user_params' contains only parameters which are able to be
            directly configured by a user, and not those derived parameters
            calculated by the RCWA solver.

    Returns:
        h: A `torch.Tensor` of shape `(pixelsX, pixelsY)` and type float
            containing heights of each pixel in the optimized design.

        loss: A 'torch.Tensor' of shape `(N+1)` and type float containing
            containing calculated loss at each optimization iteration.

        params: A `dict` containing simulation and optimization settings.
            The same as the provided user_params, but also contains
            derived parameters calculated by the RCWA solver.
            
        focal_plane: A `torch.Tensor` of shape `(batchSize, params['upsample'] * pixelsX,
            params['upsample'] * pixelsY)` and dtype `torch.complex64` specifying the 
            the electric fields at the output plane.
    '''
    pass


def hyperparameter_gridsearch(user_params):
    '''
    Runs a grid search for good hyperparameters for layered metasurface
    optimization.

    Args:
        user_params: A `dict` containing simulation and optimization settings.
            As opposed to dicts named simply 'params' elsewhere in this code,
            'user_params' contains only parameters which are able to be
            directly configured by a user.

            Specifically, the entry param_grid is used to configure ranges for
            the grid search.

    Returns:
        results: A list of `dict`s, each of which corresponds to one
            run of metasurface optimization for some selected hyperparameters.
            Each dict contains a list of the used hyperparameters, as well as
            the height representation of the reulting metasurface, evaluation
            score assigned to that metasurface, the loss curve for the
            optimization run, and the focal plane scatter pattern produced
            by the optimized device.
    '''
    
    # Allocate list of results.
    # Each entry is a dictionary containing the list of hyperparameters used,
    # height representation of the resulting metasurface, focal plane intensity
    # pattern produced by the metasurface, evaluation score of the metasurface,
    # and list of optimization losses for that run.
    results = []
    
    # Get dimensions of grid.
    hp_grid = user_params['param_grid'].values()
    hp_names = user_params['param_grid'].keys()
    
    if user_params['enable_print']: print('Beginning hyperparameter grid search...')
    
    # Iterate over the grid.
    for hyperparams in itertools.product(*hp_grid):
        
        if user_params['enable_print']: print('\nTrying hyperparameters: ' + str(list(hp_names)))
        if user_params['enable_print']: print(hyperparams)

        # Update parameter list dict with selected parameters.
        for i, name in enumerate(hp_names):
            user_params[name] = hyperparams[i]
            
        # Update log file name.
        hyperparameter_string = '-'.join([h + str(v) for (h,v) in zip(hp_names,hyperparams)])
        user_params['parameter_string'] = hyperparameter_string

        # Run optimization with selected parameters.
        h, loss, params, focal_plane = optimize_device(user_params)
        
        # Get the evaluation score of the resulting solution.
        eval_score = evaluate_solution(focal_plane, params)
        
        # Save result.
        result = {'hyperparameter_names': list(hp_names),
            'hyperparameters': hyperparams,
            'h': h,
            'loss': loss,
            'focal_plane': focal_plane,
            'eval_score': eval_score,
            'params': params }
        results.append(result)

    return results

                           
def log_result(result, log_filename):
    '''
    Writes the result of a single optimization run to an output file.
    
    Args:
        result: A dict which corresponds to one run of metasurface optimization for 
            some selected hyperparameters. It should contain the keys:
            
            hyperparameter_names: A list of string names of the hyperparameters being optimized.
            hyperparameters: A list of hyperparameter values used for this run, corresponding to
                those in hyperparameter_names.
               
            h: A list of shape `(pixelsX, pixelsY)` and type float
                containing heights of each pixel in the optimized design.

            loss: A list of shape `(N+1)` and type float containing
                containing calculated loss at each optimization iteration.
                
            focal_plane: A list of shape `(batchSize, params['upsample'] * pixelsX,
                params['upsample'] * pixelsY)` and dtype float specifying the 
                the real part of electric fields at the output plane.
            
            eval_score: Float evaluation score of the solution in range [0, inf).

            params: A `dict` containing simulation and optimization settings.
        
        log_filename: A string specifying the relative path of the file to write result to.
            File is created if it does not exist and overwritten if it does.

    Returns:
        None
    '''
    
    # Open log file in write mode.
    with open(log_filename, 'w', encoding="utf-8") as f:
        
        # Get json representation of results dict and write to log file.
        json.dump(make_result_loggable(result), f)

        
def make_result_loggable(result):
    '''
    Prepares the results dictionary of an optimization run for writing to an output file.
    
    Args:
        result: A dict which corresponds to one run of metasurface optimization for 
            some selected hyperparameters, structured as in log_result, except that
            h, loss, and focal_plane may be 'torch.Tensor's.

    Returns:
        loggable_result: A dict with the same contents as result but with element
            types converted such that it can be passed to log_result.
    '''
    
    # Modify result dict to only include necessary elements
    # and ensure that they are all json serializable.
    loggable_result = {'hyperparameter_names': result['hyperparameter_names'],
                       'hyperparameters': result['hyperparameters'],
                       'h': result['h'].detach().cpu().numpy().tolist(),
                       'loss': result['loss'].tolist(),
                       'focal_plane': result['focal_plane'].type(torch.float32).detach().cpu().numpy().tolist(),
                       'eval_score': result['eval_score'] }
    
    return loggable_result


def load_result(log_filename):
    '''
    Read results of an optimization run from a file.
    
    Args:
        log_filename: A string specifying the relative path to the output file to be read.
    
    Returns:
        result: A dict containing all information about the recorded optimization run,
            structured as in log_result.
    '''
    
    # Open log file in read mode.
    with open(log_filename, 'r', encoding="utf-8") as f:

        # Read json representation of results from log file.
        result = json.load(f)
        result['h'] = torch.tensor(result['h'], dtype=torch.float32)
        result['loss'] = np.array(result['loss'])
        result['focal_plane'] = torch.tensor(result['focal_plane'], dtype=torch.float32)
        return result
