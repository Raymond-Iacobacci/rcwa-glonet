import tensorflow as tf
import numpy as np
import itertools
import json
import solver
import rcwa_utils
import tensor_utils
import matplotlib.pyplot as plt
from matplotlib import colors
from tensorflow.keras import layers, activations
# tf.keras.backend.set_floatx('float64') # Remove if there are further errors, this is only to stop us having to cast in gpo-v3.py, that may not even be an issue

def generate_scaled_metasurface(k, params):
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * np.ones(materials_shape) # Why is UR_t not influenced by k..?
    print(k.shape)
    k=tf.clip_by_value(k, clip_value_min = 0, clip_value_max = params['ers'])
    # print(np.sum(k ** 2))
    # print("This is pulled in: ",k.shape)
    # These should reshape k, these two lines
    k=tf.transpose(k, perm=[1,2,0])
    k = k[tf.newaxis, :, :, :, tf.newaxis, tf.newaxis]
    k = tf.tile(k, multiples = (batchSize, 1, 1, 1, Nx, Ny))
    # print(k.shape)
    layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    # ER_substrate = params['ers'] * tf.ones(layer_shape, dtype = tf.float32)
    # ER_t = tf.concat(values = [k, ER_substrate], axis = 3)
    ER_t = k
    # Cast to complex for subsequent calculations.
    ER_t = tf.cast(ER_t, dtype = tf.complex64)
    UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
    UR_t = tf.cast(UR_t, dtype = tf.complex64)

    return ER_t, UR_t


def generate_layered_metasurface(h, params):
    '''
    Generates permittivity/permeability for a multilayer metasurface design,
    based on a height representation of the metasurface.

    Args:
        h: A `tf.Tensor` of shape `(pixelsX, pixelsY)` specifying the
        metasurface height at each unit cell. Each entry in this tensor should
        be a float in [0,params['Nlay']-1].
        
        params: A `dict` containing simulation and optimization settings.

    Returns:
        ER_t: A `tf.Tensor` of shape 
        `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)` specifying the relative
        permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape
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
    UR = params['urd'] * np.ones(materials_shape)
    
    # Limit optimization range.
    h = tf.clip_by_value(h, clip_value_min = 0, clip_value_max = Nlay-1)
    
    # Convert height representation of to stacked representation.
    # print(f"This is h: {h}")
    z = diff_height_to_stacked(h, params)
    
    # Repeat entries in z so that it has the shape
    # (batchSize, pixelsX, pixelsY, 1, Nx, Ny).
    z = z[tf.newaxis, :, :, :, tf.newaxis, tf.newaxis]
    z = tf.tile(z, multiples = (batchSize, 1, 1, 1, Nx, Ny))

    # Build substrate layer and concatenate along the layers dimension.
    layer_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * tf.ones(layer_shape, dtype = tf.float32)
    ER_t = tf.concat(values = [z, ER_substrate], axis = 3)

    # Cast to complex for subsequent calculations.
    ER_t = tf.cast(ER_t, dtype = tf.complex64)
    UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
    UR_t = tf.cast(UR_t, dtype = tf.complex64)
    return ER_t, UR_t


def diff_height_to_stacked(h, params):
    # instead of the height representation, we want 5 sequences of floats that showcase how much they exist at a point
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
        h: A `tf.Tensor` of shape `(pixelsX, pixelsY)` and type float
            containing heights of each pixel.
        
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        z: A `tf.Tensor` of shape `(pixelsX, pixelsY, Nlay-1)` and type float
            containing relative permittivity of the device at each pixel and on
            each non-substrate layer.

    '''
    
    Nlay = params['Nlay']
    z = tf.stack( [diff_threshold(h, thresh=Nlay-1-i,
                                  coeff=params['sigmoid_coeff'],
                                  offset=Nlay-2-i,
                                  output_scaling = [params['eps_min'],params['eps_max']]) for i in range(Nlay-1) ] )
    return tf.transpose(z, perm=[1,2,0])


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
    
    x_new = tf.math.sigmoid(coeff * (x - (offset + (thresh - offset)/2)) )
    x_new = output_scaling[0] + (output_scaling[1] - output_scaling[0]) * x_new
    return x_new


def get_substrate_layer(params):
    '''
    Generates a tensor representing the substrate layer of the device.
    
    Args:
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        ER_substrate: A `tf.Tensor` of shape
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
    ER_substrate = params['ers'] * tf.ones(layer_shape, dtype = tf.float32)
    
    return ER_substrate


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
        init = np.random.rand(params['pixelsX'], params['pixelsY'])
        return init * (params['Nlay'] - 1)
    else:
        return np.ones(shape=(params['pixelsX'], params['pixelsY'])) * initial_height
    
def init_scaled_metasurface(params):
    if params['enable_random_init']:
        return np.random.rand(params['Nlay'], params['pixelsX'], params['pixelsY']) * (params['erd'] - 1) + 1
    else:
        return params['initial_k']


def display_layered_metasurface(ER_t, params):
    '''
    Displays stacked representation of a metasurface.

    Args:
        ER_t: A `tf.Tensor` of shape
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
        img = tf.transpose(tf.squeeze(ER_t[0,:,:,l,:,:]),[0,2,1,3])
        img = tf.math.real(tf.reshape(img, (params['pixelsX']*params['Nx'],params['pixelsY']*params['Ny'])))
        images.append(axes[l].matshow(img, interpolation='nearest'))
        axes[l].get_xaxis().set_visible(False)
        axes[l].get_yaxis().set_visible(False)
        images[l].set_norm(norm)
        
    plt.show()


def evaluate_solution(focal_plane, params):
    '''
    Generates an evaluation score of a metasurface solution which can be used
    to compare a solution to others.
    
    Args:
        focal_plane: A `tf.Tensor` of shape
            `(batchSize, pixelsX * upsample, pixelsY * upsample)` describing 
            electric field intensity on the focal plane.
    
        params: A `dict` containing simulation and optimization settings.
    
    Returns:
        eval_score: Float evaluation score in range [0, inf).

    '''
    
    r = params['focal_spot_radius']
    index = (params['pixelsX'] * params['upsample']) // 2

    eval_score = tf.math.reduce_sum(
        tf.abs(focal_plane[0, index-r:index+r, index-r:index+r]) )

    return float(eval_score.numpy())


def optimize_device(user_params): # Gets called once with wavelengths inbound
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
        h: A `tf.Tensor` of shape `(pixelsX, pixelsY)` and type float
            containing heights of each pixel in the optimized design.

        loss: A 'tf.Tensor' of shape `(N+1)` and type float containing
            containing calculated loss at each optimization iteration.

        params: A `dict` containing simulation and optimization settings.
            The same as the provided user_params, but also contains
            derived parameters calculated by the RCWA solver.

    '''
    
    # Initialize and populate dictionary of solver parameters, based on the
    # dictionary of user-provided parameters.
    params = solver.initialize_params(wavelengths=user_params['wavelengths'],
                                      thetas=user_params['thetas'],
                                      phis=user_params['phis'],
                                      pte=user_params['pte'],
                                      ptm=user_params['ptm'],
                                      pixelsX=user_params['pixelsX'],
                                      pixelsY=user_params['pixelsY'],
                                      erd=user_params['erd'],
                                      ers=user_params['ers'],
                                      PQ=user_params['PQ'],
                                      Lx=user_params['Lx'],
                                      Ly=user_params['Ly'],
                                      L=user_params['L'],
                                      Nx=16,
                                      eps_min=0.0,
                                      eps_max=user_params['erd'])
    
    # Merge with the user-provided parameter dictionary.
    params['N'] = user_params['N']
    params['sigmoid_coeff'] = user_params['sigmoid_coeff']
    params['sigmoid_update'] = user_params['sigmoid_update']
    params['learning_rate'] = user_params['learning_rate']
    # params['focal_spot_radius'] = user_params['focal_spot_radius']
    params['enable_random_init'] = user_params['enable_random_init']
    # params['initial_height'] = user_params['initial_height']
    params['enable_debug'] = user_params['enable_debug']
    params['enable_print'] = user_params['enable_print']
    params['enable_logging'] = user_params['enable_logging']
    params['initial_k'] = user_params['initial_k']
    
    # Get the loss function.
    loss_function = user_params['loss_function']
    
    # This flag is set if the solver encounters an error.
    params['err'] = False
    # Define the free-space propagator and input field distribution
    # for the metasurface.
    params['f'] = user_params['f'] * 1E-9
    params['upsample'] = user_params['upsample']
    params['propagator'] = solver.make_propagator(params, params['f'])
    params['input'] = solver.define_input_fields(params)
    # Change to scaled metasurface
    # TODO change
    h = tf.Variable(
        init_scaled_metasurface(params),#, initial_height=params['initial_height']),
        dtype=tf.float32) # Essentially, this is contained within the model, we need to input the randomness into the initial model somehow--ideally the model gives out values that are just like h
    # Returns array of (Nlay, pixelsX, pixelsY) of floats
    # Define an optimizer.
    # Store losses as a tensor so that it works in graph mode.
    # This initial h is unuzsed in the program
    opt = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    # Define model
    init_dim = 3
    class Generator(tf.keras.Model):
        def __init__(self):
            super(Generator, self).__init__()
            self.gk = tf.random.normal((init_dim, init_dim), mean=0, stddev=4) # Just for the first generation, the next are all from stddev
            # self.FC1 = tf.keras.Sequential([
            #     layers.Dense(25 ** 2, activation='linear', use_bias=True),
            #     layers.LeakyReLU(0.2),
            # ])
            self.FC2 = tf.keras.Sequential([
                layers.Dense(100 ** 2, activation='linear', use_bias=True),
                layers.LeakyReLU(0.2),
            ])
            self.conv1 = layers.Conv2DTranspose(16, kernel_size=6, padding='same', strides=1, use_bias=True)
            self.bn = layers.BatchNormalization(momentum=False)
            self.conv2 = layers.Conv2DTranspose(16, kernel_size=4, strides=1, use_bias=True)
        def call(self, inputs):
            net = tf.reshape(inputs, [1, -1])
            # net = self.FC1(net)
            net = self.FC2(net)
            net = tf.reshape(net, [1, 100, 100, 1]) # Channels last, only one channel
            # net = self.conv1(net)
            net = self.bn(net)
            # print(net.shape)
            # net = self.conv2(net)
            # net = self.bn(net)
            net = tf.reduce_mean(net, axis = [0, -1])
            net = (activations.tanh(net * params['sigmoid_coeff']) * 0.5 + 0.5) * (params['erd'] - 1) + 1
            # net = net[: -1, :],  # Drop the last layer for purposes of substrate embedding
            # net = tf.reshape(net, (99, 100, 1))

            return tf.expand_dims(net, axis = -1)
        # We also need to write a generation function each time, it does start from each training iteration...?
    generator = Generator()
    NUM_TRIALS = 1
    # generator.train()
    # Begin optimization.
    if params['enable_print']: print('Optimizing... ', end="")
    N = user_params['N']
    loss = np.zeros(N+1)
    gk = tf.random.normal((init_dim, init_dim), mean=0, stddev=4, dtype = tf.dtypes.float64)
    gk_array = [tf.random.normal((init_dim, init_dim), mean=0, stddev=4, dtype = tf.dtypes.float64) for i in range(NUM_TRIALS)]
    import pickle
    best_loss = np.inf
    for i in range(N):
        if params['enable_print']: print(str(i) + ', ', end="")
        # Calculate gradients.
        # Add in binarization coefficient
        with tf.GradientTape() as tape:
            l = []
            # l = loss_function(h, params)
            # // Uncomment the next 4 lines and comment the fifth to start a randomization procedure after every step
            # for j in range(NUM_TRIALS):
            #     random_kernel = tf.random.normal(shape = (init_dim, init_dim), stddev = 2.0)
            #     l.append(loss_function(generator(random_kernel), params))
            # l = tf.reduce_mean(l)
            # l = loss_function(generator(gk), params)
            for j in gk_array:
                out = generator(j)
                print("This is the out shape",out.shape)
                # with open(f"../values_2/prior_{NUM_TRIALS}_{i}.txt", 'wb') as f:
                #     pickle.dump(out, f)
                
                l.append(loss_function(out, params))
            l = tf.reduce_mean(l)
            # with open(f'../values_2/prior_{NUM_TRIALS}_loss_history.txt', 'a+') as f:
            #     f.write(str(l))
            #     f.write('\n')
            print(l)
            grads = tape.gradient(l, generator.trainable_weights) # This is wehre the casting issue pops up
            # grads = tape.gradient(l, [h])
        # Apply gradients to variables.
        opt.apply_gradients(zip(grads, generator.trainable_weights))
        # opt.apply_gradients(zip(grads, [h]))
        # Keep track of iteration loss.
        loss[i] = l

        # Save model to file
        checkpoint_path = f"training_2/cp-{i}.ckpt"
        if loss[i] < best_loss:
            best_loss = loss[i]
            generator.save_weights(checkpoint_path)

        # Anneal sigmoid coefficient.
        params['sigmoid_coeff'] += (params['sigmoid_update'] / N)
    
    if params['enable_print']: print('Done.')
        
    # Round off to a final, admissable, solution.
    # Do a final range clip.
    # TODO change
    h = generator(gk)
    h = tf.clip_by_value(h, clip_value_min=1, clip_value_max=params['erd']) # alternatively a single final prediction with the generator
    # Do a final range clip.
    # k = tf.clip_by_value(k, clip_value_min=0, clip_value_max=params['Nlay']-1)
    
    # Round heights to nearest integer.
    # k = tf.math.round(k) # NOTE: also something flipping with k
    
    # Get final loss.
    loss[N] = loss_function(h,params)

    
    # Get scattering pattern of final solution.
    # TODO This also should be generate_scaled_metasurface
    ER_t, UR_t = generate_scaled_metasurface(h, params)
    # ER_t, UR_t = generate_layered_metasurface(k, params)
    outputs = solver.simulate(ER_t, UR_t, params)
    field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
    focal_plane = solver.propagate(params['input'] * field, params['propagator'], params['upsample'])
    
    return h, loss, params, focal_plane


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
            Each dict contains a list of the used hyperparameters.
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
        
        # Otherwise, proceed.
        if user_params['enable_print']: print('\nTrying hyperparameters: ' + str(list(hp_names)))
        if user_params['enable_print']: print(hyperparams)

        # Update parameter list dict with selected parameters.
        for i, name in enumerate(hp_names):
            user_params[name] = hyperparams[i]

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
        
        # Log result.
        if params['enable_logging']:
            hyperparameter_string = '-'.join([h + str(v) for (h,v) in zip(hp_names,hyperparams)])
            log_result(result, user_params['log_filename_prefix'] + hyperparameter_string + user_params['log_filename_extension'])
    
    return results

def log_result(result, log_filename):
    
    # Open log file in write mode.
    with open(log_filename, 'w', encoding="utf-8") as f:
        
        # Get json representation of results dict and write to log file.
        json.dump(make_result_loggable(result), f)

        
def make_result_loggable(result):
    
    # Modify result dict to only include necessary elements
    # and ensure that they are all json serializable.
    loggable_result = {'hyperparameter_names': result['hyperparameter_names'],
                       'hyperparameters': result['hyperparameters'],
                       'h': result['h'].numpy().tolist(),
                       'loss': result['loss'].tolist(),
                       'focal_plane': tf.cast(result['focal_plane'], tf.float32).numpy().tolist(),
                       'eval_score': result['eval_score'] }
    
    return loggable_result


def load_result(log_filename):
     
    # Open log file in read mode.
    with open(log_filename, 'r', encoding="utf-8") as f:

        # Read json representation of results from log file.
        result = json.load(f)
        result['h'] = tf.convert_to_tensor(result['h'], dtype=tf.float32)
        result['loss'] = np.array(result['loss'])
        result['focal_plane'] = tf.convert_to_tensor(result['focal_plane'], dtype=tf.float32)
        return result
