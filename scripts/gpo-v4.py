from sys import path
path.append('../src/')
path.append('../rcwa_tf/src/')
import tensorflow as tf, solver, solver_metasurface_pt, numpy as np
from copy import deepcopy
p = {}
p['enable_print'] = True
p['pixelsX'] = 100
p['pixelsY'] = 1
p['N'] = 1000