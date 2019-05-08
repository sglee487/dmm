import os
import numpy as np
from utils.misc import loadHDF5
assert os.path.exists('chkpt-ipython/'),'Run the notebook DMM-Setup.ipynb first'

opt_params = np.load('./chkpt-ipython/DMM_lr-0_0008-dh-40-ds-2-nl-relu-bs-200-ep-40-rs-80-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid-EP30-optParams.npz')
for k in opt_params:
    print k, opt_params[k].shape

opt_params = np.load('./chkpt-ipython/DMM_lr-0_0008-dh-40-ds-2-nl-relu-bs-200-ep-40-rs-80-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid-EP30-params.npz')
for k in opt_params:
    print k, opt_params[k].shape


import glob, os, sys, time
sys.path.append('../')
from utils.misc import getConfigFile, readPickle, displayTime
start_time = time.time()
from   model_th.dmm import DMM
import model_th.learning as DMM_learn
import model_th.evaluate as DMM_evaluate
displayTime('importing DMM',start_time, time.time())

#This is the prefix we will use
DIR    = './chkpt-ipython/'
prefix = 'DMM_lr-0_0008-dh-40-ds-2-nl-relu-bs-200-ep-40-rs-80-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid'
pfile  =  os.path.join(DIR,prefix+'-config.pkl')
print 'Hyperparameters in: ',pfile, 'Found: ',os.path.exists(pfile)

#The hyperparameters are saved in a pickle file - lets load them here
params = readPickle(pfile, quiet=True)[0]


#Reload the model at Epoch 30
EP     = '-EP30'
#File containing model paramters
reloadFile  =  os.path.join(DIR,prefix+EP+'-params.npz')
print 'Model parameters in: ',reloadFile
#Don't load the training functions for the model since its time consuming
params['validate_only'] = True
dmm_reloaded  = DMM(params, paramFile = pfile, reloadFile = reloadFile)

print 'Model Reloaded: ',type(dmm_reloaded)