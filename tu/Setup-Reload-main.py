#General Purpose Imports
import numpy as np
import glob, os, sys, time
sys.path.append('../')
from utils.misc import getConfigFile, readPickle, displayTime


#Import load function to load synthetic data
from dmm_data.load import load
dataset = load('synthetic')
print type(dataset), dataset.keys()

print 'Dimensionality of the observations: ', dataset['dim_observations']
print 'Data type of features:', dataset['data_type']
for dtype in ['train','valid','test']:
    print 'dtype: ',dtype, ' type(dataset[dtype]): ',type(dataset[dtype])
    print [(k,type(dataset[dtype][k]), dataset[dtype][k].shape) for k in dataset[dtype]]
    print '--------\n'


start_time = time.time()
from   model_th.dmm import DMM
import model_th.learning as DMM_learn
import model_th.evaluate as DMM_evaluate
displayTime('importing DMM',start_time, time.time())


params = readPickle('../default.pkl')[0]
for k in params:
    print k, '\t',params[k]
params['data_type'] = dataset['data_type']
params['dim_observations'] = dataset['dim_observations']


#The dataset is small, lets change some of the default parameters and the unique ID
params['dim_stochastic'] = 2
params['dim_hidden']     = 40
params['rnn_size']       = 80
params['epochs']         = 40
params['batch_size']     = 200
params['unique_id'] = params['unique_id'].replace('ds-100','ds-2').replace('dh-200','dh-40').replace('rs-600','rs-80')
params['unique_id'] = params['unique_id'].replace('ep-2000','ep-40').replace('bs-20','bs-200')

#Create a temporary directory to save checkpoints
params['savedir']   = params['savedir']+'-ipython/'
os.system('mkdir -p '+params['savedir'])

#Specify the file where `params` corresponding for this choice of model and data will be saved
pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'

print 'Checkpoint prefix: ', pfile
dmm  = DMM(params, paramFile = pfile)



#savef specifies the prefix for the checkpoints - we'll use the same save directory as before
savef    = os.path.join(params['savedir'],params['unique_id'])
savedata = DMM_learn.learn(dmm, dataset['train'], epoch_start =0 ,
                                epoch_end = params['epochs'],
                                batch_size = 200,
                                savefreq   = params['savefreq'],
                                savefile   = savef,
                                dataset_eval=dataset['valid'],
                                shuffle    = True )



# -----------------------------------------------------
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

#--------------------------------------------------------------------------
