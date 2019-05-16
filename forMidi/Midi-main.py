# https://github.com/sglee487/dmm

import os
import numpy as np
from utils.misc import loadHDF5, createIfAbsent
import glob, os, sys, time
sys.path.append('../')
from utils.misc import getConfigFile, readPickle, displayTime, downloadData, loadHDF5, getLowestError
start_time = time.time()
from   model_th.dmm import DMM
import model_th.learning as DMM_learn
import model_th.evaluate as DMM_evaluate
displayTime('importing DMM',start_time, time.time())

# =====================================

if not os.path.exists('./midi/'):
    download_dir = './'
    files        = {}
    files['midi.zip'] = 'http://www.iro.umontreal.ca/~lisa/deep/midi.zip'
    downloadData(download_dir,files)
    os.system('unzip midi.zip;rm -rf midi.zip')
else:
    print ('./midi found')

# =============================================

from midi.utils import midiread, midiwrite
# assert os.system('timidity -h')==0,'Install Timidity from http://timidity.sourceforge.net/'

# ========================================

# #change the dataset to one of ['jsb','nottingham','musedata','piano']
# DATASET= 'jsb'
# DATASET= 'ipython'
DATASET = 'synthetic'
DIR    = '../expt/chkpt-'+DATASET+'/'
# DIR    = './chkpt-'+DATASET+'/'
# assert os.path.exists('../expt/chkpt-'+DATASET+'/'),'Run the shell files in ../expt first'
# prefix = 'DMM_lr-0_0008-dh-200-ds-100-nl-relu-bs-20-ep-2000-rs-600-rd-0_1-infm-R-tl-2-el-2-ar-2000_0-use_p-approx-rc-lstm-DKF-ar'
prefix = 'DMM_lr-0_0008-dh-200-ds-100-nl-relu-bs-20-ep-20-rs-600-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid'
# prefix = 'DMM_lr-0_0008-dh-40-ds-2-nl-relu-bs-200-ep-40-rs-80-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid'
stats  = loadHDF5(os.path.join(DIR,prefix+'-final.h5'))
# stats  = loadHDF5(os.path.join('chkpt-ipython/DMM_lr-0_0008-dh-40-ds-2-nl-relu-bs-200-ep-40-rs-80-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid-EP30-stats.h5'))
epochMin, valMin, idxMin = getLowestError(stats['valid_bound'])
pfile  =  os.path.join(DIR,prefix+'-config.pkl')

params = readPickle(pfile, quiet=True)[0]
print 'Hyperparameters in: ',pfile, 'Found: ',os.path.exists(pfile)
EP     = '-EP'+str(int(epochMin))
reloadFile  =  os.path.join(DIR,prefix+EP+'-params.npz')
print 'Model parameters in: ',reloadFile
#Don't load the training functions for the model since its time consuming
params['validate_only'] = True
dmm_reloaded  = DMM(params, paramFile = pfile, reloadFile = reloadFile)

# forViz/chkpt-ipython/DMM_lr-0_0008-dh-40-ds-2-nl-relu-bs-200-ep-40-rs-80-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid-EP30-stats.h5

# expt/chkpt-synthetic/DMM_lr-0_0008-dh-200-ds-100-nl-relu-bs-20-ep-20-rs-600-rd-0_1-infm-R-tl-2-el-2-ar-2_0-use_p-approx-rc-lstm-uid-final.h5


# =============================================


params, zvec = DMM_evaluate.sample(dmm_reloaded, T= 200)
bin_prob     = params[0]
print 'Samples: ',bin_prob.shape


# ===========================================


#Parameters for music sampling taken from http://deeplearning.net/tutorial/rnnrbm.html
rval = (21, 109)
dt   = 0.3
SAVEDIR = './samples/'
createIfAbsent(SAVEDIR)
print 'Saving wav...'
for idx, sample in enumerate(bin_prob):
    piano_roll = (sample>0.5)*1.
    filename = SAVEDIR+DATASET+'-'+str(idx)+'.mid'
    midiwrite(filename, piano_roll, rval, dt)
    print idx,

print 'Converting...'
print os.system('cd '+SAVEDIR+';timidity -Ow1S '+DATASET+'*.mid;cd ../')
print '\nFiles\n',', '.join(os.listdir(SAVEDIR))

# ================================================

import IPython
IPython.display.Audio('./samples/jsb-0.wav')