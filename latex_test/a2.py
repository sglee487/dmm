# https://github.com/spyder-ide/spyder/issues/3606

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from matplotlib.pyplot import rc # this is the matplotlib suggestion
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('text', usetex=True)

mat = np.random.rand(5,5)

fig = plt.figure()
gsall = gs.GridSpec(6, 1)
ax1 = fig.add_subplot(gsall[:])
source1 = ax1.contourf(mat,extend='both',cmap=mpl.cm.viridis)
plt.title('Random field \n test')
plt.colorbar(source1)
plt.show()