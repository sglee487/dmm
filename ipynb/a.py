# https://github.com/spyder-ide/spyder/issues/3606
# 여기서 실행하면 파이썬 파일 경로로 에러가 뜬다. 경로가 너무 길어서 자른댄다..
# https://eehoeskrap.tistory.com/138
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