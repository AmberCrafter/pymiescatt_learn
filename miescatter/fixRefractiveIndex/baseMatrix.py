import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import weight_map
path=__file__.split("\\")
os.chdir('\\'.join(path[:-1]))

mshape=100
mREAL=1.495
mIMAG=0.09
parameter='Sca'
radius=5
wmap=weight_map.weightingMap(radius=5)

Qdata=np.load('./data/Qdata_{mREAL}-{mIMAG}.npy'.format(mREAL=mREAL,mIMAG=mIMAG),allow_pickle=True)
Qparameter=dict(
    Ext=0,
    Sca=1,
    Abs=2,
    q=3,
    Pr=4,
    Back=5,
    Ratio=6
)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

'''parameter=['Ext', 'Sca', 'Abs', 'g', 'Pr', 'Back', 'Ratio']'''
nparameter=Qparameter[parameter]
# x=Qdata[:,0].reshape(mshape,mshape)    # wavelength
# y=(Qdata[:,1]/1000).reshape(mshape,mshape)    # diameter
# z=np.array([val[nparameter] for val in Qdata[:,2]]).reshape(mshape,mshape)   # parameter

wavelength=[450,550,700]
diameter=[1000,2500,7500]
matrix=[]
for wl in wavelength:
    for dm in diameter:
        i=find_nearest(Qdata[:,0],wl)[0] # wavelength
        j=int(find_nearest(Qdata[:,1],dm)[0]/mshape) # diameter
        z=np.array([val[nparameter] for val in Qdata[:,2]]).reshape(mshape,mshape)   # parameter
        k=np.nanmean(z[i-radius:i+radius+1,j-radius:j+radius+1])

        matrix.append([wl,dm,k])
print(matrix)



import PyMieScatt as mie
m=1.495+0.02j
wavelength=[450,550,700]
diameter=[1000,2500,7500]
for wl in wavelength:
    for dm in diameter:
        print(wl,dm,mie.MieQ(m,wl,dm)[1])
