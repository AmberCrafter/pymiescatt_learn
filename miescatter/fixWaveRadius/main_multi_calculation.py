from multiprocessing import Pool
import PyMieScatt as mie
import numpy as np
import datetime
import os

path=__file__.split('\\')
path='\\'.join(path[:-1])
os.chdir(path)
# matrix shape
mshape=100

# light wavelength
wavelength=550

# refractive index m=n+ik
# n=1.3 .. 2 linear
# k=0 .. 0.3 linear
n=np.linspace(1.3,2,mshape)
k=np.linspace(0,0.3,mshape)
n,k = np.meshgrid(n,k)
m=list(map(complex,n.ravel(),k.ravel()))

# size: 1um
diameter=1000

def MieQ_fix_wd(m):
    return [m.real, m.imag, mie.AutoMieQ(m,wavelength,diameter)]

def ScatterFunc_fix_wd(m):
    return [m.real, m.imag, mie.ScatteringFunction(m,wavelength,diameter,angleMeasure='degrees')]

if __name__ == "__main__":
    st=datetime.datetime.now()
    with Pool() as p:
        data=p.map(MieQ_fix_wd,m)
    data=np.array(data)
    np.save('./data/Qdata',data)
    et=datetime.datetime.now()
    print("Spend time: {}".format(et-st))

    st=datetime.datetime.now()
    with Pool() as p:
        data=p.map(ScatterFunc_fix_wd,m)
    data=np.array(data)
    np.save('./data/SLRdata',data)
    et=datetime.datetime.now()
    print("Spend time: {}".format(et-st))