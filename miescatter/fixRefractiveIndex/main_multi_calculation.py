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
# 200 .. 1200 mm
wavelength=np.linspace(200,1200,mshape)

# refractive index m=n+ik
m = 1.79+0.20j

# size
# 0.1 .. 10 um
diameter=np.linspace(100,10000,mshape)

# meshgrid of wavelength and diameter
w,d = np.meshgrid(wavelength,diameter)

def MieQ_fix_ri(wavelength,diameter):
    return [wavelength, diameter, mie.MieQ(m,wavelength,diameter)]

def ScaFunc_fix_ri(wavelength,diameter):
    return [wavelength, diameter, mie.ScatteringFunction(m,wavelength,diameter,angleMeasure='degrees')]

if __name__ == "__main__":
    st=datetime.datetime.now()
    with Pool() as p:
        data=p.starmap(MieQ_fix_ri,zip(w.reshape(-1),d.reshape(-1)))
    data=np.array(data)
    np.save('./data/Qdata',data)
    et=datetime.datetime.now()
    print("Spend time: {}".format(et-st))

    # st=datetime.datetime.now()
    # with Pool() as p:
    #     data=p.starmap(ScaFunc_fix_ri,zip(w.reshape(-1),d.reshape(-1)))
    # data=np.array(data)
    # np.save('./data/SLRdata',data)
    # et=datetime.datetime.now()
    # print("Spend time: {}".format(et-st))