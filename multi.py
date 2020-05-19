from multiprocessing import Pool
import PyMieScatt as mie
import numpy as np

wavelength=550
diameter=1000

def ScaFunc(m):
    print(m)
    return [m.real,m.imag,mie.ScatteringFunction(m,wavelength,diameter)]

# refractive index m=n+ik
# n=1.3 .. 2 linear
# k=0 .. 1   log
n=np.linspace(1.3,2,100)
k=np.linspace(0,0.3,100)
# k=np.log(np.linspace(1,np.exp(1),100))

n,k = np.meshgrid(n,k)
m=list(map(complex,n.ravel(),k.ravel()))
# m=np.reshape(m,(100,100))

if __name__ == "__main__":
    with Pool(4) as p:
        data=p.map(ScaFunc,m)

    data=np.array(data)
    # data[:,0]=(data[:,0]*180/np.pi)
    print(data)