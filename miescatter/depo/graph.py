import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
path=__file__.split("\\")
os.chdir('\\'.join(path[:-1]))

data_rad=np.load('../fixWaveRadius/data/data.npy',allow_pickle=True)
data_spt=np.load('../fixRefractiveIndex/data/data.npy',allow_pickle=True)
mshape=100
mREAL=1.495
mIMAG=0.09

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def g_Qback_depo(angle,Wavelength=550,Diameter=1000):
    n=data_rad[:,0].reshape(mshape,mshape)
    k=data_rad[:,1].reshape(mshape,mshape)
    Qback=np.array([val[5] for val in data_rad[:,2]]).reshape(mshape,mshape)

    x=find_nearest(data_spt[:,0],Wavelength)
    y=find_nearest(data_spt[:,1],Diameter)
    z=data_spt[x[0]+y[0]]

    x=z[2][0]
    SL=z[2][1]
    SR=z[2][2]
    SU=z[2][3]



    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    cf=ax.pcolormesh(n,k,z,cmap='jet',norm=matplotlib.colors.LogNorm())
    ct=ax.contourf(n,k,)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('Qback_depo -- Re(m): {mREAL}; Im(m): {mIMAG}; Wave: {wavelength}mm; Diameter: {diameter}mm'.format(mREAL=mREAL,mIMAG=mIMAG,wavelength=Wavelength,diameter=Diameter))
    plt.colorbar(cf,ax=ax)
    plt.savefig('./figure/Qback_depo_w{}_d{}.jpg'.format(Wavelength,Diameter))
    print('w: {}\td: {}'.format(Wavelength,Diameter))
    # plt.show()


def main():
    g_S(550,1000)
    pass

if __name__ == "__main__":
    main()