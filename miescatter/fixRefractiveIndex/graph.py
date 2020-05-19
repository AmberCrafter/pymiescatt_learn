import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
path=__file__.split("\\")
os.chdir('\\'.join(path[:-1]))

mshape=100
mREAL=1.495
mIMAG=0.09
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
SLRdata=np.load('./data/SLRdata.npy',allow_pickle=True)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def g_S(Wavelength,Diameter):
    x=find_nearest(SLRdata[:,0],Wavelength)
    y=find_nearest(SLRdata[:,1],Diameter)
    z=SLRdata[x[0]+y[0]]

    x=z[2][0]
    SL=z[2][1]
    SR=z[2][2]
    SU=z[2][3]

    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    plt.plot(x,SL,linewidth=2,label='Parallel')
    plt.plot(x,SR,linewidth=2,label='Perpendicular')
    plt.plot(x,SU,linewidth=2,label='Unpolarized')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Scattered intensity')
    ax.set_yscale('log')
    ax.set_title('Spectrum -- Re(m): {mREAL}; Im(m): {mIMAG}; Wave: {wavelength}mm; Diameter: {diameter}mm'.format(mREAL=mREAL,mIMAG=mIMAG,wavelength=Wavelength,diameter=Diameter))
    plt.savefig('./figure/Spectrum_w{}_d{}.jpg'.format(Wavelength,Diameter))
    print('w: {}\td: {}'.format(Wavelength,Diameter))
    # plt.show()

def g_Q(parameter:str):
    '''parameter=['Ext', 'Sca', 'Abs', 'g', 'Pr', 'Back', 'Ratio']'''
    nparameter=Qparameter[parameter]
    x=Qdata[:,0].reshape(mshape,mshape)    # wavelength
    y=(Qdata[:,1]/1000).reshape(mshape,mshape)    # diameter
    z=np.array([val[nparameter] for val in Qdata[:,2]]).reshape(mshape,mshape)   # parameter

    # i=find_nearest(Qdata[:,0],700) # wavelength
    # j=find_nearest(Qdata[:,1],1500)  # diameter
    # k=Qdata[i[0]+j[0]]
    # print(i[0],j[0],k[2][1])

    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    cf=ax.pcolormesh(y,x,z,cmap='jet')
    ax.axhline(450,linewidth=2)
    ax.axhline(550,linewidth=2)
    ax.axhline(700,linewidth=2)
    ax.axvline(1,linewidth=2)
    ax.axvline(1.5,linewidth=2)
    ax.axvline(0.5,linewidth=2)

    ax.set_xlabel(r'Diameter ($\mu$m)')
    ax.set_ylabel('Wavelength (mm)')
    ax.set_title('Q{parameter} -- mREAL: {mREAL}; mIMAG: {mIMAG}'.format(parameter=parameter.lower(),mREAL=mREAL,mIMAG=mIMAG))
    plt.colorbar(cf,ax=ax)
    # plt.savefig('./figure/Qext.jpg')
    plt.show()

    

def main():
    # g_S(550,1000)
    g_Q('Sca')
    pass

if __name__ == "__main__":
    main()