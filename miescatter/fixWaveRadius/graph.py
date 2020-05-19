import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
path=__file__.split("\\")
os.chdir('\\'.join(path[:-1]))

data=np.load('./data/Qdata.npy',allow_pickle=True)
WAVELENGTH=550
DIAMETER=1
mshape=100

def g_Qext():
    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    x=data[:,0].reshape(100,100)
    y=data[:,1].reshape(100,100)
    z=np.array([val[0] for val in data[:,2]]).reshape(100,100)
    cf=ax.pcolormesh(x,y,z,cmap='jet')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('Qext -- wavelength: {wavelength}mm; diameter: {diameter}um'.format(wavelength=WAVELENGTH,diameter=DIAMETER))
    plt.colorbar(cf,ax=ax)
    plt.savefig('./figure/Qext.jpg')
    # plt.show()

def g_Qsca():
    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    x=data[:,0].reshape(100,100)
    y=data[:,1].reshape(100,100)
    z=np.array([val[1] for val in data[:,2]]).reshape(100,100)
    cf=ax.pcolormesh(x,y,z,cmap='jet')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('Qsca -- wavelength: {wavelength}mm; diameter: {diameter}um'.format(wavelength=WAVELENGTH,diameter=DIAMETER))
    plt.colorbar(cf,ax=ax)
    plt.savefig('./figure/Qsca.jpg')
    # plt.show()

def g_Qabs():
    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    x=data[:,0].reshape(100,100)
    y=data[:,1].reshape(100,100)
    z=np.array([val[2] for val in data[:,2]]).reshape(100,100)
    cf=ax.pcolormesh(x,y,z,cmap='jet')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('Qabs -- wavelength: {wavelength}mm; diameter: {diameter}um'.format(wavelength=WAVELENGTH,diameter=DIAMETER))
    plt.colorbar(cf,ax=ax)
    plt.savefig('./figure/Qabs.jpg')
    # plt.show()

def g_g(): # asymmeter
    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    x=data[:,0].reshape(100,100)
    y=data[:,1].reshape(100,100)
    z=np.array([val[3] for val in data[:,2]]).reshape(100,100)
    cf=ax.pcolormesh(x,y,z,cmap='jet')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('g -- wavelength: {wavelength}mm; diameter: {diameter}um'.format(wavelength=WAVELENGTH,diameter=DIAMETER))
    plt.colorbar(cf,ax=ax)
    plt.savefig('./figure/g.jpg')
    # plt.show()

def g_Qpr(): # asymmeter
    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    x=data[:,0].reshape(100,100)
    y=data[:,1].reshape(100,100)
    z=np.array([val[4] for val in data[:,2]]).reshape(100,100)
    cf=ax.pcolormesh(x,y,z,cmap='jet')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('Qpr -- wavelength: {wavelength}mm; diameter: {diameter}um'.format(wavelength=WAVELENGTH,diameter=DIAMETER))
    plt.colorbar(cf,ax=ax)
    plt.savefig('./figure/Qpr.jpg')
    # plt.show()

def g_Qback(): # asymmeter
    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    x=data[:,0].reshape(100,100)
    y=data[:,1].reshape(100,100)
    z=np.array([val[5] for val in data[:,2]]).reshape(100,100)
    cf=ax.pcolormesh(x,y,z,cmap='jet',norm=matplotlib.colors.LogNorm())
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('Qback -- wavelength: {wavelength}mm; diameter: {diameter}um'.format(wavelength=WAVELENGTH,diameter=DIAMETER))
    plt.colorbar(cf,ax=ax)
    plt.savefig('./figure/Qback.jpg')
    # plt.show()

def g_Qratio(): # asymmeter
    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    x=data[:,0].reshape(100,100)
    y=data[:,1].reshape(100,100)
    z=np.array([val[6] for val in data[:,2]]).reshape(100,100)
    cf=ax.pcolormesh(x,y,z,cmap='jet')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('Qratio -- wavelength: {wavelength}mm; diameter: {diameter}um'.format(wavelength=WAVELENGTH,diameter=DIAMETER))
    plt.colorbar(cf,ax=ax)
    plt.savefig('./figure/Qratio.jpg')
    # plt.show()

def g_Qback_depo(angle):
    n=data[:,0].reshape(mshape,mshape)
    k=data[:,1].reshape(mshape,mshape)
    Qback=np.array([val[5] for val in data[:,2]]).reshape(mshape,mshape)

    SLRdata=np.load('./data/SLRdata.npy',allow_pickle=True)
    theta=SLRdata[0][2][0]
    mask=theta>=angle
    depo=[]
    for value in SLRdata:
        SL=value[2][1]
        SR=value[2][2]
        SU=value[2][3]
        depo.append((sum(SR[mask])/sum(SL[mask]))*2)
    depo=np.array(depo,dtype=float).reshape(mshape,mshape)

    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    cf=ax.pcolormesh(n,k,Qback,cmap='jet',norm=matplotlib.colors.LogNorm())
    cs=ax.contour(n,k,depo,colors='k',linewidth=2)
    ax.clabel(cs, cs.levels, inline=True, fontsize=14)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('Qback_depo -- wavelength: {wavelength}mm; diameter: {diameter}um; angle: {angle}'.format(wavelength=WAVELENGTH,diameter=DIAMETER,angle=angle))
    plt.colorbar(cf,ax=ax)
    plt.savefig('./figure/Qback_depo_{angle}.jpg'.format(angle=angle))
    # plt.show()


def g_SU_depo(angle):
    n=data[:,0].reshape(mshape,mshape)
    k=data[:,1].reshape(mshape,mshape)

    SLRdata=np.load('./data/SLRdata.npy',allow_pickle=True)
    theta=SLRdata[0][2][0]
    mask=theta>=angle
    depo=[]
    SU=[]
    for value in SLRdata:
        sl=value[2][1]
        sr=value[2][2]
        su=value[2][3]
        SU.append(sum(su[mask])*2)
        depo.append((sum(sr[mask])/sum(sl[mask]))*2)
    SU=np.array(SU,dtype=float).reshape(mshape,mshape)
    depo=np.array(depo,dtype=float).reshape(mshape,mshape)

    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)

    cf=ax.pcolormesh(n,k,SU,cmap='jet',norm=matplotlib.colors.LogNorm())
    cs=ax.contour(n,k,depo,colors='k',linewidth=2)
    ax.clabel(cs, cs.levels, inline=True, fontsize=14)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('SU_depo -- wavelength: {wavelength}mm; diameter: {diameter}um; angle: {angle}'.format(wavelength=WAVELENGTH,diameter=DIAMETER,angle=angle))
    plt.colorbar(cf,ax=ax)
    plt.savefig('./figure/SU_depo_{angle}.jpg'.format(angle=angle))
    # plt.show()

def main():
    # g_Qext()
    # g_Qsca()
    # g_Qabs()
    # g_g()
    # g_Qpr()
    # g_Qback()
    # g_Qratio()
    # g_Qback_depo(175)
    # g_Qback_depo(90)
    g_SU_depo(175)
    g_SU_depo(90)
    pass

if __name__ == "__main__":
    main()