import sys
import enum
import numpy as np
import matplotlib.pyplot as plt
# K=np.arange(1,26)

# gamma=2**K*10**-23


# plt.plot(K,gamma)
# plt.show()

def smooth_function(n,order):
    '''
    n: smooth_function size = n*n \n
    order:\n
    0: [1]\n
    1: [1, 2, 1]\n
    2: [1, 4, 6, 4, 1]\n
    '''
    if order>2: raise Exception('Undefined Behavior.')
    # if n%2==0: raise Exception('The dimension(n) of matrix must odd.')
    if n<order**2+1: raise Exception('The dimension(n) of matrix is too small.')
    order_dict=[
        [1],        # 0
        [2,1],      # 1
        [6,4,1],    # 2
        [20,15,6,1] # 3
    ]
    dummy=np.zeros((n,n))
    center=int((n-1)/2)
    for c in range(center):
        for i in range(order+1):
            dummy[center+c,center+c-i]=order_dict[order][i]
            dummy[center-c,center-c+i]=order_dict[order][i]
            if ((center-c)<0) or ((center+c)>=n) or ((center-c-i)<0) or ((center+c+i)>=n): continue
            dummy[center-c,center-c-i]=order_dict[order][i]
            dummy[center+c,center+c+i]=order_dict[order][i]
    if n%2==0:
        for i in range(order+1):
            dummy[-2,-2-i]=order_dict[order][i]
            if ((center-c)<0) or ((center+c)>=n) or ((center-c-i)<0) or ((center+c+i)>=n): continue
            dummy[-2,-2+i]=order_dict[order][i]

    for i in range(order+1):
        dummy[0+i,0]   = order_dict[order-1][(i+1)%order]
        dummy[0,0+i]   = order_dict[order-1][(i+1)%order]
        dummy[-1-i,-1] = order_dict[order-1][(i+1)%order]
        dummy[-1,-1-i] = order_dict[order-1][(i+1)%order]
    
    # set sign
    i,j=dummy.shape
    for ii in range(i):
        for jj in range(j):
            dummy[ii,jj]*=(-1)**(ii+jj)
    
    # recheck diag val
    for i in range(order):
        dummy[i,i]=-(sum(dummy[i])-dummy[i,i])
        dummy[-1-i,-1-i]=-(sum(dummy[-1-i])-dummy[-1-i,-1-i])
    return dummy
    # print(dummy)

def B_spline(r,rmin,rbot,rc,rtop,rmax):
    '''
    |-----------------------------------|
                 ^
                / \\
               /   \\
              /     \\
             /       \\
            /         \\
    |--------/--------|---------\\  ---------------->|\n
    rmin .. rbot ...... rc .......... rtop  ...................  rmax
    '''
    if (r<=rbot) or (r>=rtop): 
        return 0
    elif rbot<r and r<rc:
        return (1-(rc-r)/(rc-rbot))
    elif rc<r and r<rtop:
        return (1-(r-rc)/(rtop-rc))
    else:
        raise Exception('bug')

def B_spline_forcus(r,rbot,rc,rtop,rmin=0,rmax=sys.maxsize):
    return B_spline(r,rmin,rbot,rc,rtop,rmax)

def B_spline_maxtrix(rlist,*args,**keywords):
    B_spline_forcus(*args,**keywords)

def mie_kernal(realm,imagm,wavelength,diameter,opticalType='Sca',*args,**keywrods):
    import PyMieScatt as mie
    class optical_enum(enum.IntEnum):
        Extinction  = 0
        Scattering  = 1
        Absorption  = 2
        g           = 3     # asymmetry parameter
        Pressure    = 4     # radiative pressure
        Back        = 5
        Ratio       = 6
        # alias
        Ext = Extinction
        ext = Extinction
        Sca = Scattering
        sca = Scattering
        Abs = Absorption
        abs = Absorption
        pr  = Pressure
        back= Back
        bsca= Back
        bSca= Back
        ratio=Ratio

    return mie.MieQ(complex(realm,imagm),wavelength,diameter,*args,**keywrods)[optical_enum[opticalType].value]

def grid_mie_kernal(xpara:str,ypara:str,*args,**keywrods):
    '''
    <parameter>\n
    realm: Real part of refractive index\n
    imagm: Image part of refractive index\n
    wavelength: wavelength of the light\n
    diameter: particle size of diameter\n
    ------------------------------------------\n
    return [ meshgrid(x), meshgrid(y), value ]
    '''
    import numpy as np
    x=keywrods[xpara]
    y=keywrods[ypara]
    x,y=np.meshgrid(x,y)
    z=[]
    for point in zip(x.reshape(-1),y.reshape(-1)):
        keywrods[xpara]=point[0]
        keywrods[ypara]=point[1]
        z.append(mie_kernal(*args,**keywrods))
    z=np.array(z).reshape(x.shape)
    return np.array([x,y,z])

def grid_mie_kernal_function_diameter(rmin,rmax,N,wavelength:list,method='linear',resolution=None,r_step=None,*args,**keywrods):
    '''
    <parameter>\n
    rmin,rmax: Be considered as the limit of observation boundary\n
    N: The numbers of optical data\n
    method: seperated method including ['linear','lognormal']
    resolution: kernal function resolution on diameter, if resolution=None, resolution=N

    wavelength: list of the light wavelength

    ==========================================================
    return [x_grid,y_grid,data_grid]
    '''
    # Amatrix = [kernalA*B-spline]

    if resolution==None:
        resolution=N

    # rc: 顆粒大小中心，用於標定反演粒徑大小格點
    if method=='linear':
        rc=np.linspace(rmin,rmax,N+2)
    if method=='lognormal':
        rc=np.logspace(np.log(rmin),np.log(rmax),N+2,base=np.exp(1))

    Amatrix=[]
    for wl in wavelength:
        for i in range(N):
            diameter=np.linspace(rc[i],rc[i+2],resolution)
            kernalA=0
            dummy=diameter[0]
            kummy=0
            for d in diameter:
                kernal=mie_kernal(diameter=d,wavelength=wl,*args,**keywrods)
                bspline=B_spline_forcus(d,rc[i],rc[i+1],rc[i+2])
                kernalA+=(kummy+(kernal*bspline))*(d-dummy)/2
                kummy=kernal*bspline
                dummy=d
            Amatrix.append([rc[i+1],wl,kernalA])  # diameter,wavelength,kernal_function
    # print(Amatrix)
    Amatrix=np.array(Amatrix)
    return [
        Amatrix[:,0].reshape(len(wavelength),N),
        Amatrix[:,1].reshape(len(wavelength),N),
        Amatrix[:,2].reshape(len(wavelength),N)
    ]


def _fit_mie_kernal(xpara:str,ypara:str,*args,**keywrods):
    '''
    <parameter>\n
    realm: Real part of refractive index\n
    imagm: Image part of refractive index\n
    wavelength: wavelength of the light\n
    diameter: particle size of diameter\n
    ------------------------------------------\n
    return [ meshgrid(x), meshgrid(y), value ]
    '''
    import numpy as np
    import scipy
    x=keywrods[xpara]
    y=keywrods[ypara]
    x,y=np.meshgrid(x,y)
    z=[]
    for point in zip(x.reshape(-1),y.reshape(-1)):
        keywrods[xpara]=point[0]
        keywrods[ypara]=point[1]
        z.append(mie_kernal(*args,**keywrods))
    z=np.array(z).reshape(x.shape)

    # Plane fit method by leastsq
    def plane_errors(para, points):
        """plane error function"""
        a0, a1, a2 = para
        return a0 * points[:, 0] + a1 * points[:, 1] + a2 - points[:, 2]
    
    tparap = scipy.optimize.leastsq(plane_errors, [1, 1, 1], z)
    para = tparap[0]

def warpper_fit_mie_kernal():
    pass
    # x=np.linspace()

def main():
    # warpper_B_spline()
    pass


H=smooth_function(6,1)

# data=grid_mie_kernal(realm=1.5,imagm=0.02,wavelength=[400,500],diameter=[1000,2500,5000],xpara="wavelength",ypara="diameter",)
data=grid_mie_kernal_function_diameter(0.001,10,6,realm=1.5,imagm=0.20,wavelength=[450,550,700],opticalType='Sca',resolution=100)
A=data[2]
data=grid_mie_kernal_function_diameter(0.001,10,6,realm=1.5,imagm=0.20,wavelength=[450,550,700],opticalType='Abs',resolution=100)
A=np.vstack((A,data[2]))
from numpy import linalg as la

print(A)


# np.dot(np.dot(la.inv(np.dot(A.T,A)+1E-12*H),A.T),[[96.41],[66.69],[41.29],[8.36],[7.44],[6.08]])