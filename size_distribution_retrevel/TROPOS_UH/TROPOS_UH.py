import sys
import enum
import numpy as np
import matplotlib.pyplot as plt

from lib.linear import B_spline_forcus
from lib.matrix import smooth_function

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


def get_data():
    datalist=[
        'BsB0_S11',
        'BsG0_S11',
        'BsR0_S11',
        'BbsB0_S11',
        'BbsG0_S11',
        'BbsR0_S11',
        'BaB0_A11',
        'BaG0_A11',
        'BaR0_A11'
    ]
    filepath='./data.csv'
    f=open(filepath,'r')
    header=f.readline().replace('\n','').replace(' ','').split(',')
    index_data=[header.index(val) for val in datalist]
    
    readin=f.readline()
    data=[]
    while readin:
        readin=readin.replace('\n','').replace(' ','').split(',')
        data.append([float(readin[i]) for i in index_data])
        break
    # print(data)
    # data=[[data[0][i] for i in [5,4,7,8,3,2,1,0,6]]]
    data=[
        data[0][0]+data[0][6],
        data[0][1]+data[0][7],
        data[0][2]+data[0][8],
        data[0][3],
        data[0][4],
        data[0][5]
    ]
    return data

def calculate_weight_coefficients(K,b):
    from numpy import linalg as la
    rmin=0.001
    rmax=10
    point=6
    resolution=100

    H=smooth_function(6,1)
    data=grid_mie_kernal_function_diameter(rmin,rmax,point,realm=1.5,imagm=0.20,wavelength=[450,550,700],opticalType='Ext',resolution=resolution)
    diameter=data[0]
    wavelength=data[1]
    A=data[2]
    data=grid_mie_kernal_function_diameter(rmin,rmax,point,realm=1.5,imagm=0.20,wavelength=[450,550,700],opticalType='bsca',resolution=resolution)
    A=np.vstack((A,data[2]))
    # data=grid_mie_kernal_function_diameter(rmin,rmax,point,realm=1.5,imagm=0.20,wavelength=[450,550,700],opticalType='Abs',resolution=resolution)
    # A=np.vstack((A,data[2]))
    # print(A)
    data=np.array(get_data(),dtype=float)
    # data=[[data[0][i] for i in [5,4,7,8,3,2,1,0,6]]]
    data=np.array(data).T
    # data=(data.reshape(3,3)).T
    # gamma=1E-12
    gamma=2**K*10**-b
    C=np.dot(np.dot(la.inv(np.dot(A.T,A)+gamma*H),A.T),data)
    # print(C)
    return (data,A,C,diameter,wavelength)

def calculate_discrepancy():
    K=np.arange(26)  # K=15 is the best
    rho=[]
    for k in K:
        data,A,C,D,W=calculate_weight_coefficients(k,23)
        dummy=np.dot(A,C)

        # Euclidean
        dummy=abs(data-dummy)/data 
        rho.append(np.nanmean(dummy))

        # print(rho)
    print(rho)
    plt.plot(rho)
    plt.show()

    
def calculate_size_distribution_expressed():
    H=smooth_function(6,1)
    
    data,A,C,D,W=calculate_weight_coefficients(15,23)
    gamma=2**15*10**-23
    dummy=np.dot(A,C)
    summy=np.dot(C.T,H)
    error=np.dot(dummy.T,dummy)+gamma*np.dot(summy,C)
    # print('stop')

    plt.plot(D[0],C)
    plt.show()
    print('stop')
    
    



# get_data()
calculate_discrepancy()
calculate_size_distribution_expressed()

# H=smooth_function(6,1)

# # data=grid_mie_kernal(realm=1.5,imagm=0.02,wavelength=[400,500],diameter=[1000,2500,5000],xpara="wavelength",ypara="diameter",)
# data=grid_mie_kernal_function_diameter(0.001,10,6,realm=1.5,imagm=0.20,wavelength=[450,550,700],opticalType='Sca',resolution=100)
# A=data[2]
# data=grid_mie_kernal_function_diameter(0.001,10,6,realm=1.5,imagm=0.20,wavelength=[450,550,700],opticalType='Abs',resolution=100)
# A=np.vstack((A,data[2]))
# from numpy import linalg as la

# print(A)


# # np.dot(np.dot(la.inv(np.dot(A.T,A)+1E-12*H),A.T),[[96.41],[66.69],[41.29],[8.36],[7.44],[6.08]])



def main():
    # warpper_B_spline()
    pass