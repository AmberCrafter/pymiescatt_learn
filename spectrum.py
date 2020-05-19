import PyMieScatt as mie
import numpy as np

# wavelength=550

# refractive index m=n+ik
# n=1.3 .. 2 linear
# k=0 .. 1   log
n=np.linspace(1.3,2,100)
k=np.linspace(0,0.3,100)
# k=np.log(np.linspace(1,np.exp(1),100))

# diameter=1000

n,k = np.meshgrid(n,k)
m=list(map(complex,n.ravel(),k.ravel()))
# # m=np.reshape(m,(100,100))


# # m=[(1.3+0j),(1.3+0.1j),(1.5+1E-7j),(1.5+0.1j),(1.5+0.2j),(1.5+0.3j),(1.79+1E-7j),(1.79+0.1j),(1.79+0.2j),(1.79+0.3j)]
# data=[mie.ScatteringFunction(mm,wavelength,diameter) for mm in m]
# data=np.array(data)
# data[:,0]=(data[:,0]*180/np.pi)
data=np.load('./data.npy')

depoR=[]
for index,val in enumerate(data):
    n=m[index].real
    k=m[index].imag
    mask=val[0]>=90
    # depo=sum(val[2][mask])/sum(val[1][mask])
    depo=sum(val[3][mask])
    depoR.append([n,k,depo])
depoR=np.array(depoR)
print(depoR)
import matplotlib
import matplotlib.pyplot as plt
cf=plt.pcolormesh(depoR[:,0].reshape(100,100),depoR[:,1].reshape(100,100),depoR[:,2].reshape(100,100),cmap='jet',alpha=0.5,norm=matplotlib.colors.LogNorm())
plt.colorbar(cf)
plt.xlabel('Real',fontsize=16)
plt.ylabel('Image',fontsize=16)
plt.title('Spectrum Range: 90')
plt.savefig('./depoR_(90).jpg')
plt.show()
# from lib import graph
# # Qlist=[Qext,Qsca,Qabs,g,qpr,qback,qratio]
# # Qname=['Qext','Qsca','Qabs','g','qpr','qback','qratio']
# Qlist=[qback]
# Qname=['qback']
# for i in range(len(Qlist)):
#     graph.pcolor(n,k,np.log10(Qlist[i]),filename='123',title=Qname[i])


print('123')