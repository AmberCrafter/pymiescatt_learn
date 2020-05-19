import PyMieScatt as mie
import numpy as np

wavelength=550

# refractive index m=n+ik
# n=1.3 .. 2 linear
# k=0 .. 1   log
n=np.linspace(1.3,2,100)
k=np.linspace(0,0.3,100)
# k=np.log(np.linspace(1,np.exp(1),100))

diameter=1000

n,k = np.meshgrid(n,k)
m=list(map(complex,n.ravel(),k.ravel()))
# m=np.reshape(m,(100,100))

data=[mie.AutoMieQ(mm,wavelength,diameter) for mm in m]
data=np.array(data)
Qext=np.reshape(data[:,0],(100,100))
Qsca=np.reshape(data[:,1],(100,100))
Qabs=np.reshape(data[:,2],(100,100))
g=np.reshape(data[:,3],(100,100))
qpr=np.reshape(data[:,4],(100,100))
qback=np.reshape(data[:,5],(100,100))
qratio=np.reshape(data[:,6],(100,100))
# print(data)

# Qext=data[:,0]
# Qsca=data[:,1]
# Qabs=data[:,2]
# g=data[:,3]
# qpr=data[:,4]
# qback=data[:,5]
# qratio=data[:,6]

from lib import graph
# Qlist=[Qext,Qsca,Qabs,g,qpr,qback,qratio]
# Qname=['Qext','Qsca','Qabs','g','qpr','qback','qratio']
Qlist=[qback]
Qname=['qback']
for i in range(len(Qlist)):
    graph.pcolor(n,k,np.log10(Qlist[i]),filename='123',title=Qname[i])


print('123')