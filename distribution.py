import PyMieScatt as mie
import numpy as np

num=50
m=(1.495+0.09j)

wavelength=np.linspace(300,1200,num)
diameter=np.logspace(np.log10(1),np.log10(1000),num)
wavelength,diameter = np.meshgrid(wavelength,diameter)
wavelength=wavelength.reshape(-1)
diameter=diameter.reshape(-1)

data=[mie.ScatteringFunction(m,wavelength[i],diameter[i]) for i in range(len(wavelength))]
data=np.array(data)
data[:,0]=(data[:,0]*180/np.pi)
# data=np.load('./data_wd')

depoR=[]
for index,val in enumerate(data):
    x=wavelength[index]
    y=diameter[index]
    mask=val[0]>=90
    depo=sum(val[2][mask])/sum(val[1][mask])
    depoR.append([x,y,depo])
depoR=np.array(depoR)
print(depoR)
# import matplotlib
import matplotlib.pyplot as plt
cf=plt.pcolormesh(depoR[:,0].reshape(num,num),depoR[:,1].reshape(num,num),depoR[:,2].reshape(num,num),cmap='jet',alpha=0.5)#,norm=matplotlib.colors.LogNorm())
plt.colorbar(cf)
plt.xlabel('Wavelength',fontsize=16)
plt.ylabel('Diameter',fontsize=16)
plt.title('Spectrum Range: 90')
plt.savefig('./depoR_wd_(90)).jpg')
plt.show()
# from lib import graph
# # Qlist=[Qext,Qsca,Qabs,g,qpr,qback,qratio]
# # Qname=['Qext','Qsca','Qabs','g','qpr','qback','qratio']
# Qlist=[qback]
# Qname=['qback']
# for i in range(len(Qlist)):
#     graph.pcolor(n,k,np.log10(Qlist[i]),filename='123',title=Qname[i])


print('123')