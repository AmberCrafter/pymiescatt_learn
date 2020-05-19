import numpy as np

def weightingMap(radius=10,method='exp'):
    wmap=np.zeros((radius*2+1,radius*2+1))

    for i in range(radius*2+1):
        for j in range(radius*2+1):
            dummy=np.sqrt((i-radius)**2+(j-radius)**2)
            wmap[i,j]=np.exp(-dummy/(radius))
    return wmap

if __name__ == "__main__":
    mask=weightingMap()
    print(mask)
    import matplotlib.pyplot as plt
    plt.pcolor(mask)
    plt.show()