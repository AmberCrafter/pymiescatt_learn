# import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import datetime
import time

def func(x,y):
    print(x,y, datetime.datetime.now(),'\n')
    time.sleep(2)
    return x*y

if __name__ == "__main__":
    x=np.linspace(1,10,10)
    y=np.linspace(3,30,10)

    with Pool(3) as p:
        Ans=p.starmap(func,zip(x,y))
    print(Ans)

