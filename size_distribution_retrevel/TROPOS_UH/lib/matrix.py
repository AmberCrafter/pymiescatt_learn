import numpy as np
def _pascal_triangle():
    L = [1]
    while True:
        yield L
        L = [sum(i) for i in zip([0]+L, L+[0])]

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