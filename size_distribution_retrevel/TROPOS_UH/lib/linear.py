import sys
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
    elif rc==r:
        return 1
    else:
        raise Exception('bug')

def B_spline_forcus(r,rbot,rc,rtop,rmin=0,rmax=sys.maxsize):
    return B_spline(r,rmin,rbot,rc,rtop,rmax)

def B_spline_maxtrix(rlist,*args,**keywords):
    B_spline_forcus(*args,**keywords)