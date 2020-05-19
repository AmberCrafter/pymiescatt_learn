def kernal(x,y,z):
    print('x: {}\ty: {}\tz: {}'.format(x,y,z))

def warpper(val,posi,*args,**keywords):
    print('Args: ',args)
    print('KeyW: ',keywords)
    keywords[posi]=val
    kernal(*args,**keywords)

def main():
    warpper([1,2,3],'x',x=1,y=2,z=3)

main()