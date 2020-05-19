import matplotlib.pyplot as plt

def pcolor(x,y,z,filename:str,title:str=None):
    fig=plt.figure(figsize=[16.00,9.00],dpi=100)
    ax=fig.add_subplot(1,1,1)
    cf=ax.pcolormesh(x,y,z,cmap='jet',alpha=0.5)
    # cs=ax.contour(x,y,z,colors='k')
    # ax.set_yscale('log')
    ax.set_ylim([0,0.3])
    # ax.clabel(cs, cs.levels, inline=True, fontsize=14)

    ax.set_xlabel('Re(m)',fontsize=16)
    ax.set_ylabel('Im(m)',fontsize=16)
    if title: ax.set_title('{}'.format(title))

    plt.colorbar(cf,ax=ax)
    # cbar.ax.get_yaxis().set_ticks([])
    # for j, lab in enumerate(['$0$','$1$','$2$','$>3$']):
    #     cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center')
    plt.show()