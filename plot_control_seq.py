import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import json
import os, sys
import numpy as np

if __name__ == "__main__":
    fname = sys.argv[1]
    data = np.loadtxt(fname)

    fig, ax = plt.subplots(1, figsize=(7,2.5))

    # output profile and set point
    y    = ax.plot(data[:,0],data[:,1], label='$y$')
    yset = ax.plot(data[:,0],data[:,2], label=r'$y_{set}$')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')

    # input profile
    axtwin = ax.twinx()
    u = axtwin.plot(data[:,0], data[:,3], label=r'$u$', color='red', linestyle='--')
    axtwin.set_ylabel('Power', color='red')
    axtwin.tick_params('y', colors='red')

    # shrink the axis for the legend on top
    box = ax.get_position()
    # Legend underneath
    #axlegend.set_position([box.x0, box.y0 + box.height * 0.1,
    #             box.width, box.height * 0.9])
    # Legend above
    ax.set_position([box.x0, box.y0 + box.height*0.05,
                 box.width, box.height * 0.9])


    alllines = y + yset + u
    alllabels = [l.get_label() for l in alllines]
    ax.legend(alllines, alllabels, ncol=3, bbox_to_anchor = (0, 1.02, 1, 0.2), loc='upper center')
    #ax.legend(alllines, alllabels, ncol=3, bbox_to_anchor = (1, 0), loc='lower right', bbox_transform=fig.transFigure)

    plt.tight_layout()
    plt.show()

    print(np.sum(np.abs(data[:,1]-data[:,2])))
    print(np.sum([np.abs(data[i+1,3]-data[i,3]) for i in range(len(data[:,3])-1)]))

    #fig, ax = plt.subplots(1, figsize=(7,2.5))

    #ax.plot(data[:,0],data[:,3], label='System')

    #ax.set_xlabel('Time')
    #ax.set_ylabel('Power')

    #ax.legend(loc='best')

    #plt.tight_layout()
    #plt.show()
