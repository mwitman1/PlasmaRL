#! /usr/bin/env python
import seaborn as sns                                                              
import pandas as pd                                                                
import matplotlib.pyplot as plt                                                    
plt.rcParams.update({'font.size': 10})                                             
import json                                                                        
import os, sys                                                                     
import numpy as np

print(sys.argv[1])
print(sys.argv[1][:-4])
data = np.loadtxt(sys.argv[1])

x = data[:,0]
y = data[:,1]
totale = data[:,2]
totaldelu = data[:,3]

fig = plt.figure(0, figsize = (1.6, 2.5))                                             
ax = fig.add_subplot(111)
cax = ax.scatter(x, y, c=totale, s=8, cmap=plt.cm.plasma)                                 
cbar = fig.colorbar(cax)                                                           
cbar.ax.set_title(r"$CE$")                                                         
ax.set_ylabel(r"$b$",rotation = 0)                                                              
ax.set_xlabel(r"$a$")                                                              
ax.yaxis.set_label_coords(-0.15, 1.05)
ax.xaxis.set_label_coords(1.3, -0.05)
plt.tight_layout()                                                              
plt.savefig(sys.argv[1][:-4]+"_totale.pdf",transparent=True)
plt.show()


fig = plt.figure(0, figsize = (1.6, 2.5))                                             
ax = fig.add_subplot(111)
cax = ax.scatter(x, y, c=totaldelu, s=8, cmap=plt.cm.plasma)                                 
cbar = fig.colorbar(cax)                                                           
cbar.ax.set_title(r"$CI$")                                                         
ax.set_ylabel(r"$b$",rotation = 0)                                                              
ax.set_xlabel(r"$a$")                                                              
ax.yaxis.set_label_coords(-0.15, 1.05)
ax.xaxis.set_label_coords(1.3, -0.05)
plt.tight_layout()                                                              
plt.savefig(sys.argv[1][:-4]+"_totaldelu.pdf",transparent=True)
plt.show()

