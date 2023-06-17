#!/usr/bin/python

import os, contextlib
import numpy as np

def load_dataset(input_file):
    DATA=np.loadtxt(input_file);
    X_train = DATA[0:432,0:2];
    y_train = DATA[0:432,2];
    
    X_valid = DATA[432:864,0:2];
    y_valid = DATA[432:864,2];
    
    X_test  = DATA[864:1296,0:2];
    y_test  = DATA[864:1296,2];
    
    return X_train,y_train,X_valid,y_valid,X_test,y_test;


def save_stats_in_txt(mydict,out_file):
    with open(out_file, 'w') as f:
        for x, y in mydict.items():
            f.write(x+'='+str(y)+'\n');
            print(x, y) 

def save_history_in_csv(mydict,out_file,sep=','):
    with open(out_file, 'w') as f:
        Key=list(mydict.keys());
        N=len(Key);
        for n in range(N):
            f.write(Key[n]);
            if n!=N-1:
                f.write(sep);
            else:
                f.write('\n');
        
        Value=list(mydict.values());
        L=len(Value[0]);
        for l in range(L):
            for n in range(N):
                f.write(str(Value[n][l]));
                if n!=N-1:
                    f.write(sep);
                else:
                    f.write('\n');

import matplotlib.pyplot as plt
def plot_mse_results(sigma,mse_train,mse_valid,out_file):
    Nparts=len(sigma);
    idx_min = np.argmin(mse_valid);
    
    line_x=Nparts*[sigma[idx_min]];
    min_y=np.min(np.minimum(mse_valid,mse_train));
    max_y=np.max(np.maximum(mse_valid,mse_train));
    line_y=np.linspace(min_y,max_y,Nparts);
    line_label='$\sigma$: %6.4f'%(sigma[idx_min]);
    
    fig = plt.figure();
    plt.semilogy( sigma, mse_train, label='train'   ,color='black',linestyle='dashdot');
    plt.semilogy( sigma, mse_valid, label='valid'   ,color='black',linestyle='solid');
    plt.semilogy(line_x,    line_y, label=line_label,color='black',linestyle='dotted');
    plt.legend(loc='lower right');
    plt.xlabel('$\sigma$');
    plt.ylabel('MSE');
    
    #fig.set_rasterized(True);
    
    ## The PostScript backend does not support transparency; partially transparent artists will be rendered opaque.
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            plt.savefig(out_file,bbox_inches='tight');

from mpl_toolkits import mplot3d
from matplotlib import cm
def plot_3d_points(X,out_file,s_size=1):
    fig = plt.figure(frameon = False);
    ax = plt.axes(projection ="3d");
    
    ax.scatter(X[:,0],X[:,1], X[:,2],c=X[:,2],cmap=cm.gray,s=s_size);

    ax.set_xlabel('$x_1$');
    ax.set_ylabel('$x_2$');
    ax.set_zlabel('$x_3$');

    plt.autoscale(tight=True)
    
    #ax.set_rasterized(True)
    
    ## The PostScript backend does not support transparency; partially transparent artists will be rendered opaque.
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            plt.savefig(out_file,bbox_inches='tight');
 


def plot_cne(CNE,out_file,M=None,xlabel='n',ylabel='CNE',cne_label='CNE',title=''):
    Nparts=np.size(CNE);
    
    count=np.linspace(1,Nparts,Nparts);
    
    fig = plt.figure();
    plt.plot( count,   CNE, label=cne_label ,color='black',linestyle='solid');
    if M!=None:
        line_x=M*np.ones((Nparts,));
        line_y=np.linspace(np.min(CNE),1,Nparts);
        line_label='m: %d'%(M);
        plt.plot(line_x,line_y, label=line_label,color='black',linestyle='dotted');
    plt.legend(loc='lower right');
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.title(title);
    
    #fig.set_rasterized(True);
    
    ## The PostScript backend does not support transparency; partially transparent artists will be rendered opaque.
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            plt.savefig(out_file,bbox_inches='tight');
