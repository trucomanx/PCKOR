#!/usr/bin/python

import sys,os

import extras

################################################################################
################################################################################
# Load dataset
data_filename='../dataset/dataset_peaks.txt';
X_train,y_train,X_valid,y_valid,X_test,y_test = extras.load_dataset(data_filename);
print('dataset train shape:',X_train.shape,y_train.shape)
print('dataset valid shape:',X_valid.shape,y_valid.shape)
print('dataset test  shape:',X_test.shape,y_test.shape)

################################################################################
################################################################################
# Applying KpcKor
sys.path.append('../src');
import PcKor.KpcKor as KpcKor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

Nparts=100;
sigma=np.linspace(0.3,1.5,Nparts);

mse_train=np.zeros((Nparts,));
mse_valid=np.zeros((Nparts,));
r2_train=np.zeros((Nparts,));
r2_valid=np.zeros((Nparts,));


for idx in range(Nparts):
    reg = KpcKor(kernel='rbf',gamma=0.5/(sigma[idx]*sigma[idx]));
    reg.fit(X_train, y_train)
    
    y=reg.predict(X_train);
    mse_train[idx]=mean_squared_error(y_train,y);
    r2_train[idx]=r2_score(y_train,y);
    
    y=reg.predict(X_valid);
    mse_valid[idx]=mean_squared_error(y_valid,y);
    r2_valid[idx]=r2_score(y_valid,y);
    
    print('idx:%2d\tsigma:%6.4f\tmse_valid:%6.4e'%(idx,sigma[idx],mse_valid[idx]));

idx_min = np.argmin(mse_valid);

################################################################################
################################################################################
## Analysis with the testing dataset
reg = KpcKor(kernel='rbf',gamma=0.5/(sigma[idx_min]*sigma[idx_min]));
reg.fit(X_train, y_train)
y=reg.predict(X_test);


################################################################################
################################################################################
output_dir='output_kpckor';
os.makedirs(output_dir, exist_ok = True)

################################################################################
################################################################################
## Saving history

history={
'sigma':sigma,
'mse_train':mse_train,
'mse_valid':mse_valid,
'r2_train':r2_train,
'r2_valid':r2_valid
};

extras.save_history_in_csv(history,out_file=os.path.join(output_dir,'history.csv'))

################################################################################
################################################################################
## Saving statistics
stats={
'sigma':sigma[idx_min],
'mse_train':mse_train[idx_min],
'mse_valid':mse_valid[idx_min],
'mse_test':mean_squared_error(y_test,y),
'r2_train':r2_train[idx_min],
'r2_valid':r2_valid[idx_min],
'r2_test':r2_score(y_test,y)
};

extras.save_stats_in_txt(stats,out_file=os.path.join(output_dir,'output.txt'))

################################################################################
################################################################################
## Eigenvalues

print('reg._eigenvalues_')
#print(reg._eigenvalues_)

################################################################################
################################################################################
## Plot
extras.plot_3d_points(np.loadtxt(data_filename),out_file=os.path.join(output_dir,'dataset_peaks.eps'));
DatTest=np.concatenate( ( X_test,np.reshape(y,(y.shape[0],1)) ), axis=1);
extras.plot_3d_points(DatTest,s_size=3,out_file=os.path.join(output_dir,'dataset_peaks_scatter3.eps'));
extras.plot_mse_results(sigma,mse_train,mse_valid,out_file=os.path.join(output_dir,'dataset_peaks_MSE.eps'));

