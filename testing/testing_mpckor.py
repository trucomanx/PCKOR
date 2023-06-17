#!/usr/bin/python

import sys,os
import extras
import numpy as np

output_dir='output_mpckor';
M=100;
Nparts=40;
sigma=np.linspace(0.3,1.5,Nparts);

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
# Applying MpcKor
sys.path.append('../src');
import PcKor.MpcKor as MpcKor
from sklearn.metrics import mean_squared_error, r2_score


mse_train=np.zeros((Nparts,));
mse_valid=np.zeros((Nparts,));
r2_train=np.zeros((Nparts,));
r2_valid=np.zeros((Nparts,));


for idx in range(Nparts):
    reg = MpcKor(kernel='rbf',M=M,gamma=0.5/(sigma[idx]*sigma[idx]));
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
reg = MpcKor(kernel='rbf',gamma=0.5/(sigma[idx_min]*sigma[idx_min]));
reg.fit(X_train, y_train)
y=reg.predict(X_test);


################################################################################
################################################################################
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

CNE=reg.cumulative_normalized_eigenvalues();
extras.plot_cne(CNE,
                out_file=os.path.join(output_dir,'dataset_peaks_lambda.eps'),
                M=M,
                xlabel='$m$',
                ylabel='Cumulative Normalized $\gamma_{m}$ - $CNG(m)$',
                cne_label='$CNG$',
                title='$\sigma=%6.4f\quad CNG(%d)=%8.6f$'%(sigma[idx_min],M,CNE[M-1])
                );

np.savetxt(os.path.join(output_dir,'eigenvalues.txt'), reg._eigenvalues_, delimiter='\n');

################################################################################
################################################################################
## Plot
extras.plot_3d_points(np.loadtxt(data_filename),out_file=os.path.join(output_dir,'dataset_peaks.eps'));
DatTest=np.concatenate( ( X_test,np.reshape(y,(y.shape[0],1)) ), axis=1);
extras.plot_3d_points(DatTest,s_size=3,out_file=os.path.join(output_dir,'dataset_peaks_scatter3.eps'));
extras.plot_mse_results(sigma,mse_train,mse_valid,out_file=os.path.join(output_dir,'dataset_peaks_MSE.eps'));


