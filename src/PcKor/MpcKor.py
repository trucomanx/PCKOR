#!/usr/bin/python

import os
import sys


import numpy as np
from numpy import linalg as LA

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels


class MpcKor(BaseEstimator, RegressorMixin):
    """Kernelized orthogonal regression with M principal components.
    
    Valid values for metric are:
        See https://scikit-learn.org/stable/modules/metrics.html
    
    Regression parameters
    ---------------------
    
    M : int default=None
        Indicates the quantity of eigenvalues (principal components) used.
        If None, the model uses M equal to the half part of samples.
        In other case uses the eigenvectors until the position `M`,
        when them are sorted from most to least significant.
        Being the position `m=0` the eigenvector with the eigenvalue most significant.
        M=M % (number of samples);
        To see the list of sorted eigenvalues, got to _eigenvectors_ .
        M take in count sort_abs.
    
    sort_abs : bool default=True
        If true, sort the eigenvalues applying the absolute value.
        In other case, sort the eingenvalues considering the sign.
        Theoretically they shouldn't exist negative values; but,
        by numerical computation problems, the calculus of eigenvalues 
        generates negative numbers close to zero.
    
    Kernel parameters
    -----------------
    To see the sense of parameters in the equation kernel goin to 
    https://scikit-learn.org/stable/modules/metrics.html
    
    kernel : str or callable, default="rbf"
        Kernel mapping used internally. This parameter is directly passed to
        `sklearn.metrics.pairwise.pairwise_kernel`.
        If `kernel` is a string, it must be one of the metrics
        in the dictionary `sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS`.
        Alternatively, if `kernel` is a callable function 
        (See function `self._get_kernel()`), it is called on
        each pair of instances (rows) and the resulting value recorded. The
        callable should take two rows from X as input and return the
        corresponding kernel value as a single number. This means that
        callables from :mod:`sklearn.metrics.pairwise` are not allowed, as
        they operate on matrices, not single samples. Use the string
        identifying the kernel instead.

    gamma : float, default=0.5
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.

    degree : int, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dict, default=None
        Additional parameters (keyword arguments later X,y) for kernel function passed
        as callable object. See function `self._get_kernel()`.

    Attributes
    ----------
    _eigenvectors_ : Numpy matrix with the eigenvector in the columns.
    
    _eigenvalues_ : list of eigenvalues.


    See Also
    --------
    PcKor.KpcKor : Kernelized orthogonal regression with K components.

    References
    ----------
    * Fernando Pujaico Rivera
      "Kernelized orthogonal regression with M principal components"

    Examples
    --------
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(n_samples, n_features)
    >>> y = rng.randn(n_samples)
    >>>
    >>> import PcKor.MpcKor as MpcKor
    >>> kkor = MpcKor(kernel='rbf', gamma=0.5)
    >>> kkor.fit(X, y)
    """
    def __init__(self, M=None, sort_abs=True, kernel='rbf', gamma=0.5, degree=3, coef0=1, kernel_params=None):
        # Regression parameters
        self.M=M;
        self.sort_abs=sort_abs;
        
        # Kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        
        # output params: eigen dats
        self._eigenvalues_=None 
        self._eigenvectors_ =None
        
        # output params: regression
        self.w=None;
        self.w0=None;
        self.X_fit_ = None;
        
    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params);
        
    def fit(self, X, y):
        
        if y.shape[0] != X.shape[0]:
            sys.exit('Shape of X and y is not compatible.');
        
        L=X.shape[0];
        N=X.shape[1];
        if self.M==None:
            nhat=int(L/2);
        elif isinstance(self.M, int) and self.M>0:
            nhat=self.M%L;
        else:
            sys.exit('M should be a positive integer or None.M: '+str(M));
        
        ## Setting variables
        IminusB=np.eye(L) - np.ones((L,L))*(1.0/L);
        b=np.ones((L,1))*(1.0/L);
        Y=np.array(y).reshape((L,1));
        Khat = self._get_kernel(X);
        #### Khat = 0.5*(Khat + Khat.T);
        
        KhatplusYYT=(Y@Y.T)+Khat;
        K=IminusB@(KhatplusYYT@IminusB);
        #### K=0.5*(K+K.T);
        
        ## obtaining eigenvalues and eigenvectors in descending order
        st0 = np.random.get_state()
        np.random.seed(0);
        self._eigenvalues_, self._eigenvectors_ = LA.eigh(K);
        np.random.set_state(st0)
        if self.sort_abs==True:
            idx = np.abs(self._eigenvalues_).argsort()[::-1];
        else:
            idx = self._eigenvalues_.argsort()[::-1];
        self._eigenvalues_  = self._eigenvalues_[idx];
        self._eigenvectors_ = self._eigenvectors_[:,idx];
        
        ## MpcKor
        U=self._eigenvectors_[:,0:nhat];
        Ginv=np.linalg.inv(np.diag(self._eigenvalues_[0:nhat]));
        
        R=IminusB@U@Ginv@U.T@IminusB;
        r=b-R@KhatplusYYT@b;
        
        tmp_vec=Y/((1-Y.T@R@Y)[0][0]);
        
        self.w=R.T@tmp_vec;
        self.w0=(r.T@tmp_vec)[0][0];
        self.X_fit_ = X;
        
        return self
    
    def predict(self, X):
        k = self._get_kernel(X, self.X_fit_).T;
        
        return (self.w0+self.w.T@k).reshape((X.shape[0],));
    
    def cumulative_normalized_eigenvalues(self):
        
        CNE=np.zeros(self._eigenvalues_.shape);
        
        N=np.size(CNE);
        S=0.0;
        for n in range(N):
            S=S+np.abs(self._eigenvalues_[n]);
            CNE[n]=S;
        
        return CNE/S;
        
