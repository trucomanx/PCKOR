#!/usr/bin/python

import PcKor.MpcKor as MpcKor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from tqdm.notebook import tqdm
import numpy as np

def FuncMpKorKfoldBestGaussian(m_list,gamma_list,X_train, y_train,K=3):
    found=False; 
    k=0;
    
    pbar=tqdm(range(np.size(m_list)));
    SCORE_AG=np.zeros((np.size(m_list),np.size(gamma_list)))
    
    for j in pbar:
        M=m_list[j]
        score_val=[];
        ng=0;
        for gamma in gamma_list:
            kor = MpcKor(M=M,kernel="rbf",gamma=gamma);
            cv = KFold(n_splits=K, random_state=1, shuffle=True);
            
            scores = cross_val_score(kor, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
            
            sv=np.mean(scores);
            sv_std=np.std(scores);
            SCORE_AG[j][ng]=sv;

            score_val.append(sv);
            if k==0:
                m_opt=M;
                gamma_opt=gamma;
                score_opt=sv;
                kor_opt=kor;

                found=True;
            else:
                if sv>score_opt:
                    m_opt=M;
                    gamma_opt=gamma;
                    score_opt=sv;
                    kor_opt=kor;

                    cad="";
                    cad=cad+"R^2 val: %.3f" % sv;
                    cad=cad+" (%.3f)" % sv_std;
                    cad=cad+"\tm:%d" % M;
                    cad=cad+"\tgamma:%.3e" % gamma;
                    pbar.set_description(cad);
                    found=True;
            k=k+1
            ng=ng+1;
        if(found):
            score_val_opt=score_val.copy();
            found=False

    kor_opt.fit(X_train, y_train);
    print("\nR^2 train+val:",kor_opt.score(X_train, y_train));

    return kor_opt, m_opt, gamma_opt, score_val_opt, SCORE_AG


def FuncMpKorKfoldBestPolynomial(m_list,gamma_list,X_train, y_train,K=3):
    found=False; 
    k=0;
    
    pbar=tqdm(range(np.size(m_list)));
    SCORE_AG=np.zeros((np.size(m_list),np.size(gamma_list)))
    
    for j in pbar:
        M=m_list[j]
        score_val=[];
        ng=0;
        for gamma in gamma_list:
            kor = MpcKor(degree=M,kernel="poly",gamma=gamma,coef0=1.0);
            cv = KFold(n_splits=K, random_state=1, shuffle=True);
            
            scores = cross_val_score(kor, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
            
            sv=np.mean(scores);
            sv_std=np.std(scores);
            SCORE_AG[j][ng]=sv;

            score_val.append(sv);
            if k==0:
                m_opt=M;
                gamma_opt=gamma;
                score_opt=sv;
                kor_opt=kor;

                found=True;
            else:
                if sv>score_opt:
                    m_opt=M;
                    gamma_opt=gamma;
                    score_opt=sv;
                    kor_opt=kor;

                    cad="";
                    cad=cad+"R^2 val: %.3f" % sv;
                    cad=cad+" (%.3f)" % sv_std;
                    cad=cad+"\tm:%d" % M;
                    cad=cad+"\tgamma:%.3e" % gamma;
                    pbar.set_description(cad);
                    found=True;
            k=k+1
            ng=ng+1;
        if(found):
            score_val_opt=score_val.copy();
            found=False

    kor_opt.fit(X_train, y_train);
    print("\nR^2 train+val:",kor_opt.score(X_train, y_train));

    return kor_opt, m_opt, gamma_opt, score_val_opt, SCORE_AG

