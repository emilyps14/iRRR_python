import numpy as np
from numpy.random import seed, randn, choice
from scipy.linalg import sqrtm

# This file provides simulation settings for Gaussian data.

def sim_normal_setting(choosesetting):
    #%% Simulation Setting
    # basic setting: two-set, low-dim, ind-pred, sep-coef
    n=500
    num=2
    p1=50
    p2=50
    p=[p1,p2]
    q=100
    r=10 # per coef mat rank

    Gamma=np.eye(p1+p2) # covariance

    seed(123456)
    L1=randn(p1,r)
    L2=randn(p2,r)
    R1=randn(q,r)
    R2=randn(q,r)
    B1=L1@R1.T
    B2=L2@R2.T
    B=[B1,B2]
    missing=[] # missing index for Y

    if choosesetting==11:
        # based on setting1, with 10% missing values
        missing=choice(n*q,np.round(0.1*n*q).astype(int),replace=False) # missing index of Y
    elif choosesetting==12:
        # 20% missing
        missing=choice(n*q,np.round(0.2*n*q).astype(int),replace=False) # missing index of Y
    elif choosesetting==13:
        # 30% missing
        missing=choice(n*q,np.round(0.3*n*q).astype(int),replace=False) # missing index of Y
    elif choosesetting==14:
        # 40% missing
        missing=choice(n*q,np.round(0.4*n*q).astype(int),replace=False) # missing index of Y
    elif choosesetting==2:
        # two-set, low-dim, CORR-PRED, sep-coef
        rho=0.9 # between var corr, across X1 and X2
        Gamma=np.ones((p1+p2,p1+p2))*rho+np.eye(p1+p2)*(1-rho) # override Gamma
    elif choosesetting==31:
        # two-set, low-dim, ind-pred, ONE-COEF
        r=20 # both [B1;B2] and B1 and B2 rank
        seed(123456)
        L1=randn(p1,r)
        L2=randn(p2,r)
        R=randn(q,r)
        B1=L1@R.T
        B2=L2@R.T
        B=[B1,B2] # override B
    elif choosesetting==32:
        # two-set, low-dim, ind-pred, ONE-COEF
        r=40 # both [B1;B2] and B1 and B2 rank
        seed(123456)
        L1=randn(p1,r)
        L2=randn(p2,r)
        R=randn(q,r)
        B1=L1@R.T
        B2=L2@R.T
        B=[B1,B2] # override B
    elif choosesetting==33:
        # two-set, low-dim, ind-pred, ONE-COEF
        r=60 # each B1 and B2 is full rank, Bk/c r>p1, r>p2
        seed(123456)
        L1=randn(p1,r)
        L2=randn(p2,r)
        R=randn(q,r)
        B1=L1@R.T
        B2=L2@R.T
        B=[B1,B2] # override B
    elif choosesetting==41:
        # THREE-SET, low-dim, ind-pred, sep-coef
        num=3
        p=[p1]*num

        Gamma=np.eye(np.sum(p))
        seed(123456)
        B = [randn(p1,r)@randn(r,q) for i in range(num)]
    elif choosesetting==42:
        # FOUR-SET, low-dim, ind-pred, sep-coef
        num=4
        p=[p1]*num

        Gamma=np.eye(np.sum(p))
        seed(123456)
        B = [randn(p1,r)@randn(r,q) for i in range(num)]
    elif choosesetting==43:
        # FIVE-SET, low-dim, ind-pred, sep-coef
        num=5
        p=[p1]*num

        Gamma=np.eye(np.sum(p))
        seed(123456)
        B = [randn(p1,r)@randn(r,q) for i in range(num)]
    elif choosesetting==44:
        # THREE-SET, one is redundant
        num=3
        p=[p1]*num

        Gamma=np.eye(np.sum(p))
        B = B+[np.zeros((p1,q))]

    hfGamma = sqrtm(Gamma)

    #%% Generate Tuning and Training predictors
    # for aRRR, we have to use the same number of samples in both sets

    # Tuning set
    cX_tune=randn(n,sum(p))@hfGamma
    cX_tune=np.subtract(cX_tune, cX_tune.mean(0))
    tempp1=np.cumsum([0]+p[:-1])
    tempp2=np.cumsum(p)
    X_tune = [cX_tune[:,i:j] for i,j in zip(tempp1,tempp2)]

    # Training set
    cX=randn(n,sum(p))@hfGamma
    cX=np.subtract(cX, cX.mean(0))
    X = [cX[:,i:j] for i,j in zip(tempp1,tempp2)]

    #%% adjust signal level in B
    # std of E is fixed to be 1
    temp=[Xk@Bk for Xk,Bk in zip(X,B)]  # linear predictor
    c=np.quantile(np.abs(temp),0.9)
    Btrue=np.vstack([Bk/c for Bk in B]) # set 90% quantile of linear predictor to be 1
    Gammatrue=Gamma # each Xi's covariance matrix

    return dict(n=n,
                num=num,
                p=p,
                q=q,
                r=r, # per coef mat rank
                missing=missing, # missing index for Y
                X_tune=X_tune,
                cX_tune=cX_tune,
                X=X,
                cX=cX,
                Btrue=Btrue,
                Gammatrue=Gammatrue
                )
