import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from scipy.linalg import pinv, svd, norm, svdvals
# from scipy.sparse.linalg import svds

def irrr_normal(Y,X,lam1,params=None):
    """
    This function uses consensus ADMM to fit the iRRR model. It is suitable
    for continuous outcomes (no missing or missing).

    Model:
    1/(2n)*|Y-1*mu'-sum(X_i*B_i)|^2  + lam1*sum(w_i*|A_i|_*) (+0.5*lam0*sum(w_i^2*|B_i|^2_F))
    s.t.  A_i=B_i

    Parameters:
    -----------
    :param Y: (n,q) continuous response data array
    :param X: K-length list, each element is a (n,p_i) predictor data matrix
               Note: X1,...XK may need some sort of standardization, because
               we use a single lam0 and lam1 for different predictor sets.
               Namely, we implicitly assume the coefficients are comparable.
    :param lam1: positive scalar, tuning for nuclear norm
    :param params: dict
           lam0     tuning for the ridge penalty, default=0

           weight   (K,1) weight vector, default: a vector of 1;
                    By theory, we should use w(i)=(1/n)*max(svd(X{i}))*(sqrt(q)+sqrt(rank(X{i}))); where X is column centered
                    Heuristically, one could also use w(i)=|X_i|_F

           randomstart   Boolean (default: False)

           varyrho  False=fixed rho (default); True=adaptive rho

           maxrho   5 (default): max rho. Unused if varyrho==0

           rho      initial step size, default rho=0.1

           Tol      default 1E-3,

           Niter	default 500

           fig      True (default) show checking figures; False no show

    Returns:
    --------
    C : (sum(p_i),q) coefficient matrix, potentially low-rank

    mu : (q,1) intercept vector (mean(Y,1)-mean(X,1)*hat{C})'
            if X and Y are non-missing and column centered, mu=zeros(q,1)

    A : list of length K, separate low-rank coefficient matrices

    B : list of length K, separate coefficient matrices

    Theta : list of length K, Lagrange parameter matrices


    Translated from iRRR_normal3.m 7/2020 by Emily Stephen
    Matlab code by Gen Li
    """

    K = len(X)

    ### Process params
    if params is None:
        params = {}

    lam0 = params.get('lam0',0)
    weight = params.get('weight',np.ones((K,1)))
    randomstart = params.get('randomstart',False)
    varyrho = params.get('varyrho',False)
    maxrho = params.get('maxrho',5)
    rho = params.get('rho',0.1)
    tol = params.get('Tol',1e-3) # stopping rule
    Niter = params.get('Niter',500) # Max iterations,
    blnFig = params.get('fig',True)

    n,q = Y.shape
    p = np.array([Xk.shape[1] for Xk in X])
    cumsum_p = np.concatenate([[0],np.cumsum(p)])
    assert(all([Xk.shape[0]==n for Xk in X]))

    # column center Xk's and normalize by the weights
    meanX = [Xk.mean(0,keepdims=True) for Xk in  X]
    X = [(Xk-mx)/w for Xk,mx,w in zip(X,meanX,weight)]
    cX = np.hstack(X)
    meanX = np.hstack(meanX)

    ### initial parameter estimates
    mu = np.nanmean(Y,axis=0,keepdims=True).T # (q x 1)

    # get a working Y by filling in Nans with best estimate
    wY,wY1,mu = _majorize_Y(Y,np.ones((n,1))@mu.T)

    if randomstart:
        B = [randn((pk,q)) for pk in p]
    else:
        B = [pinv(Xk.T @ Xk) @ Xk.T @ wY1 for Xk in X] # OLS

    Theta = [np.zeros((pk,q)) for pk in p] # Lagrange parameters for B
    cB = np.vstack(B) # vertically concatenated B

    A = B.copy()
    cA = cB.copy()
    cTheta = np.zeros((sum(p),q))

    _,D_cX,Vh_cX = svd((1/np.sqrt(n))*cX,full_matrices=False)
    if not varyrho: # fixed rho
        DeltaMat = Vh_cX.T @ np.diag(1/(D_cX**2+lam0+rho)) @ Vh_cX + \
            (np.eye(sum(p)) - Vh_cX.T @ Vh_cX)/(lam0+rho)   # inv(1/n*X'X+(lam0+rho)I)

    # compute initial objective values
    obj = [_objValue(Y,X,mu,A,lam0,lam1),  # full objective function (with penalties) on observed data
           _objValue(Y,X,mu,A,0,0)] # only the least square part on observed data


    ################
    # ADMM
    niter = 0
    diff = np.inf
    rec_obj = np.zeros((Niter+1,2)) # record objective values
    rec_obj[0,:] = obj
    rec_Theta = np.zeros((Niter)) # record Frobenius norm of Theta
    rec_primal = np.zeros((Niter)) # record total primal residual
    rec_dual = np.zeros((Niter)) # record total dual residual
    while niter<Niter and np.abs(diff)>tol:
        cB_old = cB.copy()

        # update working Y and mu
        wY,wY1,mu = _majorize_Y(Y,np.ones((n,1))@mu.T + cX@cB) # fill with current linear predictor

        # estimate concatenated B
        if varyrho:
            DeltaMat = Vh_cX.T @ np.diag(1/(D_cX**2+lam0+rho)) @ Vh_cX + \
                (np.eye(sum(p)) - Vh_cX.T @ Vh_cX)/(lam0+rho)
        cB = DeltaMat@((1/n)*cX.T@wY1 + rho*cA + cTheta)

        # partition cB into components
        B = [cB[cp:nextcp,:] for cp,nextcp in zip(cumsum_p[:-1],cumsum_p[1:])]

        # estimate each Ak and update Theta todo parallelize?
        for k,(Bk,Thetak) in enumerate(zip(B,Theta)):
            temp = Bk-Thetak/rho
            [tempU,tempD,tempVh] = svd(temp,full_matrices=False)
            A[k] = tempU @ np.diag(_softThres(tempD,lam1/rho)) @ tempVh
            Theta[k] = Theta[k]+rho*(A[k]-Bk)

        # update cA and cTheta
        for cp,nextcp,Ak,Thetak in zip(cumsum_p[:-1],cumsum_p[1:],A,Theta):
            cA[cp:nextcp,:] = Ak
            cTheta[cp:nextcp,:] = Thetak

        # update rho
        if varyrho:
            rho = min(maxrho,1.1*rho) # steadily increasing rho

        # check residuals
        primal = norm(cA-cB,ord='fro')**2
        rec_primal[niter] = primal
        dual = norm(cB-cB_old,ord='fro')**2
        rec_dual[niter] = dual

        # check objective values
        obj = [_objValue(Y,X,mu,A,lam0,lam1),  # full objective function (with penalties) on observed data
               _objValue(Y,X,mu,A,0,0)] # only the least square part on observed data
        rec_obj[niter+1,:] = obj

        # stopping rule
        diff = primal #max(primal,rho*dual)

        # plot
        if blnFig:
            # obj fcn values
            fig = plt.figure(1,figsize=[14,7.5])
            fig.clf()
            ax = fig.add_subplot(221)
            ax.plot(np.arange(niter+2),rec_obj[:niter+2,0],'bo-',label='Full Obj Value')
            ax.plot(np.arange(niter+2),rec_obj[:niter+2,1],'ro-',label='LS Obj Value')
            ax.legend()
            ax.set_title(f'Objective function value (decrease in full={rec_obj[niter,0]-rec_obj[niter+1,0]:.4f})')

            # primal and dual residuals
            ax1 = fig.add_subplot(223)
            ax1.plot(np.arange(niter+1)+1,rec_primal[:niter+1],'o-')
            ax1.set_title(f'|A-B|^2: {primal:.4f}')
            ax2 = fig.add_subplot(224)
            ax2.plot(np.arange(niter+1)+1,rec_dual[:niter+1],'o-')
            ax2.set_title(f'Dual residual |B-B|^2: {dual:.4f}')

            ax = fig.add_subplot(222)
            rec_Theta[niter] = norm(Theta[0],ord='fro')
            ax.plot(np.arange(niter+1)+1,rec_Theta[:niter+1],'o-');
            ax.set_title('Theta: Lagrange multiplier for B1')

            fig.suptitle(f'Lambda1: {lam1:.4f}')

            plt.pause(0.05)

        niter = niter+1

    if niter==Niter:
        print(f'iRRR does NOT converge after {Niter} iterations!')
    else:
        print(f'iRRR converges after {niter} iterations.')

    # rescale parameter estimate, add back mean
    A = [Ak/w for Ak,w in zip(A,weight)]
    B = [Bk/w for Bk,w in zip(B,weight)]
    C = np.vstack(A)
    mu = (mu.T - meanX@C).T

    return C,mu,A,B,Theta


def _majorize_Y(Y,Eta):
    wY = Y.copy()
    wY[np.isnan(wY)] = Eta[np.isnan(wY)] # fill in the nans
    mu = np.nanmean(wY,axis=0,keepdims=True).T # new estimate of mu
    wY1 = wY-mu.T # column centered wY

    return wY,wY1,mu

def _objValue(Y,X,mu,B,lam0,lam1):
    # Calc 1/(2n)|Y-1*mu'-sum(Xi*Bi)|^2 + lam0/2*sum(|Bi|_F^2) + lam1*sum(|Bi|_*)
    # with column centered  Xi's and (potentially non-centered and missing) Y
    n,q = Y.shape
    K = len(X)
    obj = 0
    pred = np.ones((n,1))@mu.T
    for i in range(K):
        pred = pred+X[i]@B[i]
        obj = obj + lam0/2*norm(B[i],ord='fro')**2+lam1*sum(svdvals(B[i]));
    obj = obj+(1/(2*n))*np.nansum((Y-pred)**2)

    return obj

def _softThres(d,lam):
    # this function soft thresholds d
    # d is the array of singular values
    # lam is a positive threshold
    dout = d.copy()
    np.fmax(d-lam,0,where=d>0,out=dout)
    np.fmin(d+lam,0,where=d<0,out=dout)

    return dout


