import numpy as np

def p(r, c, P):
    """
    Violations computation for a matrix M and is used in Greenkhorn and Stochastic Sinkhorn

    Parameters
    ----------
    M : matrix of dimension nxn
    r : simplex vector of dimension n
    c : simplex vector of dimension n

    Returns
    -------
    p_M : the marginal violations vector measures how far the matrix
    M is from the transport polytope. M is in the polytope if and only
    if all the entries of p_M are zero

    Paper Reference:

    Brahim Khalil Abid and Robert Gower. Stochastic algorithms for entropy-regularized optimal transport problems. In 
    Amos Storkey and Fernando Perez-Cruz, editors, Proceedings of the Twenty-First International Conference on 
    Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pages 1505–1512. 
    PMLR, 09–11 Apr2018.

    """
    r_P = np.sum(P, axis = 1) - r
    c_P = np.sum(P, axis = 0) - c
    v_P = np.concatenate((r_P, c_P), axis=None)
    return v_P

def dist(X, Y, squared=False):
    """
    Compute cost matrix where the cost of moving i to j is squared euclidean distance

    Parameters
    ----------
    X, Y : discrete measures

    Returns
    -------
    M : cost matrix of moving between points in these discrete measures.
    
    Follows implementation in POT.
    """
    a2 = np.einsum('ij,ij->i', X, X)
    b2 = np.einsum('ij,ij->i', Y, Y)

    c = -2 * np.dot(X, Y.T)
    c += a2[:, None]
    c += b2[None, :]

    c = np.maximum(c, 0)

    if not squared:
        c = np.sqrt(c)

    return c

def round(P, r, c):
    """
    Rounds a transportation matrix to the given polytope

    Parameters
    ----------
    P : the transportation matrix of dimension nxn
    r, c : the respective source and target measures parametrizing the polytope

    Returns
    -------
    P* : the rounded transportation matrix

    Follows implementation from:

    https://github.com/nazya/AAM/blob/main/ot.ipynb

    Paper Reference:
    Brahim Khalil Abid and Robert Gower. Stochastic algorithms for entropy-regularized optimal transport problems. In 
    Amos Storkey and Fernando Perez-Cruz, editors, Proceedings of the Twenty-First International Conference on 
    Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pages 1505–1512. 
    PMLR, 09–11 Apr2018.

    """
    X = r / np.sum(P, axis=1) # n * n
    X[X>1] = 1. # n
    F = X.reshape(-1, 1) * P # n * n * (n - 1)
    Y = c / np.sum(P, axis = 0) # n * n
    Y[Y>1] = 1.
    F = F*Y.reshape(1,-1)
    err_r = r - np.sum(F, axis=1)
    err_c = c - np.sum(F, axis=0)
    return F + np.outer(err_r, err_c) / np.sum(np.abs(err_r))

def objective(M, gamma, P):
    """
    The computation of the objective function of the regularized, discrete OT problem

    Parameters
    ----------

    M : cost matrix of dimension nxn
    gamma : constant coefficient on entropy of transportation matrix h(P)
    P : transportation matrix of dimension nxn

    Returns
    -------
    <M,P> + gamma*h(P) : the objective function of the regularized OT problem

    Follows implementation from:

    https://github.com/nazya/AAM/blob/main/ot.ipynb

    Paper Reference:

    Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances
    in neural information processing systems, 26:2292–2300, 2013.

    """
    n = M.shape[0]
    y = (P.reshape(-1)).copy()
    y[y == 0.] = 1.
    y = y.reshape(n, -1)
    return (M * P).sum() + gamma * (P * np.log(y)).sum()