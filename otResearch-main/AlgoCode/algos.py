import numpy as np
import time
import warnings
from util import p,round,objective

def sinkhorn(r, c, M, reg, acc, maxIter=1000000):
    """
    Sinkhorn Algorithm

    Parameters
    ----------
    r : source measure
    c : target measure
    M : cost matrix for OT problem
    reg : regularization term >0
    acc : the desired accuracy
    k : max iterations

    Returns
    -------
    P : an nxn matrix that is the approximated solution 
    matrix P to the OT problem

    t : the time to complete the algorithm in seconds

    oper_log : the number of arithmetic operations to complete the algorithm

    This algorithm mostly follows the format of the Sinkhorn-Knopp algorithm
    already found in the POT package.

    Paper References:

    Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances
    in neural information processing systems, 26:2292–2300, 2013.

    Jason Altschuler, Jonathan Weed, and Philippe Rigollet. Near-linear time approximation algorithms for optimal 
    transport via Sinkhorn iteration. In Proceedings of the 31st International Conference on Neural Information 
    Processing Systems, pages 1961–1971, 2017.

    Pavel Dvurechensky, Alexander Gasnikov, and Alexey Kroshnin. Computational optimal transport: Complexity by 
    accelerated gradient descent is better than by Sinkhorn’s algorithm. In International conference on machine 
    learning, pages 1367–1376. PMLR, 2018.
    """
    time0 = time.time()
    n = M.shape[0]
    m = M.shape[1]
    oper_log = 0
    P = np.zeros((n,n))
    K = np.exp(- M / reg)
    oper_log += 2 * n * n

    Kp = (1/r)[:,None] * K
    oper_log += n + n * n

    u = np.full((M.shape[0],), 1 / M.shape[0])
    v = np.full((M.shape[1],), 1 / M.shape[1])
    oper_log += 2 * n + 2
    for _ in range(maxIter):
        uprev = u
        vprev = v

        KtransposeU = np.dot(K.T, u)
        oper_log += 2 * n * n - n

        v = c / KtransposeU
        oper_log += n

        u = 1. / np.dot(Kp, v)
        oper_log += 3 * n - 1

        if (np.any(KtransposeU == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            u = uprev
            v = vprev
            break

        P = u[:, None] * K * v[None, :]
        oper_log += 2 * n * n

        if np.allclose(r, P.sum(1), atol=acc) and np.allclose(c, P.sum(0), atol=acc):
            return P, time.time() - time0, oper_log
    return P, time.time() - time0, oper_log

def greenkhorn(r, c, M, reg, acc, maxIter=1000000):
    """
    Greedy Sinkhorn Algorithm

    Parameters
    ----------
    r : source measure
    c : target measure
    M : cost matrix for OT problem
    reg : regularization term >0
    acc: desired accuracy or distance between the solution and the polytope

    Returns
    -------
    P : an dxd matrix that is the approximated solution 
    matrix P to the OT problem.

    t : the time to complete the algorithm in seconds

    oper_log : the number of arithmetic operations to complete the algorithm 

    Followed from Greenkhorn in POT.

    Paper References:

    Jason Altschuler, Jonathan Weed, and Philippe Rigollet. Near-linear time approximation algorithms for optimal 
    transport via Sinkhorn iteration. In Proceedings of the 31st International Conference on Neural Information 
    Processing Systems, pages 1961–1971, 2017.

    Brahim Khalil Abid and Robert Gower. Stochastic algorithms for entropy-regularized optimal transport problems. In 
    Amos Storkey and Fernando Perez-Cruz, editors, Proceedings of the Twenty-First International Conference on 
    Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pages 1505–1512. 
    PMLR, 09–11 Apr2018.

    """
    time0 = time.time()
    n = M.shape[0]
    oper_log = 0
    
    K = np.exp(- M / reg)
    oper_log += 2 * n * n

    u = np.full((M.shape[0],), 1 / n)
    v = np.full((M.shape[1],), 1 / n)
    oper_log += 2 * n + 2

    P = u[:, None] * K * v[None, :]
    oper_log += 2 * n * n

    for _ in range(maxIter):
        viol = p(r, c, P)
        oper_log += 2 * n * (n + 1)

        i = np.argmax(np.abs(viol))
        oper_log += 2 * n

        if i < n:
            new_u = r[i] / np.dot(K[i,:], v)
            P[i,:] = new_u * K[i,:] * v
            viol[i] = P[i,:].sum() - r[i]
            viol[n:] += (K[i, :].T * (new_u - u[i]) * v)
            u[i] = new_u
        else:
            i -= n
            oper_log += 1
            new_v = c[i] / np.dot(K[:, i].T, u)
            P[:, i] = u * K[:, i] * new_v
            viol[:n] += (new_v - v[i]) * K[:, i] * u
            viol[i+n] = P[:,i].sum() - c[i]
            v[i] = new_v
        # Note the number of arithmetic in each block is 8 * n except the
        # else block has an extra 1 operation for i -= n.
        oper_log += 8 * n
        if np.allclose(r, P.sum(1), atol=acc) and np.allclose(c, P.sum(0), atol=acc):
            return P, time.time() - time0, oper_log

    return P, time.time() - time0, oper_log


def stochSinkhorn(r, c, M, reg, acc, maxIter=1000000):
    """
    Stochastic Sinkhorn Algorithm

    Parameters
    ----------
    r : source measure
    c : target measure
    M : cost matrix for OT problem
    reg : regularization term >0
    acc: desired accuracy or distance between the solution and the polytope

    Returns
    -------
    P : an dxd matrix that is the approximated solution 
    matrix P to the OT problem

    t : the time to complete the algorithm in seconds

    oper_log : the number of arithmetic operations to complete the algorithm
    
    Similar to Stochastic Sinkhorn in POT.

    Paper References:

    Brahim Khalil Abid and Robert Gower. Stochastic algorithms for entropy-regularized optimal transport problems. In 
    Amos Storkey and Fernando Perez-Cruz, editors, Proceedings of the Twenty-First International Conference on 
    Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pages 1505–1512. 
    PMLR, 09–11 Apr2018.

    """
    np.random.seed(123)
    time0 = time.time()
    n = M.shape[0]
    oper_log = 0

    K = np.exp(- M / reg)
    oper_log +=  2 * n * n

    u = np.full((M.shape[0],), 1 / M.shape[0])
    v = np.full((M.shape[1],), 1 / M.shape[1])
    oper_log += 2 * n + 2

    P = u[:, None] * K * v[None, :]
    oper_log += 2 * n * n

    for _ in range(maxIter):
        viol = p(r, c, P)
        oper_log += 2 * n * (n + 1)

        g = np.abs(viol)
        oper_log += 2 * n

        g = g / g.sum()
        oper_log += 4 * n - 1

        i = np.random.multinomial(1, g).argmax()
        oper_log += 2 * n

        if i < n:
            new_u = r[i] / np.dot(K[i,:], v)
            P[i,:] = new_u * K[i,:] * v
            viol[i] = P[i,:].sum() - r[i]
            viol[n:] += (K[i, :].T * (new_u - u[i]) * v)
            u[i] = new_u
        else:
            i = i-n
            oper_log += 1

            new_v = c[i] / np.dot(K[:, i].T, u)
            P[:, i] = u * K[:, i] * new_v
            viol[:n] += (new_v - v[i]) * K[:, i] * u
            viol[i+n] = P[:,i].sum() - c[i]
            v[i] = new_v
        oper_log += 8 * n
        if np.allclose(r, P.sum(1), atol=acc) and np.allclose(c, P.sum(0), atol=acc):
            return P, time.time() - time0, oper_log

    return P, time.time() - time0, oper_log

def sag(r, c, M, reg, eps, maxIter=100000, lr=None):
    """
    Stochastic Averaged Gradient Algorithm

    Parameters
    ----------
    r : source measure
    c : target measure
    M : cost matrix for OT problem
    reg : regularization term >0
    k : max iterations
    lr : learning rate

    Returns
    -------
    P : an dxd matrix that is the approximated solution 
    matrix P to the OT problem

    t : the time to complete the algorithm in seconds

    oper_log : the number of arithmetic operations to complete the algorithm
    
    Similar to SAG in POT.

    Paper References:

    Mark Schmidt, Nicolas Le Roux, and Francis Bach. Minimizing finite sums with the stochastic average gradient, 2013.

    Aude Genevay, Marco Cuturi, Gabriel Peyr´e, and Francis Bach. Stochastic optimization for large-scale optimal 
    transport. In NIPS 2016-Thirtieth Annual Conference on Neural Information Processing System, 2016.

    """
    np.random.seed(123)
    time0 = time.time()
    n = M.shape[0]
    oper_log = 0

    if not lr:
        lr = 1. / np.max(r / reg)
        oper_log = 3 * n

    v = np.zeros(n)
    d = np.zeros(n)
    oper_log += 2 * n

    stored_grad = np.zeros((n,n))
    oper_log += n * n

    for k in range(maxIter):
        idx = np.random.randint(n)

        expo = np.exp(-1. * (M[idx,:] - v) / reg) * c
        oper_log += 4 * n

        term = expo / np.sum(expo)
        oper_log += 2 * n - 1

        grad = r[idx] * (c - term)
        oper_log += 2 * n

        d += grad - stored_grad[idx]
        oper_log += 2 * n

        stored_grad[idx] = grad
        oper_log += 1

        v += (lr / n) * d
        oper_log += n + 1
        if k % 1000 == 0:
            n = M.shape[0]
            # Get other dual variable
            u = np.zeros(n)
            oper_log += n
            for i in range(n):
                a = M[i,:] - v
                oper_log += n
                min_a = np.min(a)
                oper_log += n
                exp_v = np.exp(-(a - min_a) / reg) * c
                oper_log += 5 * n
                u[i] = min_a - reg * np.log(np.sum(exp_v))
                oper_log += n + 2
            
            # P is computation matrix
            P = (np.exp((u[:,None] + v[None, :] - M[:,:]) / reg) * r[:,None] * c[None, :])
            oper_log += 6 * n * n
            if np.allclose(r, P.sum(1), atol=eps) and np.allclose(c, P.sum(0), atol=eps):
                return P, time.time() - time0, oper_log
    n = M.shape[0]
    # Get other dual variable
    u = np.zeros(n)
    oper_log += n
    for i in range(n):
        a = M[i,:] - v
        oper_log += n
        min_a = np.min(a)
        oper_log += n
        exp_v = np.exp(-(a - min_a) / reg) * c
        oper_log += 5 * n
        u[i] = min_a - reg * np.log(np.sum(exp_v))
        oper_log += n + 2
    
    # P is computation matrix
    P = (np.exp((u[:,None] + v[None, :] - M[:,:]) / reg) * r[:,None] * c[None, :])
    oper_log += 6 * n * n
    return P, time.time() - time0, oper_log



def apdagd(r, c, M, eps):
    """
    Adaptive Primal-Dual Accelerated Gradient Descent

    Parameters
    ----------
    r : source measure
    c : target measure
    M : cost matrix for OT problem
    eps : desired accuracy or distance between the solution and the polytope

    Returns
    -------
    P : an dxd matrix that is the approximated solution 
    matrix P to the OT problem

    t : the time to complete the algorithm in seconds

    oper_log : the number of arithmetic operations to complete the algorithm
    
    References
    ----------
    Follows implementation from:

    https://github.com/nazya/AAM/blob/main/ot.ipynb

    Paper References:

    ** Note this is the paper associated with the GitHub link above **
    Sergey Guminov, Pavel Dvurechensky, Nazarii Tupitsa, and Alexander Gasnikov. On a combination of alternating
    minimization and Nesterov's momentum. In International Conference on Machine Learning, pages 3886-3698. PMLR, 2021.

    Pavel Dvurechensky, Alexander Gasnikov, and Alexey Kroshnin. Computational optimal transport: Complexity by 
    accelerated gradient descent is better than by Sinkhorn’s algorithm. In International conference on machine 
    learning, pages 1367–1376. PMLR, 2018.
    """
    warnings.filterwarnings('ignore')
    oper_log = 0
    time0 = time.time()
    n = M.shape[0]

    epsp = eps / 8
    oper_log += 1

    p, q = (1 - epsp / 8) * r + epsp / (8 * n), (1 - epsp / 8) * c + epsp / (8 * n)
    oper_log += 4 * n + 8

    gamma = eps / (3 * np.log(n))
    oper_log += 3

    K = - M / gamma
    oper_log += n * n

    L = 1
    xi = np.zeros(2 * n)
    eta = xi.copy()
    z = xi.copy()
    eta_new = xi.copy()
    z_new = xi.copy()
    grad_phi_new = xi.copy()
    betta = 0
    one = np.ones(n)
    primal_var = np.zeros((n, n))
    while True:

        L = L / 2
        oper_log += 1

        while True:
            alpha_new = (1 + np.sqrt(4*L*betta + 1)) / 2 / L
            oper_log += 7

            betta_new = betta + alpha_new
            oper_log += 1

            tau = alpha_new / betta_new
            oper_log += 1

            lamu_new = tau * z + (1 - tau) * eta
            oper_log += 6 * n + 1
                    
            logB = (K + np.outer(lamu_new[:n], one) + np.outer(one, lamu_new[n:]))
            oper_log += 4 * n * n

            max_logB =logB.max()
            oper_log += n * n

            logB_stable = logB - max_logB
            oper_log += n * n

            B_stable = np.exp(logB_stable)
            oper_log += n * n

            u_hat_stable, v_hat_stable = B_stable.dot(one), B_stable.T.dot(one)
            oper_log += 2 * n * (n - 1)
            
            Bs_stable = u_hat_stable.sum()
            oper_log += n - 1

            phi_new = gamma*(-lamu_new[:n].dot(p) - lamu_new[n:].dot(q) + np.log(Bs_stable) + max_logB)
            oper_log += 2 * n * (n + 1)

            grad_phi_new = gamma*np.concatenate((-p + u_hat_stable/Bs_stable, -q + v_hat_stable/Bs_stable),0)
            oper_log += 8 * n
            
            z_new = z - alpha_new * grad_phi_new
            oper_log += 4 * n

            eta_new = tau * z_new + (1-tau) * eta
            oper_log += 6 * n + 1

            logE = (K + np.outer(eta_new[:n], one) + np.outer(one, eta_new[n:]))
            oper_log += 4 * n * n

            max_logE =logE.max()
            oper_log += n * n

            logE_stable = logE - max_logE
            oper_log += n * n

            E_stable = np.exp(logE_stable)
            oper_log += n * n

            u_hat_stable, v_hat_stable = E_stable.dot(one), E_stable.T.dot(one)
            oper_log += 2 * n * (n - 1)
            
            Es_stable = u_hat_stable.sum()
            oper_log += n - 1

            phi_eta = gamma*(-eta_new[:n].dot(p) - eta_new[n:].dot(q) + np.log(Es_stable) + max_logE)
            oper_log += 2 * n * (n + 1)

            diff = eta_new - lamu_new
            oper_log += 2 * n

            oper_log += 8 * n + 1
            if phi_eta <= phi_new + grad_phi_new.dot(diff) + L * (diff * diff).sum() / 2:
                betta = betta_new
                z = z_new.copy()
                eta = eta_new.copy()
                break

            L = L * 2
            oper_log += 1

        primal_var = tau * B_stable/Bs_stable + (1 - tau) * primal_var
        oper_log += 3 * n * n + 1
        
        oper_log += 3 * n * n - n + 2
        if (M * (round(primal_var, p, q) - primal_var)).sum() <= eps/6 and abs(objective(M, gamma, primal_var) + phi_eta) <= eps/6:
            return round(primal_var, r, c), time.time() - time0, oper_log
  
def aam(r, c, M, eps):
    """
    Adaptive Alternating Minimization

    Parameters
    ----------
    r : source measure
    c : target measure
    M : cost matrix for OT problem
    eps : desired accuracy or distance between the solution and the polytope

    Returns
    -------
    P : an dxd matrix that is the approximated solution 
    matrix P to the OT problem

    t : the time to complete the algorithm in seconds

    oper_log : the number of arithmetic operations to complete the algorithm
    
    References
    ----------
    Follows implementation from:

    https://github.com/nazya/AAM/blob/main/ot.ipynb

    Paper Reference:

    Sergey Guminov, Pavel Dvurechensky, Nazarii Tupitsa, and Alexander Gasnikov. On a combination of alternating
    minimization and Nesterov's momentum. In International Conference on Machine Learning, pages 3886-3698. PMLR, 2021.

    """
    warnings.filterwarnings('ignore')
    n = M.shape[0]
    oper_log = 0
    time0 = time.time()

    epsp = eps / 8
    oper_log += 1

    p, q = (1 - epsp / 8) * r + epsp / (8 * n), (1 - epsp / 8) * c + epsp / (8 * n)
    oper_log += 4 * n + 8

    gamma = eps / (3 * np.log(n))
    oper_log += 3

    K = - M / gamma
    oper_log += n * n

    L = 1
    step = 2
    xi = np.zeros(2 * n)
    eta = xi.copy()
    zeta = xi.copy()
    alpha_new = alpha = 0
    ustep = np.zeros(n)
    vstep = np.zeros(n)
    one = np.ones(n)
    primal_var = np.zeros((n, n))

    while True:

        L_new = L / step
        oper_log += 1

        while True:
            alpha_new = 1/2/L_new + np.sqrt(1/4/L_new/L_new + alpha* alpha * L / L_new)
            oper_log += 11

            tau = 1 / alpha_new / L_new
            oper_log += 2

            xi = tau * zeta + (1 - tau) * eta
            oper_log += 6 * n + 1

            logB = (K + np.outer(xi[:n], one) + np.outer(one, xi[n:]))

            max_logB = logB.max()
            oper_log += n * n

            logB_stable = logB - max_logB
            oper_log += n * n

            B_stable = np.exp(logB_stable)
            oper_log += n * n

            u_hat_stable, v_hat_stable = B_stable.dot(one), B_stable.T.dot(one)
            oper_log += 2 * n * (n - 1)
            
            Bs_stable = u_hat_stable.sum()
            oper_log += n - 1

            f_xi = gamma*(-xi[:n].dot(p) - xi[n:].dot(q) + np.log(Bs_stable) + max_logB)
            oper_log += 2 * n * (n + 1)

            grad_f_xi = gamma*np.concatenate((-p + u_hat_stable/Bs_stable, -q + v_hat_stable/Bs_stable),0)
            oper_log += 8 * n

            gu, gv = (grad_f_xi[:n]**2).sum(), (grad_f_xi[n:]**2).sum()
            oper_log += 2 * n * (n - 1)

            norm2_grad_f_xi = (gu+gv)
            oper_log += 1

            if gu > gv:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    try:
                        ustep = p/u_hat_stable
                        oper_log += n
                    except Warning as e:
                        u_hat_stable/=u_hat_stable.max()
                        oper_log += 2 * n

                        u_hat_stable[u_hat_stable<1e-150] = 1e-150
                        oper_log += n

                        ustep = p/u_hat_stable
                        oper_log += n

                ustep/=ustep.max()
                oper_log += n

                xi[:n]+=np.log(ustep)
                oper_log += 2 * n

                Z=ustep[:,None]*B_stable
                oper_log += n * n
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    try:
                        vstep = q/v_hat_stable
                        oper_log += n
                    except Warning as e:
                        v_hat_stable/=v_hat_stable.max()
                        oper_log += 2 * n

                        v_hat_stable[v_hat_stable<1e-150] = 1e-150
                        oper_log += n

                        vstep = q/v_hat_stable
                        oper_log += n

                vstep/=vstep.max()
                oper_log += n

                xi[n:]+=np.log(vstep)
                oper_log += 2 * n

                Z=B_stable*vstep[None,:]
                oper_log += n * n

            f_eta_new=gamma*(np.log(Z.sum())+max_logB-xi[:n].dot(p)-xi[n:].dot(q))
            oper_log += 3 * n * (n - 1) + 5
            
            oper_log += 3
            if f_eta_new <= f_xi - (norm2_grad_f_xi)/2/L_new: # can be optimized 2 itmes
                primal_var = (alpha_new * B_stable/Bs_stable + L * alpha**2 * primal_var) /(L_new*alpha_new**2)
                oper_log += 4 * n * n + 5

                zeta -= alpha_new * grad_f_xi
                oper_log += 4 * n
                eta = xi.copy()
                alpha = alpha_new
                L = L_new
                
                break
            L_new*=step
            oper_log += 1

        oper_log += 3 * n * n - n + 2
        if (M * (round(primal_var, r, c) - primal_var)).sum() <= eps/6 and abs(objective(M, gamma, primal_var) + f_eta_new) <= eps/6:
            return round(primal_var, r, c), time.time() - time0, oper_log


def apdrcd(r, c, M, eps):
    """
    Adaptive Primal-Dual Randomized Coordinate descent

    Parameters
    ----------
    r : source measure
    c : target measure
    M : cost matrix for OT problem
    eps : desired accuracy or distance between the solution and the polytope

    Returns
    -------
    P : an dxd matrix that is the approximated solution 
    matrix P to the OT problem

    t : the time to complete the algorithm in seconds

    oper_log : the number of arithmetic operations to complete the algorithm
    
    Paper Reference:

    Wenshuo Guo, Nhat Ho, and Michael Jordan. Fast algorithms for computational optimal transport and Wasserstein 
    barycenter. In International Conference on Artificial Intelligence and Statistics, pages 2088–2097. PMLR, 2020.

    """
    oper_log = 0
    time0 = time.time()
    seed = 123
    np.random.seed(seed)
    n = M.shape[0]

    epsp = eps / (8)
    oper_log += n * n + 2

    gamma = eps / (4 * np.log(n))
    oper_log += 2

    p, q = (1 - epsp / 8) * r + (epsp / (8 * n)), (1 - epsp / 8) * c + (epsp / (8 * n))
    oper_log += 4 * n + 8

    one = np.ones(n)
    y = np.zeros(2*n)
    z = np.zeros(2*n)
    lmda = np.zeros(2 * n)
    primal_var = np.zeros((n, n))
    C = 1
    x_y = np.zeros((n,n))
    D = np.zeros((n, n))
    L = 4 / gamma
    oper_log += 1
    err = 100
    theta = 1
    i = 0
    while err >= epsp / 2:
        y[i] = (1 - theta) * lmda[i] + theta * z[i]
        oper_log += 4

        if i < n:
            x_y[i,:] = np.exp( (-M[i,:] + y[i] + y[n:]) / gamma - 1 )
        else:
            x_y[:,i-n] = np.exp( (-M[:,(i - n)] + y[i] + y[:n]) / gamma - 1 )
        oper_log += 5 * n

        D += x_y / theta
        oper_log += 2 * n * n

        primal_var = D / C
        oper_log += n * n
        i = np.random.choice(2 * n)
        if i < n:
            grad_phi = np.exp( (-M[i,:] + y[i] + y[n:]) / gamma - 1 ).sum() - p[i]
        else:
            grad_phi = np.exp( (-M[:,(i - n)] + y[i] + y[:n]) / gamma - 1 ).sum() - q[i - n]
        oper_log += 7 * n
        
        lmda[i] = y[i] - grad_phi / L
        oper_log += 2

        z[i] = z[i] - grad_phi / (2 * n * L * theta)
        oper_log += 5

        C = C + 1 / theta
        oper_log += 2

        theta = (-theta * theta + theta * np.sqrt(theta * theta + 4)) / 2
        oper_log += 8

        err = abs(primal_var.sum(1) - p).sum() + abs(primal_var.sum(0) - q).sum()
        oper_log += 2 * n * n + 2 * n - 1
    return round(primal_var, r, c), time.time() - time0, oper_log
