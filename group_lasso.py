#
# Author: Fabian Pedregosa <fabian@fseoane.net>
# License: BSD
import random
import numpy as np
from scipy import linalg


## .. dual gap ..
#max_inc = linalg.norm(w_old - w_new, np.inf)
#if max_inc < rtol * np.amax(w_new):
#    residual = np.dot(X, w_new) - y
#    group_norm = alpha * np.sum([linalg.norm(w_new[g], 2)
#                                 for g in group_labels])
#    if H is not None:
#        norm_Anu = [linalg.norm(np.dot(H[g], w_new) - Xy[g])\
#                    for g in group_labels]
#    else:
#        # TODO: Uses H !!!!
#        norm_Anu = [linalg.norm(np.dot(X[:, g].T, residual))\
#                    for g in group_labels]
#    if np.any(norm_Anu > alpha):
#        nnu = residual * np.min(alpha / norm_Anu)
#    else:
#        nnu = residual
#    primal_obj =  .5 * np.dot(residual, residual) + group_norm
#    dual_obj   = -.5 * np.dot(nnu, nnu) - np.dot(nnu, y)
#    dual_gap = primal_obj - dual_obj
#    if verbose:
#        print 'Relative error: %s' % (dual_gap / dual_obj)
#    if np.abs(dual_gap / dual_obj) < rtol:
#        break

MAX_ITER = 1000

def soft_threshold(a, b):
    """accepts vectors"""
    return np.sign(a) * np.fmax(np.abs(a) - b, 0)

#@profile
def sparse_group_lasso(X, y, alpha, rho, groups, max_iter=MAX_ITER, rtol=1e-6,
                verbose=False):
    """
    Linear least-squares with l2/l1 regularization solver.

    Solves problem of the form:

    .5 * ||Xb - y||^2_2 + n_samples * (alpha * (1 - rho) * sum(sqrt(#j) * ||b_j||_2) + alpha * rho ||b_j||_1)

    where b_j is the coefficients of b in the
    j-th group. Also known as the `group lasso`.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Design Matrix.

    y : array of shape (n_samples,)

    alpha : float or array
        Amount of penalization to use.

    groups : array of shape (n_features,)
        Group label. For each column, it indicates
        its group apertenance.

    rtol : float
        Relative tolerance. ensures ||(x - x_) / x_|| < rtol,
        where x_ is the approximate solution and x is the
        true solution. TODO duality gap

    Returns
    -------
    x : array
        vector of coefficients

    References
    ----------
    "A sparse-group lasso", Noah Simon et al.
    """
    # .. local variables ..
    X, y, groups, alpha = map(np.asanyarray, (X, y, groups, alpha))
    if groups.shape[0] != X.shape[1]:
        raise ValueError('Groups should be of shape %s got %s instead' % ((X.shape[1],), groups.shape))
    w_new = np.zeros(X.shape[1], dtype=X.dtype)
    n_samples = X.shape[0]
    alpha = alpha * n_samples

    # .. use integer indices for groups ..
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    Xy = np.dot(X.T, y)
    K = np.dot(X.T, X)
    step_size = 1. / (linalg.norm(X, 2) ** 2)

    for n_iter in range(max_iter):
        w_old = w_new.copy()
        random.shuffle(group_labels)
        for i, g in enumerate(group_labels):
            w_tmp = w_new.copy()
            w_tmp[g] = 0.
            Kg = K[g]
            X_residual = Xy[g] - np.dot(Kg, w_tmp)
            residual = y - np.dot(X, w_tmp)
            s = soft_threshold(X_residual, alpha * rho)
            # .. step 2 ..
            if np.linalg.norm(s) <= (1 - rho) * alpha:
                w_new[g] = 0.
            else:
                # .. step 3 ..
                for _ in range(3 * g.size): # just a heuristic
                    grad_l =  - (X_residual - np.dot(Kg[:, g], w_new[g]))
                    tmp = soft_threshold(w_new[g] - step_size * grad_l, step_size * rho * alpha)
                    tmp *= max(1 - step_size * (1 - rho) * alpha / np.linalg.norm(tmp), 0)
                    delta = linalg.norm(tmp - w_new[g])
                    w_new[g] = tmp
                    if delta < 1e-3:
                        break

                assert np.isfinite(w_new[g]).all()

        if np.linalg.norm(w_new - w_old) / np.linalg.norm(w_new) < rtol:
            break
    return w_new



def group_lasso_check_kkt(A, b, x, penalty, groups):
    """Auxiliary function.
    Check KKT conditions for the group lasso

    Returns True if conditions are satisfied, False otherwise
    """
    from scipy import linalg
    group_labels = [groups == i for i in np.unique(groups)]
    penalty = penalty * A.shape[0]
    z = np.dot(A.T, np.dot(A, x) - b)
    safety_net = .5 # sort of tolerance
    for g in group_labels:
        if linalg.norm(x[g]) == 0:
            if not linalg.norm(z[g]) < penalty + safety_net:
                return False
        else:
            w = - penalty * x[g] / linalg.norm(x[g], 2)
            if not np.allclose(z[g], w, safety_net):
                return False
            return True


if __name__ == '__main__':
    np.random.seed(0)
    alpha = .1

    from sklearn import datasets
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    groups = np.arange(X.shape[1]) // 5
    coefs = sparse_group_lasso(X, y, alpha, 0., groups, verbose=True)
    print('KKT conditions verified:', group_lasso_check_kkt(X, y, coefs, alpha, groups))

    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    groups = np.arange(X.shape[1]) // 10
    coefs = sparse_group_lasso(X, y, alpha, 0., groups, verbose=True)
    print('KKT conditions verified:', group_lasso_check_kkt(X, y, coefs, alpha, groups))

    X = np.random.randn(100, 1000)
    y = np.random.randn(100)
    groups = np.arange(X.shape[1]) // 100
    coefs = sparse_group_lasso(X, y, alpha, 0., groups, verbose=True)
    print('KKT conditions verified:', group_lasso_check_kkt(X, y, coefs, alpha, groups))


