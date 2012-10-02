#
# Author: Fabian Pedregosa <fabian@fseoane.net>
# License: BSD

import numpy as np


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

MAX_ITER = 100

def soft_threshold(a, b):
    """accepts vectors"""
    return np.sign(a) * np.fmax(np.abs(a) - b, 0)

def sparse_group_lasso(X, y, alpha, rho, groups, max_iter=MAX_ITER, step_size=.1, rtol=1e-6,
                verbose=False):
    """
    .5 * ||Xb - y||^2_2 + n_samples * (alpha * (1 - rho) * sum(sqrt(#j) * ||b_j||_2) + alpha * rho ||b_j||_1)


    Linear least-squares with l2/l1 regularization solver.

    Solves problem of the form:

    .5 * |Xb - y| + n_samples * alpha * Sum(w_j * |b_j|)

    where |.| is the l2-norm and b_j is the coefficients of b in the
    j-th group. This is commonly known as the `group lasso`.

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
    true solution.

    Returns
    -------
    x : array
    vector of coefficients

    References
    ----------
    "Efficient Block-coordinate Descent Algorithms for the Group Lasso",
    Qin, Scheninberg, Goldfarb

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

    for n_iter in range(max_iter):
        for i, g in enumerate(group_labels):
            w_i = w_new.copy() # XXX
            w_i[g] = 0.
            rk = y - np.dot(X, w_i)
            s = soft_threshold(np.dot(X[:, g].T, rk), alpha * rho) # XXX: update inplace    #X_residual = np.dot(X[:, g].T, np.dot(X, w_i)) - Xy[g] # X^t(y - X w)
            # .. step 2 ..
            if np.linalg.norm(s) <= (1 - rho) * alpha:
                w_new[g] = 0.
            else:
                # .. step 3 ..
                stp = step_size
                for _ in range(100):
                    stp *= .8
                    grad_l = np.dot(X[:, g].T, rk - np.dot(X[:, g], w_new[g])) # XXX update inplace
                    tmp = soft_threshold(w_new[g] + step_size * grad_l, step_size * rho * alpha)
                    w_new[g] = max(1 - step_size * (1 - rho) * alpha / np.linalg.norm(tmp), 0) * tmp
                assert np.isfinite(w_new[g]).all()

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
    coefs = sparse_group_lasso(X, y, alpha, 0., groups, verbose=True, step_size=.01)
    print('KKT conditions verified:', group_lasso_check_kkt(X, y, coefs, alpha, groups))

    X = np.random.randn(100, 1000)
    y = np.random.randn(100)
    groups = np.arange(X.shape[1]) // 100
    coefs = sparse_group_lasso(X, y, alpha, 0., groups, verbose=True, step_size=.001)
    print('KKT conditions verified:', group_lasso_check_kkt(X, y, coefs, alpha, groups))


