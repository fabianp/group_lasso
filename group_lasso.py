#
# Author: Fabian Pedregosa <fabian@fseoane.net>
# License: BSD
import math
import numpy as np
from scipy import linalg


MAX_ITER = 1000

def soft_threshold(a, b):
    # vectorized version
    return np.sign(a) * np.fmax(np.abs(a) - b, 0)


def sparse_group_lasso(X, y, alpha, rho, groups, max_iter=MAX_ITER, rtol=1e-6,
                verbose=False):
    """
    Linear least-squares with l2/l1 + l1 regularization solver.

    Solves problem of the form:

    (1 / (2 n_samples)) * ||Xb - y||^2_2 +
        [ (alpha * (1 - rho) * sum(sqrt(#j) * ||b_j||_2) + alpha * rho ||b_j||_1) ]

    where b_j is the coefficients of b in the
    j-th group. Also known as the `sparse group lasso`.

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
    _K = [K[group][:, group] for group in group_labels]

    for n_iter in range(max_iter):
        w_old = w_new.copy()
        perm = np.random.permutation(len(group_labels))
        X_residual = Xy - np.dot(K, w_new) # could be updated, but kernprof says it's peanuts
        for i in perm:
            group = group_labels[i]
            #import ipdb; ipdb.set_trace()
            p_j = math.sqrt(group.size)
            Kgg = _K[i]
            X_r_k = X_residual[group] + np.dot(Kgg, w_new[group])
            s = soft_threshold(X_r_k, alpha * rho)
            # .. step 2 ..
            if np.linalg.norm(s) <= (1 - rho) * alpha * p_j:
                w_new[group] = 0.
            else:
                # .. step 3 ..
                for _ in range(2 * group.size): # just a heuristic
                    grad_l =  - (X_r_k - np.dot(Kgg, w_new[group]))
                    tmp = soft_threshold(w_new[group] - step_size * grad_l, step_size * rho * alpha)
                    tmp *= max(1 - step_size * p_j * (1 - rho) * alpha / np.linalg.norm(tmp), 0)
                    delta = linalg.norm(tmp - w_new[group])
                    w_new[group] = tmp
                    if delta < 1e-3:
                        break

                assert np.isfinite(w_new[group]).all()

        norm_w_new = max(np.linalg.norm(w_new), 1e-10)
        if np.linalg.norm(w_new - w_old) / norm_w_new < rtol:
            #import ipdb; ipdb.set_trace()
            break
    return w_new



def group_lasso_check_kkt(A, b, x, alpha, groups):
    """Auxiliary function.
    Check KKT conditions for the group lasso

    Returns True if conditions are satisfied, False otherwise
    """
    from scipy import linalg
    group_labels = [groups == i for i in np.unique(groups)]
    alpha = alpha * A.shape[0]
    z = np.dot(A.T, np.dot(A, x) - b)
    safety_net = .1 # sort of tolerance
    for g in group_labels:
        alpha_g = alpha * np.sqrt(np.sum(g))
        if linalg.norm(x[g]) == 0:
            if not linalg.norm(z[g]) < alpha_g + safety_net:
                return False
        else:
            w = - alpha_g * x[g] / linalg.norm(x[g], 2)
            if not np.allclose(z[g], w, safety_net):
                return False
            return True
    return True # all zeros


if __name__ == '__main__':
    np.random.seed(0)
    alpha = 0.1

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

    X = np.random.randn(1000, 300)
    y = np.random.randn(1000)
    groups = np.arange(X.shape[1]) // 100
    coefs = sparse_group_lasso(X, y, alpha, 0., groups, verbose=True)
    print('KKT conditions verified:', group_lasso_check_kkt(X, y, coefs, alpha, groups))


