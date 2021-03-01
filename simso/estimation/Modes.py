from numpy import array, dot, zeros
from math import log, sqrt, pi
from numpy.linalg import inv as inv_matrix
from scipy.stats import norm, gumbel_r as Gumbel
from sklearn.mixture import GaussianMixture


class Modes:

    def __init__(self, Mmax, alpha=1e-2):
        self.M = Mmax
        self.n = None
        self._vn, self._un = None, None
        self.gmm = None
        self.alpha = alpha
        self.cvar = None
        self.transmat = None

    def fit(self, X, y=None):
        """
        Compute parameters of a Gaussian mixture using l1 penalty on precision matrices, and estimating
        the copula coefficient of each pair wise copula
        y is useless
        """

        self.n = X.shape[1]

        # Identifying modes
        lowest_bic = 1e10
        mean_bic = []
        n_components_range = range(1, self.M)
        K = 10
        bics = zeros((K, len(n_components_range)))

        for l, n_components in enumerate(n_components_range):
            print(n_components)
            for k in range(K):
                gmm = GaussianMixture(n_components=n_components, covariance_type='full').fit(X)
                bics[k, l] = gmm.bic(X)
            if bics[:, l].mean() < lowest_bic:
                lowest_bic = bics[:, l].mean()
                self.M = n_components
                self.gmm = gmm
        print('final M={}'.format(self.M))

        self.gmm = GaussianMixture(self.M, covariance_type='full').fit(X)
        y = self.gmm.predict(X)

        tmp = zeros((self.M, self.M))
        for i, j in zip(y[:-1], y[1:]):
            tmp[i, j] += 1
        self.transmat = array([tmp[i, :] / tmp[i, :].sum() for i in range(self.M)])

        self._vn = sqrt(2 * log(1e6))
        self._un = self._vn - log(4 * pi * log(1e6)) / (2 * self._vn)

        self.cvar = zeros((self.M, self.n))

        for m in range(self.M):
            for k in range(self.n):
                self.cvar[m, k] = self.gmm.means_[m, k] + \
                                  sqrt(self.gmm.covariances_[m, k, k]) * norm.pdf(norm.ppf(self.alpha)) / (1 - self.alpha)

        print('This may take a while...')

        return self

    def conditional_dmp(self, X, k, I=None):
        if not all(X):
            return 0
        X = array(X).reshape(1, -1)
        m_star = self.gmm.predict(X)
        dmp = 0

        for m in range(self.M):
            muk_X, sigmak_X = self._condition(m, k, X, I)
            uk_X = muk_X + sqrt(sigmak_X) * self._un
            vk_X = sqrt(sigmak_X) / self._vn
            dmp += self.transmat[m_star, m] * \
                   self.gmm.weights_[m] * Gumbel.cdf(self.cvar[m, k] if vk_X > 0.05 else uk_X, uk_X, vk_X)
        return dmp

    def _condition(self, m, k, X, I):

        if I and len(I) > 1:
            ind_X = I
        elif len(I) == 1:
            ind_X = I[0]
        else:
            ind_X = list(range(k)) + list(range(k + 1, self.n))

        mu, sigma = array(self.gmm.means_[m]), array(self.gmm.covariances_[m])
        sigma_X = sigma[ind_X][:, ind_X]
        if len(ind_X) > 1:
            a = dot(sigma[ind_X, k], inv_matrix(sigma_X))
            muk_X = dot(a, X[0, ind_X] - mu[ind_X]) + mu[k]
            sigmak_X = sigma[k, k] - dot(a, sigma[k, ind_X])
        elif len(ind_X) == 1:
            a = sigma[k, ind_X] / sigma_X
            muk_X = a * (X[ind_X] - mu[ind_X]) + mu[k]
            sigmak_X = sigma[k, k] - a * sigma[k, ind_X]
        return muk_X, sigmak_X