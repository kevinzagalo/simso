from numpy import array, dot, zeros, random
from math import log, sqrt, pi
from numpy.linalg import inv as inv_matrix
from scipy.stats import norm, gumbel_r as Gumbel, boxcox
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
#from hmmlearn import hmm


class Modes:

    def __init__(self, Mmax, alpha=1e-3):
        self.M = Mmax
        self.n = None
        self._vn, self._un = None, None
        self.gmm = None
        self.hmm = None
        self.alpha = alpha
        self.cvar = None
        self.transmat = None

    def fit(self, X, y=None):
        """
        Compute parameters of a Gaussian mixture using l1 penalty on precision matrices, and estimating
        the copula coefficient of each pair wise copula
        y is useless
        """
        for i, x in enumerate(X):
            if all(x):
                X = X[i:, :]
                break
        self.n = X.shape[1]
        self.alpha = [self.alpha] * self.n

        #import matplotlib.pyplot as plt
#
        #fig, ax = plt.subplots(self.n//2, 2)
        #for i, ax_ in enumerate(ax.flatten()):
        #    ax_.hist(X[:, i])
#
        #plt.show()

        #LAMBDA = []
#
        #tmp = zeros(X.shape)
        #for i in range(self.n):
        #    Xbox, lambd = boxcox(X[:, i])
        #    LAMBDA.append(lambd)
        #    tmp[:, i] = Xbox
#
        #X = tmp


        print('This may take a while...')


        # Identifying modes
        lowest_bic = 1e10
        mean_bic = []
        n_components_range = range(1, self.M)
        K = 10
        bics = zeros((K, len(n_components_range)))

        kf = KFold(n_splits=K, random_state=123, shuffle=True)

        for l, n_components in enumerate(n_components_range):
            for k, (train_ind, test_ind) in enumerate(kf.split(X)):
                gmm = GaussianMixture(n_components=n_components, covariance_type='full').fit(X[train_ind, :])
                bics[k, l] = gmm.bic(X[test_ind, :])
            if bics[:, l].mean() < lowest_bic:
                lowest_bic = bics[:, l].mean()
                self.M = n_components
                self.gmm = gmm
        print('M={}'.format(self.M))

        self.gmm = GaussianMixture(self.M, covariance_type='full').fit(X)
        y = self.gmm.predict(X)

        tmp = zeros((self.M, self.M))
        for i, j in zip(y[:-1], y[1:]):
            tmp[i, j] += 1
        self.transmat = array([tmp[i, :] / tmp[i, :].sum() for i in range(self.M)])

        #self.hmm = hmm.GaussianHMM(n_components=self.M, covariance_type='full',
        #                           startprob_prior=self.gmm.weights_, transmat_prior=transmat,
        #                           means_prior=self.gmm.means_, covars_prior=self.gmm.covariances_)
        #self.hmm = self.hmm.fit(X)

        self._vn = sqrt(2 * log(1e6))
        self._un = self._vn - log(4 * pi * log(1e6)) / (2 * self._vn)

        self.cvar = zeros((self.M, self.n))

        for m in range(self.M):
            for k in range(self.n):
                self.cvar[m, k] = self.gmm.means_[m, k] + \
                                  sqrt(self.gmm.covariances_[m, k, k]) * norm.pdf(norm.ppf(self.alpha[k])) / (1 - self.alpha[k])
        return self

    def conditional_deadline(self, X, k, I=None):
        if not all(X):
            return 0
        X = array(X).reshape(1, -1)
        m_star = self.gmm.predict(X)[0]
        m = random.choice(a=range(self.M), p=self.transmat[m_star, :])

        muk_X, sigmak_X = self._condition(m, k, X, I)
        if sigmak_X <= 0:
            uk_X = muk_X
            vk_X = 0
        else:
            uk_X = muk_X + sqrt(sigmak_X) * self._un
            vk_X = sqrt(sigmak_X) / self._vn
        if vk_X > 0.01:
            d = uk_X - vk_X * log(-log(self.alpha[k]))
        elif self.cvar[m, k] < uk_X:
            d = uk_X

        return d

    def conditional_dmp(self, X, k, I=None):
        if not all(X):
            return 0
        X = array(X).reshape(1, -1)
        m_star = self.gmm.predict(X)[0]

        dmp = 0

        for m in range(self.M):
            muk_X, sigmak_X = self._condition(m, k, X, I)
            if sigmak_X <= 0:
                uk_X = muk_X
                vk_X = 0
            else:
                uk_X = muk_X + sqrt(sigmak_X) * self._un
                vk_X = sqrt(sigmak_X) / self._vn
            if vk_X > 0.01:
                dmp += self.transmat_[m_star, m] * \
                       self.gmm.weights_[m] * Gumbel.sf(self.cvar[m, k], uk_X, vk_X)
            elif self.cvar[m, k] < uk_X:
                dmp += self.transmat_[m_star, m] * self.gmm.weights_[m]

        return dmp

    def _condition(self, m, k, X, I):

        if I and len(I) > 1:
            ind_X = I
        elif len(I) == 1:
            ind_X = I[0]
        else:
            ind_X = list(range(k)) + list(range(k + 1, self.n))

        mu, sigma = array(self.hmm.means_[m]), array(self.hmm.covars_[m])
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