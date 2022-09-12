from scipy.optimize import fsolve
import numpy as np
from numpy.random import rand as random, seed
from scipy.stats import kendalltau, gumbel_r as gumbel
from simso.estimation.Modes import Modes
from scipy.stats import multivariate_normal as mvn, norm


class Gumbel:
    def __init__(self, d=None, beta=None, tau=None):
        self.uv = None
        self.n = None
        self.d = d
        if beta:
            self.tau = self.f_tau(beta)
            self.beta = beta
        elif tau:
            self.tau = tau
            self.beta = self.f_beta(tau)
        else:
            self.beta = beta
            self.tau = tau

    def _KendallDistribution(self, t):
        return t - (t * np.log(t) / self.beta)

    def _derivativeKendallDistribution(self, t):
        return 1 - np.log(t) / self.beta - 1 / self.beta

    def _Generator(self, t):
        return (-np.log(t)) ** self.beta

    def _GeneratorInverse(self, t):
        return np.exp(-t ** (1 / self.beta))

    def _derivativeGenerator(self, t):
        return - self.beta / t * (-np.log(t)) ** (self.beta - 1)

    def f_tau(self, b):
        return 1 - 1 / b

    def f_beta(self, t):
        return 1 / (1 - t)

    def copulaFunction(self, u):
        return self._GeneratorInverse(sum(self._Generator(uu) for uu in u))

    def _kendallInverse(self, q, x0):
        # Newton Raphson
        return fsolve(lambda t: self._KendallDistribution(t) - q, np.array(0.00001),
                      fprime=self._derivativeKendallDistribution)
                      #fprime2=self._secondDerivativeKendallDistribution)

    def simulate(self, Number):
        """
        Wu, Florence, Emiliano Valdez, and Michael Sherris.
        "Simulating from exchangeable Archimedean copulas."
        Communications in Statistics—Simulation and Computation® 36.5 (2007): 1019-1034.
        https://www.researchgate.net/profile/Michael_Sherris/publication/228620506_Simulating_exchangeable_multivariate_Archimedean_copulas_and_its_applications/links/0deec51798b793282f000000.pdf
        """
        w = random(Number, self.d)

        self.uv = np.random.rand(Number, self.d)
        if self.beta > 1:
            for i in range(Number):
                S = [ww ** (1 / (k + 1)) for k, ww in enumerate(w[i, :-1])]
                t = self._kendallInverse(w[i, -1], 1e-3/Number)
                self.uv[i, 0] = self._GeneratorInverse(np.prod(S) * self._Generator(t))
                if self.d > 2:
                    self.uv[i, 1:-1] = [
                        self._GeneratorInverse((1 - S[k - 1]) * np.prod(S[k:]) * self._Generator(t))
                        for k in range(1, self.d-1)
                    ]
                self.uv[i, -1] = self._GeneratorInverse((1 - S[-1]) * self._Generator(t))
        if Number == 1:
            return np.array(self.uv).reshape(-1)
        else:
            return np.array(self.uv)


class GraphicalGumbel(Gumbel):

    def __init__(self, d=None, theta=None, beta=None, tau=None, alpha=None):
        super().__init__(d=d, beta=beta, tau=tau)
        self.beta_ = None
        self.tau_ = None
        self.U = None
        self.V = None
        self.eta = None
        self._dependence_graph = []
        self.alpha = alpha
        self.delta = None
        self.theta = theta

    def fit(self, X, tresh=0):
        self.n, self.d = X.shape
        self.U = [0] * self.d
        self.V = [0] * self.d
        self.eta = [0] * self.d
        self.tau_ = np.zeros((self.d, self.d))
        self.beta_ = np.ones((self.d, self.d))
        if self.alpha is None:
            self.alpha = self._alpha(self.n)
        self.delta = self._delta(self.alpha)
        mean, covariance, precision = self.theta
        #correlation = np.eye(self.d)
        for i in range(self.d):
            self.beta_[i, i] = np.inf
            #for j in range(i):
            #    correlation[i, j] = abs(covariance[i, j] / np.sqrt(covariance[i, i] * covariance[j, j]))
            #    correlation[j, i] = correlation[i, j]
        # Adjacense matrix of graphical model
        adj_matrix = 1*(abs(precision) > tresh) - np.eye(self.d)
        N = np.sum(adj_matrix, axis=1)
        for k in range(self.d):
            #BMk = self.block_maxima(X[:, k])
            muk, sigmak = self.estimate_conditional_marginals(k, mean, covariance, np.log(X))
            self.U[k], self.V[k] = self.estimate_margins(muk, sigmak)  # gumbel.fit(BMk)
            #if N[k]:
            #    self.eta[k] = 1 / N[k]
            #else:
            #    self.eta[k] = 0
        #if sum(N) == 0:
        #    self.beta = 1
        #    self.tau = 0
        #else:
        #    for k, l in zip(*np.where(adj_matrix)):
        #        self.estimate_beta(k, l, X)
        #        if self.beta_[k, l] > 1:
        #            self.add_dependence((k, l))
        #    self.tau = self._tau_global()
        #    self.beta = self.f_beta(self.tau)
        return self

    def add_dependence(self, e):
        self._dependence_graph.append(e)

    def estimate_beta(self, k, l, X):
        Yk, Yl = self.biBlockMaxima(X[:, (k, l)])
        self.tau_[k, l] = abs(kendalltau(Yk, Yl)[0])
        self.tau_[l, k] = self.tau_[k, l]
        self.beta_[k, l] = self.f_beta(self.tau_[k, l])
        self.beta_[l, k] = self.beta_[k, l]

    def estimate_tail(self, k, l, X, alpha=None):
        # Tail index
        if alpha is None:
           alpha = self.alpha

        qk_alpha = self.valueAtRisk(k, alpha=alpha)
        ql_alpha = self.valueAtRisk(l, alpha=alpha)

        ind_k = X[:, k] > qk_alpha
        ind_l = X[ind_k, l] > ql_alpha

        if sum(ind_k) > sum(ind_l) > 0:
           lambda_alpha = sum(ind_l) / sum(ind_k)
        else:
           lambda_alpha = 0

        return lambda_alpha

    def estimate_margins(self, muk, sigmak):
        # Norming sequences
        u_n = np.sqrt(2 * np.log(self.n)) - (np.log(4 * np.pi) + np.log(np.log(self.n))) / (2 * np.sqrt(2 * np.log(self.n)))
        v_n = 1 / np.sqrt(2 * np.log(self.n))
        # Gumbel parameters
        uk = np.exp(muk + sigmak * u_n)
        vk = uk * sigmak * v_n
        return uk, vk

    def estimate_conditional_marginals(self, k, mu, sigma, X):
        d = sigma.shape[0]
        ind_X = list(range(k)) + list(range(k + 1, d))
        if d > 2:
            sigma_X = sigma[ind_X, ind_X]
            a = np.dot(sigma[k, ind_X], np.linalg.inv(sigma_X))
            return (np.dot(a, X[:, ind_X] - mu[ind_X]) + mu[k]).mean(), sigma[k, k] - np.dot(a, sigma[k, ind_X])
        elif d == 2:
            ind_X = ind_X[0]
            sigma_X = sigma[ind_X, ind_X]
            a = sigma[k, ind_X] / sigma_X
            return (a * (X[:, ind_X] - mu[ind_X]) + mu[k]).mean(), sigma[k, k] - a * sigma[k, ind_X]

    def margin(self, x, j):
        """
        cdf of a Gumbel distribution
        """
        return np.exp(-np.exp(-(x - self.U[j]) / self.V[j]))

    def margin_inv(self, u, j):
        """
        inverse cdf of a Gumbel distribution
        """
        return self.U[j] - self.V[j] * np.log(-np.log(u))

    def _alpha(self, n):
        return 1/np.sqrt(2 * np.log(n))

    def _li(self, alpha, num=100000):
        """
        Riemann integral of int_0^alpha dx/ln(x)
        """
        if isinstance(alpha, np.ndarray):
            ll = alpha[0] / np.log(alpha[0])
            l = [ll]
            for i, a in enumerate(alpha[1:]):
                ll += (a - alpha[i]) / np.log(a)
                l.append(ll)
            return np.array(l)
        else:
            x = np.linspace(0, alpha, num)
            return sum([(xx - x[i]) / np.log(xx) for i, xx in enumerate(x[1:])])

    def _delta(self, alpha):
        return (0.57721 - self._li(alpha) + alpha * np.log(-np.log(alpha))) / (1-alpha)

    def valueAtRisk(self, j, alpha=None):
        """
        https://en.wikipedia.org/wiki/Expected_shortfall#Generalized_extreme_value_distribution_(GEV)
        """
        if alpha is None:
            alpha = self.alpha
        return self.U[j] - self.V[j] * np.log(-np.log(alpha))

    def conditionalValueAtRisk(self, j, alpha=None):
        """
        https://en.wikipedia.org/wiki/Expected_shortfall#Generalized_extreme_value_distribution_(GEV)
        """
        if alpha:
            return self.U[j] + self.V[j] * self._delta(alpha)
        else:
            return self.U[j] + self.V[j] * self.delta

    def conditional_cdmp(self, x, k, l):
        exp_d = np.exp(-self.delta)
        if (k, l) in self.dependence_graph():
            exp_z = np.exp(-(x - self.U[k]) / self.V[k])
            x1 = (self.eta[l] * exp_d) ** self.beta_[k, l]
            x1 += (self.eta[k] * exp_z) ** self.beta_[k, l]
            x1 = x1 ** (1. / self.beta_[k, l])
            x2 = (1 - self.eta[l]) * exp_d - self.eta[k] * exp_d
            x3 = self.eta[k] * (1 - (self.eta[k] * exp_z / x1) ** (self.beta_[k, l] - 1))
            return 1 - np.exp(-x1 - x2) * (1 - x3)
        else:
            return 1 - np.exp(-exp_d)

    def dependence_graph(self):
        return self._dependence_graph

    def _tau_global(self):
        t = []
        for i in range(1, self.d):
            for j in range(i):
                t.append(self.tau_[i, j])
        return np.mean(t)

    def biBlockMaxima(self, X, k=0, l=1, b_size=None):
        if b_size is None:
            b_size = int(1/self.alpha)
        z = []
        for b in range(X.shape[0] // b_size):  # b for block
            i = np.argmax(X[b * b_size:(b + 1) * b_size, k]) + b * b_size
            j = np.argmax(X[b * b_size:(b + 1) * b_size, l]) + b * b_size
            z.append(X[i, (k, l)])
            z.append(X[j, (k, l)])
        return np.array(z)[:, 0], np.array(z)[:, 1]

    def block_maxima(self, X, b_size=None):
        if b_size is None:
            b_size = int(1/self.alpha)
        return [max(X[k*b_size:(k+1)*b_size]) for k in range(X.shape[0]//b_size)]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(5, 2, figsize=(20, 10))

    for beta, ax_ in zip(np.linspace(1.1, 5, 10), ax.reshape(-1)):
        U = Gumbel(d=3, beta=beta).simulate(2)
        ax_.scatter(U[:, 0], U[:, 1], alpha=0.2)

    for i in range(1000):
        U = Gumbel(d=3, beta=beta).simulate(1)

    fig.suptitle("Gumbel copulas")
    fig.show()

    m = False
    if m:
        import matplotlib.pyplot as plt
        from scipy.stats import gumbel_r

        # DATASET
        n = 100000
        l_beta = []
        l_beta_chap = []
        l_beta_tail = []

        RHO = np.linspace(0, 0.99)

        for rho in RHO:
            gaussian = mvn(mean=[3, 4], cov=[[1, rho], [rho, 1]])
            X = np.exp(gaussian.rvs(size=n))

            modes = Modes(M=1).fit(X)

            k, l = 0, 1
            g_cop = modes.copulas[0]
            X1, X2 = g_cop.biBlockMaxima(X, k, l)
            maxX = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)])
            print(g_cop.beta_[k, l])

            # lambd = 0
            # no = 100
            # for alpha in np.linspace(g_cop.alpha, 1/n, no):
            #	l = g_cop.estimate_tail(k, l, X)
            #	lambd += l
            #	lambd /= no
            # l_beta_tail.append(np.log(2)/np.log(2 - lambd))
            #
            # U = Gumbel(d=g_cop.d, beta=l_beta_tail[-1]).simulate(n)
            # Y1 = g_cop.margin_inv(U[:, k], k)
            # Y2 = g_cop.margin_inv(U[:, l], l)

            # for beta in BETA:
            # Simulation of same marginals than X with given beta
            U = g_cop.simulate(n)
            plt.scatter(U[:, k], U[:, l], alpha=0.2)
            plt.show()

            Z1 = g_cop.margin_inv(U[:, k, np.newaxis], k)
            Z2 = g_cop.margin_inv(U[:, l, np.newaxis], l)
            Z = np.concatenate([Z1, Z2], axis=1)
            print(kendalltau(Z1, Z2)[0])

            beta = 1 / (1 - abs(kendalltau(Z1, Z2)[0]))
            l_beta.append(beta)
            l_beta_chap.append(g_cop.beta)

            # sns.jointplot(x='X', y='Y', KopernicPasseVirtuel2020data=Z, marginal_kws=dict(bins=100), marker='o', alpha=0.2)
            # plt.xlabel("X", size=16)
            # plt.ylabel("y", size=16)
            # plt.show()
            # sns.scatterplot(x='X', y='Y', data=Z)
            #
            # beta_chap = []
            # alpha = 1/np.log(n)
            #
            ##for alpha in np.linspace(1/np.sqrt(2 * np.log(n))**3, 1/np.sqrt(2 * np.log(n)), 10):
            #

            # l_tau.append(g_cop.f_beta(tau))
            # plt.xlabel("X", size=16)
            # plt.ylabel("y", size=16)
            # plt.title('beta = {}'.format(g_cop.beta_[k, l]))
            # plt.show()
            fig, ax = plt.subplots(1, 2)
            ax[0].scatter(Z1, Z2, color='blue', alpha=0.2)
            # ax[0, 1].scatter(Y1, Y2, color='red', alpha=0.2)
            ax[1].scatter(X1, X2, color='green', alpha=0.2)

            plt.suptitle('rho = {}'.format(rho))
            ax[0].set_title('beta = {}'.format(beta))
            # ax[0, 1].set_title('beta = {}'.format(l_beta_tail[-1]))
            ax[1].set_title('beta = {}'.format(g_cop.beta_[k, l]))

            plt.show()

        plt.plot(RHO, l_beta)
        plt.plot(RHO, l_beta_chap)
        plt.plot(RHO, l_beta_tail)
        plt.show()
