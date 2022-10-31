import matplotlib.pyplot as plt
import numpy as np
from numpy import random, array, where, mean, convolve, exp, lcm, random, linspace, cumsum, log, sqrt, std, arange
from scipy.stats import lognorm, norm, expon, uniform, poisson
from scipy.signal import convolve
from itertools import product
import bisect
from tqdm import tqdm, trange


def make_instance(offsets, periods, deadlines, T=None):
    if T is None:
        T = lcm.reduce(periods)
    M = [T // p for p in periods]

    jobs = [(i, o + k * p, o + k * p + d)
                for i, (o, p, d) in enumerate(zip(offsets, periods, deadlines)) for k in range(1, M[i])]
    return sorted(jobs, key=lambda j: (j[1], j[0]))


def conv(w, c, k=0):
    assert len(w[0]) == len(w[1]) and len(c[0]) == len(c[1]), 'values and prob not matching'
    p = []
    r = []
    prod_values = list(product(w[0][k:], c[0]))

    for x, y in prod_values:
        #if x+y in r:
        #    pass
        #else:
        #    r.append(x+y)
        j1 = bisect.bisect_left(w[0][k:], x)
        j2 = bisect.bisect_left(c[0], y)
        #j1 = int(where(array(w[0][k:]) == x)[0][0])
        #j2 = int(where(array(c[0]) == y)[0][0])
        if x + y in r:
            j = bisect.bisect_left(array(r), x+y)
            #j = int(where(array(r) == x+y)[0][0])
            p[j] += w[1][k:][j1] * c[1][j2]
        else:
            j = bisect.bisect_left(r, x+y)
            r.insert(j, x+y)
            p.insert(j, w[1][k:][j1] * c[1][j2])
            #p.append(array(w[1][k:])[j1] * array(c[1])[j2])
    return r, p  # convolve(w[1], c[1])


def diaz_conv(r, delta, c):
    if max(r[0]) <= delta:
        return r
    k = min(where(array(r[0]) > delta)[0])
    rr, p = conv(r, c, k)
    return r[0][:k] + rr, r[1][:k] + p


def shrink(w, delta):
    assert len(w[0]) == len(w[1]), 'values and prob not matching'
    if delta == 0:
        return w
    ind_to_shrink = list(array(w[0]) <= delta)
    ind_to_keep = list(array(w[0]) > delta)
    p = [sum(array(w[1])[ind_to_shrink])]
    return [0] + list(array(w[0])[ind_to_keep] - delta), p + list(array(w[1])[ind_to_keep])


def update_workload(ww, c, delta):
    assert len(ww[0]) == len(ww[1]) and len(c[0]) == len(c[1]), 'values and prob not matching'
    return conv(shrink(ww, delta), c)


def support_size(mu, nu):
    tmp = list(mu)
    miss = list(set(nu[0]) - set(mu[0]))
    for v in miss:
        k = bisect.bisect(mu[0], v)
        tmp[0].insert(k, v)
        tmp[1].insert(k, 0.)
    return tuple(tmp)


def dist_K(mu, nu):
    mu, nu = list(mu), list(nu)
    mu = support_size(mu, nu)
    nu = support_size(nu, mu)
    F_mu = cumsum(mu[1])
    F_nu = cumsum(nu[1])
    return max([abs(m - n) for m, n in zip(F_mu, F_nu)])


def backlogs(ww, level, delta):
    assert len(ww[0]) == len(ww[1]), 'values and prob not matching'
    bl0 = ([0], [1.])
    for task_level in ww.keys():
        if task_level <= level:
            bl0 = conv(shrink(ww[task_level], delta), bl0)
    return bl0


def work_to_load(w, t):
    return tuple((list(array(w[0]) / t), w[1]))


def stationary_backlog(offsets, execution_times, periods, deadlines, tol=1e-3, eps=1e-2, verbose=False,n=10,
                       sched='RM', max_iter=None, exp=False, return_int=False, T=None):
    H = lcm.reduce(periods)
    if max_iter:
        pass
    else:
        max_iter = 10000 // H
    instance = make_instance(offsets, periods, deadlines, n=n, exp=exp, T=T, return_int=return_int)
    print("checkpoint")
    n_iter = 0
    variation = tol + 1
    w = execution_times[instance[0][0]]
    prev_w = ([0], [1.])
    Ubar = sum([sum([c0 * c1 for c0, c1 in zip(*c)]) / periods[i] for i, c in execution_times.items()])
    T = 0
    width = "\ "
    width = width[0]+'textwidth'
    for t, job in tqdm(enumerate(instance[1:])):
        task, release, deadline = job
        delta = release - instance[t][1]
        T += delta
        w = update_workload(w, execution_times[task], delta)
        #x_range = linspace(min(w[0])-delta, max(w[0]))
        #if not exp:
        #if T % H == 0:
        #    n_iter += 1
        #    if T > 0 and task == len(periods)-1:
        #        variation = dist_K(shrink(prev_w), shrink(w))
        #        if verbose:
        #            plt.bar(*w)
        #            plt.plot(x_range, backlog_pdf(x_range, T, execution_times, periods), color='red', label='gauss')
        #            plt.xlim(0, 30)
        #            plt.title('iteration {}'.format(t+1))
        #            plt.savefig('backlog_{}.pdf'.format(n_iter))
        #            #tpl.save('backlog_{}.tex'.format(n), axis_width=width)
        #            plt.show()
#
        #        converged = variation < eps
        #        if converged or n_iter >= max_iter:
        #            print('{} iteration to converge'.format(n_iter))
        #            return w, n_iter, T
        #else:
        #    if T > 0 and task == len(periods)-1:
        #        n_iter += 1
        #        variation = dist_K(prev_w, w)
        #        if verbose:
        #            plt.bar(*w)
        #            plt.plot(x_range, backlog_pdf(x_range, T, execution_times, periods), color='red', label='gauss')
        #            plt.title('iteration {}'.format(t + 1))
        #            plt.savefig('backlog_{}.pdf'.format(n))
        #            #tpl.save('backlog_{}.tex'.format(n), axis_width=width)
        #            plt.show()
#
        #        converged = variation < eps
        #        if converged:
        #            print('{} iteration to converge'.format(n_iter))
        #            return w, n_iter, T
    w[1][0] = 1 - sum(w[1][1:])
    return w


def diaz(offsets, execution_times, periods, deadlines, T ,sched='RM'):
    ### COMPUTE INSTANCE  #################################################
    instance = make_instance(offsets, periods, deadlines, T=T)  # Instance sorted by release and deadline
    #H = lcm.reduce(periods)
    #N = [H // p for p in periods]
    #n = len(periods)
    ### INIT DIAZ ALGO  ###################################################
    RT = {}
    RT[instance[0][0]] = execution_times[instance[0][0]]  # First element of instance is not preempted and backlog is zero
    check = [0]*len(periods)
    check[0] = 1
    ### RESPONSE TIMES  ###################################################
    for t, job in enumerate(instance[1:]):  # job index in instance[1:] is t+1
        task, release, deadline = job
        if check[task]:
            continue
        # COMPUTE BACKLOG + EXECUTION TIME ################################
        #bl = backlogs(w, level=task, delta=release - instance[t][1])
        rt0 = execution_times[task]

        # COMPUTE PREEMPTIONS #############################################
        HP = [j for j in instance if j[0] < task and (j[2] > release or j[1] < deadline)]

        for hp_task, hp_release, hp_deadline in HP:
            if hp_release < release:
                rt0 = diaz_conv(RT[hp_task], release - hp_release, execution_times[hp_task])
                check[task] += 1
            else:
                rt0 = conv(rt0, execution_times[hp_task])
                check[task] += 1

        # COMPUTE WORKLOAD DISTRIBUTION ###################################
        #w[task] = update_workload(w[task], execution_times[task], release - instance[t][1])  # release - previous release
        RT[task] = rt0
    return RT, all(check)
    ## COMPUTE DISTRIBUTION OF R_i as the mean of the pdfs
    #Z = dict((task, {}) for task in RT.keys())
    #for task, R_task in RT.items():
    #    values = []
    #    for (x, y) in R_task:
    #        values += x
    #    values = list(set(values))
    #    Z[task] = dict((v, 0) for v in values)
    #    m = len(R_task)
    #    for (x,y) in R_task:
    #        for xx, yy in zip(x, y):
    #            Z[task][xx] += yy
    #    for xx in Z[task].keys():
    #        Z[task][xx] /= m
    #return dict((task, (list(Z[task].keys()), list(Z[task].values()))) for task, R_task in RT.items())


def backlog_cdf(x, t, execution_times, periods):
    U = [sum([c0 * c1 for c0, c1 in zip(*c)]) / periods[i] for i, c in execution_times.items()]
    var_C = [(sum([c0**2 * c1 for c0, c1 in zip(*c)]))/periods[i] for i, c in enumerate(C)]
    #G = lambda y: norm.cdf(y[0], sum(U) , sqrt(sum(var_C)/y[1]))
    sigma = sqrt(sum(var_C)*t)
    return norm.cdf(x,  t * sum(U), sigma)


def backlog_pdf(x, t, execution_times, periods):
    U = [sum([cc[0] * cc[1] for cc in zip(*c)]) / periods[i] for i, c in execution_times.items()]
    var_C = [(sum([cc[0] ** 2 * cc[1] for cc in zip(*c)]) - U[i] ** 2) /periods[i] for i, c in enumerate(C)]
    sigma = sqrt(sum(var_C)*t)
    return norm.pdf(x, t * (sum(U)-1), sigma)



#if __name__ == '__main__':
#    from simso.generator.task_generator import generate_ptask_set
#    from dSumExponential.sum_exponential import SumExponential
#    from statsmodels.distributions.empirical_distribution import ECDF
#
#    n = 3
#    C, periods = generate_ptask_set(n, 1.2, periodmin=10, periodmax=30)
#    execution_times = dict((i, c) for i, c in enumerate(C))
#    assert all((len(c[0]) == len(c[1]) for c in C)), 'values and prob not matching'
#
#    offsets = [0] * n
#    #periods = list(range(4, 20, 2))[:n]
#    H = lcm.reduce(periods)
#    deadlines = periods #(9, 12, 18)
#
#    #var_C = [ sum([ cc * cc * c[1][i] for cc in c[0]]) / periods[i] for i, c in enumerate(C) ]
#    #print(var_C)
#
#    ### CHECK DIAZ HYPOTHESIS  ############################################
#    U = [sum([p * cc for cc, p in zip(*c)]) / periods[i] for i, c in execution_times.items()]
#    Ubar = sum(U)
#    Umax = sum(max(c[0]) / periods[i] for i, c in execution_times.items())
#    print('feasible ? :', Umax, Umax <= 1)
#    print('schedulable ? :', Umax, Umax <= n*(2**(1/n) - 1))
#    print('stationary ? :', Ubar,  Ubar <= 1.)
#    m = array(([sum([c0 * c1 for c0, c1 in zip(*c)]) for c in C]))
#    var_C = array([(sum([c0 ** 2 * c1 for c0, c1 in zip(*c)]) - m[i] ** 2) / periods[i] for i, c in enumerate(C)])
#
#    #mu = ([0], [1.])
##
#    #for c in C:
#    #    mu = conv(mu, c)
#    #plt.bar(*mu)
#    #plt.show()
##
#    T = 100 #* max(periods)
#    instance = make_instance(offsets, periods, deadlines, n=n, exp=exp, T=T, return_int=True)
#    W = ([0], [1.])
#
#    mu = ([0], [1.])
#    H = lcm.reduce(periods)
#
#    for i, c in enumerate(C):
#        for _ in trange(H // periods[i]):
#            mu = conv(mu, c)
#
#    for _ in trange(10):
#        W = update_workload(W, mu, H)
#    W[1][0] = 1 - sum(W[1][1:])
#
#    #R = diaz(offsets, execution_times, periods, deadlines, n=10)
#
#    #for r in R.values():
#    #    plt.bar(*r)
#    #    plt.show()
#
#    eta = 2 * (1 - Ubar) / sum(var_C)
#
#    x_range = linspace(0, 60)
#    fig, ax = plt.subplots(1, 2)
#    sample = random.choice(a=W[0], p=W[1], size=1000) #SumExponential(execution_times=execution_times.values(), periods=periods).sample(1000)
#    cdf = lambda t: ECDF(sample)(t)
#
#    #w_min = min(where(array(W[1]) > 0))[0]
#    ax[0].bar(array(W[0]), W[1], label='Diaz')#
#    ax[1].bar(array(W[0]), cumsum(W[1]), label='Diaz')
#    ax[1].plot(x_range, cdf(x_range), color='red', label='brownian')
#
#    for ax_, title in zip(ax, ('pdf', 'cdf')):
#        #ax_.set_xlim(0, 200)
#        ax_.set_title(title)
#    plt.legend()
#    plt.show()

if __name__ == "__main__":
    from read_csv import read_csv

    execution_times, periods = read_csv("/home/kzagalo/Documents/rInverseGaussian/data/")
    offsets = [0] * len(periods)
    response_times, check = diaz(offsets=offsets, execution_times=execution_times,
                                 periods=periods, deadlines=periods, T=100000)
    print(check)
    for i, r in response_times.items():
        print(r)
        plt.bar(*r)
        plt.title(f'Task {i}')
        plt.show()
    print(response_times)
