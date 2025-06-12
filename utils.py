import sys
import numpy as np
from scipy.linalg import expm, sinm, cosm
from scipy.special import factorial
from numpy.linalg import matrix_power
import torch
from sympy.codegen.ast import float64
from tqdm import tqdm
import pandas as pd
import random

sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')
sys.path.append(r'C:\Users\Eshel\workspace\butools2\Python')

from scipy.linalg import expm
from scipy.integrate import quad

from butools.queues import *

def ph_density(x, alpha, T):
    return alpha @ expm(T * x) @ (-T @ np.ones(len(alpha)))

def kl_divergence(alpha1, T1, alpha2, T2):
    def integrand(x):
        f1 = ph_density(x, alpha1, T1)
        f2 = ph_density(x, alpha2, T2)
        if f1 > 0 and f2 > 0:
            return f1 * np.log(f1 / f2)
        else:
            return 0.0  # Avoid log(0) or division by zero

    result, _ = quad(integrand, 0, np.inf, limit=100, epsabs=1e-6)
    return result

def moment_analytics(ms, comp):
    original_moments = ms.detach().numpy()
    computed_moments = [m.detach().item() for m in comp]
    moment_table = pd.DataFrame([computed_moments, original_moments], index="computed target".split()).T
    moment_table["delta"] = moment_table["computed"] - moment_table["target"]
    moment_table["delta-relative"] = 100*moment_table["delta"] / moment_table["target"]
    return moment_table

def compute_moments(a, T, k, n):
    """ generate first n moments of FT (a, T)
    m_i = ((-1) ** i) i! a T^(-i) 1
    """
    T_in = torch.inverse(T)
    T_powers = torch.eye(k).double()
    signed_factorial = 1.
    one = torch.ones(k).double()

    moms = []
    for i in range(1, n+1):
        signed_factorial *= -i
        T_powers = torch.matmul(T_powers, T_in)      # now T_powers is T^(-i)
        moms.append( signed_factorial * a @ T_powers @ one)

    return moms

# def compute_moments(a, T, k, n):
#     """ generate first n moments of FT (a, T)
#     m_i = ((-1) ** i) i! a T^(-i) 1
#     """
#     T_in = torch.inverse(T)
#     T_powers = torch.eye(k)
#     signed_factorial = 1.
#     one = torch.ones(k)
#
#     for i in range(1, n+1):
#         signed_factorial *= -i
#         T_powers = torch.matmul(T_powers, T_in)      # now T_powers is T^(-i)
#         yield signed_factorial * a @ T_powers @ one


def make_ph(lambdas, ps, alpha, k):
    """ Use the arbitrary parameters, and make a valid PT representation  (a, T):
        lambdas: positive size k
        ps: size k x k
        alpha: size k
    """
    ls = lambdas ** 2
    a = torch.nn.functional.softmax(alpha, 0)
    p = torch.nn.functional.softmax(ps, 1)
    lambdas_on_rows = ls.repeat(k, 1).T
    T = (p + torch.diag(-1 - torch.diag(p))) * lambdas_on_rows

    return a, T


# def get_feasible_moments(original_size, n = 10):
#     """ Compute feasible k-moments by sampling from high order PH and scaling """
#     k = original_size
#     ps = torch.randn(k, k)
#     lambdas = torch.rand(k) * 100
#     alpha = torch.randn(k)
#     a, T = make_ph(lambdas, ps, alpha, k)
#
#     # Compute mean
#     ms = compute_moments(a, T, k, 1)
#     m1 = torch.stack(list(ms)).item()
#
#     # Scale
#     T = T * m1
#     ms = compute_moments(a, T, k, n)
#     momenets = torch.stack(list(ms))
#     return a, T , momenets

def get_feasible_moments(original_size):
    """ Compute feasible k-moments by sampling from high order PH and scaling """
    k = original_size
    ps = torch.randn(k, k).astype(np.float64)
    lambdas = torch.rand(k).astype(np.float64) * 100
    alpha = torch.randn(k).astype(np.float64)
    a, T = make_ph(lambdas, ps, alpha, k)
    # Compute mean
    ms = compute_moments(a, T, k, 1)
    m1 = torch.stack(list(ms)).item()
    # Scale
    T = T * m1
    # ms = compute_moments(a, T, k, n)
    # momenets = torch.stack(list(ms))
    return a, T

def embedd_next_ph(a, T, k):
    """ Embedd the order-k PH (a, T) in order (k+1)
    a' = [a; 0]
    T' = [[T 0], [0..., 1]]
    """
    a1 = torch.hstack([a, torch.Tensor([0.])])
    T1 = torch.zeros((k+1, k+1))
    T1[:-1, :-1] = T
    T1[-1, -1] = -1.
    return a1, T1


def embedd_next_parametrization(l, ps, a, k):
    a1 = torch.hstack([a, torch.Tensor([-INF])])
    l1 = torch.hstack([l, torch.Tensor([1.])])
    p1 = torch.ones(k+1, k+1) * -INF
    p1[:-1, :-1] = ps
    p1[-1, -1] = 1.
    return l1, p1, a1





def ser_mean(alph, T):
    e = np.ones((T.shape[0], 1))
    try:
        return -np.dot(np.dot(alph, np.linalg.inv(T)), e)
    except:
        return False

def compute_pdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return np.dot(np.dot(s, expm(A * x)), A0)


def compute_cdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return 1 - np.sum(np.dot(s, expm(A * x)))

def kroneker_sum(G,H):
    size_g = G.shape[0]
    size_h = H.shape[0]
    return np.kron(G, np.identity(size_h)) + np.kron( np.identity(size_g),H)


def ser_moment_n(s, A, mom):
    '''
    ser_moment_n
    :param s:
    :param A:
    :param mom:
    :return:
    '''
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) *factorial(mom)*np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False

def compute_first_n_moments(s, A, n=3):
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list


def compute_cdf_within_range(x_vals, s, A):
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_cdf(x, s, A).flatten())

    return pdf_list


def compute_pdf_within_range(x_vals, s, A):
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_pdf(x, s, A).flatten())

    return pdf_list

def compute_R(lam, alph, T):
    e = torch.ones((T.shape[0], 1))
    return np.array(lam * torch.inverse(lam * torch.eye(T.shape[0]) - lam * e @ alph - T))


def steady_i(rho, alph, R, i):
    return (1 - rho) * alph @ matrix_power(R, i)

def ser_mean_torch(alph, T):
    e = torch.ones((T.shape[0], 1))
    try:
        return -alph @ torch.inverse(T) @ e
    except:
        return False


def compute_steady(s_arrival, A_arrival, s_service, A_service, y_size=1000, eps=0.00000001):
    print('start')
    inter_arrival_expected = ser_moment_n(s_arrival, A_arrival, 1)
    inter_service_expected = ser_moment_n(s_service, A_service, 1)
    print('Construct matrices')
    A_service0 = -np.dot(A_service, np.ones((A_service.shape[0], 1)))
    A_arrival0 = -np.dot(A_arrival, np.ones((A_arrival.shape[0], 1)))

    A0 = A_arrival
    A1 = np.kron(np.identity(A_arrival.shape[0]), A_service0)
    A = kroneker_sum(np.zeros(A_arrival.shape), np.dot(A_service0, s_service))
    B = kroneker_sum(A_arrival, A_service)
    C = kroneker_sum(np.dot(A_arrival0, s_arrival), np.zeros((A_service.shape[0], A_service.shape[0])))
    C0 = np.kron(np.dot(A_arrival0, s_arrival), s_service)
    print('Construct R')
    R = QBDFundamentalMatrices(A, B, C, "R")

    rho = (ser_moment_n(s_service, A_service, 1) / ser_moment_n(s_arrival, A_arrival, 1))[0][0]
    print('Start transpose')
    A0T = A0.transpose()
    A1T = A1.transpose()
    C0T = C0.transpose()
    BRAT = np.array(B + np.dot(R, A)).transpose()
    print('Start equations')
    eqns = np.concatenate((np.concatenate((A0T, A1T), axis=1), np.concatenate((C0T, BRAT), axis=1)), axis=0)[:-1, :]

    sys_size = \
    np.concatenate((np.concatenate((A0T, A1T), axis=1), np.concatenate((C0T, BRAT), axis=1)), axis=0)[:-1, :].shape[1]
    u0_size = A0.shape[0]
    u0_eq = np.zeros((1, sys_size))
    u0_eq[0, :u0_size] = 1

    tot_eqns = np.concatenate((eqns, u0_eq), axis=0)

    u0 = 1 - rho
    B = np.zeros(sys_size)
    B[-1] = u0

    X = np.linalg.solve(tot_eqns, B)

    steady = np.zeros(y_size)
    steady[0] = np.sum(X[:A0.shape[0]])
    steady[1] = np.sum(X[A0.shape[0]:])
    tot_sum = np.sum(X)
    for ind in tqdm(range(2, y_size)):

        steady[ind] = np.sum(np.dot(X[u0_size:], matrix_power(R, ind - 1)))
        if np.sum(steady) > 1 - eps:
            break

    return steady




def compute_skewness_and_kurtosis_from_raw(m1, m2, m3, m4):
    # Compute central moments
    mu2 = m2 - m1 ** 2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4

    # Compute skewness and kurtosis
    skewness = mu3 / (mu2 ** 1.5)
    kurtosis = mu4 / (mu2 ** 2)
    excess_kurtosis = kurtosis - 3

    return skewness, kurtosis


# def get_PH_general_with_zeros(original_size):
#
#     a, T = get_feasible_moments(original_size)
#
#     matrix = T  # torch.randint(1, 10, (n, n))
#     n = T.shape[0]
#     # Set the diagonal to non-zero values to avoid diagonal zeros
#
#     # Determine the number of off-diagonal zeros to insert (arbitrary, between 1 and n^2 - n)
#     num_zeros = random.randint(1, n ** 2 - n)
#
#     # Get all off-diagonal indices
#     off_diagonal_indices = [(i, j) for i in range(n) for j in range(n) if i != j]
#
#     # Randomly select locations for zeros
#     zero_indices = random.sample(off_diagonal_indices, num_zeros)
#
#     # Insert zeros at the selected locations
#     for i, j in zero_indices:
#         matrix[i, j] = 0
#
#     # Print the resulting matrix
#
#     ms = compute_moments(a, T, n, 1)
#     m1 = torch.stack(list(ms)).item()
#     # Scale
#     T = T * m1
#
#     ms = compute_moments(a, T, n, 10)
#
#     return a, T, torch.stack(ms)
def get_PH_general_with_zeros(original_size):

    a, T = get_feasible_moments(original_size)


    matrix = T  # torch.randint(1, 10, (n, n))
    n = T.shape[0]
    # Set the diagonal to non-zero values to avoid diagonal zeros

    # Determine the number of off-diagonal zeros to insert (arbitrary, between 1 and n^2 - n)
    num_zeros = random.randint(1, n ** 2 - n)

    # Get all off-diagonal indices
    off_diagonal_indices = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Randomly select locations for zeros
    zero_indices = random.sample(off_diagonal_indices, num_zeros)

    # Insert zeros at the selected locations
    for i, j in zero_indices:
        matrix[i, j] = 0

    # Print the resulting matrix

    ms = compute_moments(a, T, n, 1)
    m1 = torch.stack(list(ms)).item()
    # Scale
    T = T * m1

    ms = compute_moments(a, T, n, 10)
    print('wwwww')

    return a, T, torch.stack(ms)


def sample_coxian(degree, max_rate):
    # print(degree)
    lambdas_ = np.random.rand(degree).astype(np.float64) * max_rate
    ps_ = np.random.rand(degree - 1).astype(np.float64)
    A = np.diag(-lambdas_) + np.diag(lambdas_[:degree - 1] * ps_, k=1)
    alpha = np.eye(degree).astype(np.float64)[[0]]
    mean_val = ser_moment_n(alpha, A, 1)
    A1 = A * mean_val.item()
    lambdas_ = lambdas_ * mean_val.item()
    return alpha, A1, lambdas_, ps_


# get_PH_general_with_zeros(10)