import os.path
import torch
import pickle as pkl
import numpy as np
import pandas as pd
import time
import random
import sys
sys.path.append('../')
from utils import *
from mom_fit import *
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(r"C:\Users\Eshel\workspace\one.deep.moment"))
sys.path.append(r'C:\Users\Eshel\workspace\butools2\Python')
sys.path.append('../../one.deep.moment/')
from utils_sample_ph import *
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def get_feasible_moments(original_size):
    """ Compute feasible k-moments by sampling from high order PH and scaling """
    k = original_size
    ps = torch.randn(k, k)
    lambdas = torch.rand(k) * 100
    alpha = torch.randn(k)
    a, T = make_ph(lambdas, ps, alpha, k)
    # Compute mean
    a = a.double()
    T = T.double()

    ms = compute_moments(a, T, k, 1)
    m1 = torch.stack(list(ms)).item()
    # Scale
    T = T * m1
    # ms = compute_moments(a, T, k, n)
    # momenets = torch.stack(list(ms))
    return a, T


def get_PH_general_with_zeros(original_size):
    a, T = get_feasible_moments(original_size)
    a = a.double()
    T = T.double()

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

    return a, T, torch.stack(ms)

def compute_kl_mom_error(res_PH, a_orig, T_orig, mm):
    kl_res = {}
    for key in res_PH.keys():
        curr_a = res_PH[key][0]
        curr_T = res_PH[key][1]
        with torch.no_grad():
            kl_res[key] = kl_divergence(curr_a.cpu().numpy(), curr_T.cpu().numpy(), a_orig.cpu().numpy().flatten(),
                                        T_orig.cpu().numpy())

    df_res_mom_acc = pd.DataFrame([])

    for key in res_PH.keys():
        with torch.no_grad():
            curr_a = res_PH[key][0]
            curr_T = res_PH[key][1]
            momss = compute_moments(curr_a.cpu(), curr_T.cpu(), curr_T.shape[0], 10)
            curr_errors = torch.abs(100 * (torch.tensor(momss).flatten() - mm) / mm)
            curr_errors = np.array(curr_errors)
            curr_ind = df_res_mom_acc.shape[0]
            df_res_mom_acc.loc[curr_ind, 'num_mom_fit'] = key
            df_res_mom_acc.loc[curr_ind, 'kl_div'] = kl_res[key]
            for mom in range(1, 1 + curr_errors.shape[0]):
                df_res_mom_acc.loc[curr_ind, str(mom)] = curr_errors[mom - 1]

    return df_res_mom_acc

def ph_density(t, alpha, T):
    """Evaluate PH density f(t) = α e^{Tt} t_vec"""
    t_vec = -T @ np.ones((T.shape[0], 1))
    e_Tt = scipy.linalg.expm(T * t)
    return float(alpha @ e_Tt @ t_vec)

def main(orig_dist_type):
    if sys.platform == 'linux':

        dump_path = '/scratch/eliransc/mom_anal/fit_6_moms_examples'
    else:
        dump_path = r'C:\Users\Eshel\workspace\data\moment_anal'


    mod_num = np.random.randint(1,10000000)
    if orig_dist_type == 'cox':

        has_maximum = False
        while not has_maximum:

            rand_val = np.random.rand()
            if rand_val < 0.4:
                ph_size = np.random.randint(6, 11)
            elif rand_val < 0.95:
                ph_size = np.random.randint(11, 80)
            else:
                ph_size = np.random.randint(80, 150)

            a_orig, T_orig, moms, something = sample_coxian(ph_size, 10)
            t_values = np.linspace(0.01, 3, 250)
            f_values = np.array([ph_density(t, a_orig, T_orig) for t in t_values])

            # Find local maxima
            local_maxima_indices = argrelextrema(f_values, np.greater)[0]

            # Check if at least one maximum
            has_maximum = len(local_maxima_indices) > 0

            mm = torch.tensor(np.array(compute_first_n_moments(a_orig, T_orig, 10)).ravel())

            a_orig, T_orig = torch.tensor(a_orig), torch.tensor(T_orig)

        # plt.plot(t_values, f_values, label="PH density")
        # plt.plot(t_values[local_maxima_indices], f_values[local_maxima_indices], 'ro', label="Local maxima")
        # plt.xlabel("t")
        # plt.ylabel("f(t)")
        # plt.legend()
        # plt.title("Phase-Type Density")
        # plt.grid()
        # plt.show()

    elif orig_dist_type == 'general':
        rand_val = np.random.rand()
        if rand_val < 0.4:
            ph_size = np.random.randint(6, 11)
        elif rand_val < 0.95:
            ph_size = np.random.randint(11, 80)
        else:
            ph_size = np.random.randint(80, 150)

        a_orig, T_orig, mm = get_PH_general_with_zeros(ph_size)
    else:
        rand_val = np.random.rand()
        if rand_val < 0.4:
            ph_size = np.random.randint(6, 11)
        elif rand_val < 0.9:
            ph_size = np.random.randint(11, 80)
        else:
            ph_size = np.random.randint(80, 150)

        a_orig, T_orig, moms = sample(ph_size)
        moms = np.array(moms).flatten()
        mm = torch.tensor(moms)
        a_orig = torch.tensor(a_orig)
        T_orig = torch.tensor(T_orig)



    m1, m2, m3, m4 = mm[:4]
    skew, kurt = compute_skewness_and_kurtosis_from_raw(m1, m2, m3, m4)
    skew, kurt = skew.item(), kurt.item()
    scv = ((m2 - m1 ** 2) / m1 ** 2).item()

    num_rep = 5000
    lr_gamma = 0.9
    init_drop = 0.9

    type_ph = 'hyper'
    num_epochs = 65000

    if ph_size < 30:
        pass
    elif ph_size < 80:
        ph_size = int(ph_size / 2)
    else:
        ph_size = int(ph_size / 2.5)

    block_sizes = create_block_sizes_new(ph_size)


    m = HyperErlangMatcher(block_sizes=block_sizes, n_replica=num_rep, num_epochs=num_epochs, lr=5e-3,
                               lr_gamma=lr_gamma)

    min_loss_dict = {2: 1e-11, 3: 1e-12, 4: 1e-13, 5: 1e-13, 6: 1e-13}
    min_loss_dict_errors = {2: 0.01, 3: 0.009, 4: 0.008, 5: 0.007, 6: 0.006}
    now = time.time()
    torch.set_default_dtype(torch.double)
    # Arrive
    moments = mm[:6]  # torch.tensor([1.00000000e+00, 6.05028076e+00, 1.95626417e+02, 1.05513745e+04,  7.22395741e+05])
    # Service
    # moments = torch.tensor([1.00000000e+00, 2.55408739e+00, 9.81573904e+00, 4.75758155e+01, 2.73684422e+02])
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Original size of PH: ', ph_size)
    print('Original PH type: ', orig_dist_type)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    res_PH = {}
    for num_moms in [2, 3, 4, 5, 6]:
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(num_moms)
        while True:

            k = np.array(block_sizes).sum()

            m = HyperErlangMatcher(block_sizes=block_sizes, n_replica=num_rep, num_epochs=num_epochs, lr=5e-3,
                                   lr_gamma=lr_gamma)

            if True:
                print('begin with: ', block_sizes)

                m.fit(target_ms=moments[:num_moms], min_loss=min_loss_dict[num_moms],
                      stop=[{"epoch": 500, "keep_fraction": .2},
                            {"epoch": 5000, "keep_fraction": .1},
                            {"epoch": 15000, "keep_fraction": .1}
                            ])
                a, T = m.get_best_after_fit()

                moment_table = moment_analytics(moments[:num_moms],
                                                compute_moments(a.to('cpu'), T.to('cpu'), k, len(moments[:num_moms])))
                print(moment_table)
                curr_score = moment_table['delta-relative'].abs().max().item()

                if curr_score < min_loss_dict_errors[num_moms]:
                    res_PH[num_moms] = (a, T, moment_table)
                    break
                # if num_moms == 2:
                #     if curr_score < min_loss_dict_errors[num_moms]:
                #         res_PH[num_moms] = (a, T, moment_table)
                #         break
                #
                # else:
                #     # if (res_PH[num_moms - 1][-1].loc[:num_moms - 1, 'delta-relative'].abs() < moment_table.loc[
                #     #                                                                           :num_moms - 2,
                #     #                                                                           'delta-relative'].abs()).sum().item() == 0:
                #     if curr_score < min_loss_dict_errors[num_moms]:
                #
                #         res_PH[num_moms] = (a, T, moment_table)
                #         break



            ph_size += 3

            block_sizes = create_block_sizes_new(ph_size)

            print('new ph size is: ', ph_size)
        file_name = 'mod_num_' + str(mod_num) + '_num_diff_moms_' + str(len(list(res_PH.keys()))) + '_orig_type_' + orig_dist_type +'_origsize_' + str(ph_size) + '.pkl'

        try:
            df_res_mom_acc = compute_kl_mom_error(res_PH, a_orig, T_orig, mm)
            pkl.dump((res_PH, a_orig, T_orig, mm, scv, skew, kurt, df_res_mom_acc), open(os.path.join(dump_path,file_name), 'wb'))
        except:
            print('res_PH not defined')

    try:
        pkl.dump((res_PH, a_orig, T_orig, mm, scv, skew, kurt, df_res_mom_acc), open(os.path.join(dump_path,file_name), 'wb'))
    except:
        print('res_PH not defined')



if __name__ == "__main__":

    for ind in range(5000):

        my_list = ['cox' or 'erlang' or 'general']
        item = random.choice(my_list)
        orig_dist_type = item # 'cox' or 'erlang' or 'general'

        main(orig_dist_type)
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
