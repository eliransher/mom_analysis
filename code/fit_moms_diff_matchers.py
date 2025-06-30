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


def compute_kl_mom_error_row(res_PH, a_orig, T_orig, mm, num_moms, eps):
    curr_a = res_PH[0]
    curr_T = res_PH[1]
    with torch.no_grad():
        kl_res = kl_divergence(curr_a.cpu().numpy(), curr_T.cpu().numpy(), a_orig.cpu().numpy().flatten(),
                                    T_orig.cpu().numpy())

    df_res_mom_acc = pd.DataFrame([])


    with torch.no_grad():

        momss = compute_moments(curr_a.cpu(), curr_T.cpu(), curr_T.shape[0], 10)
        curr_errors = torch.abs(100 * (torch.tensor(momss).flatten() - mm) / mm)
        curr_errors = np.array(curr_errors)
        curr_ind = df_res_mom_acc.shape[0]
        df_res_mom_acc.loc[curr_ind, 'num_mom_error'] = num_moms
        df_res_mom_acc.loc[curr_ind, 'eps'] = eps
        df_res_mom_acc.loc[curr_ind, 'kl_div'] = kl_res
        for mom in range(1, 1 + curr_errors.shape[0]):
            df_res_mom_acc.loc[curr_ind, str(mom)] = curr_errors[mom - 1]

    return df_res_mom_acc

def ph_density(t, alpha, T):
    """Evaluate PH density f(t) = α e^{Tt} t_vec"""
    t_vec = -T @ np.ones((T.shape[0], 1))
    e_Tt = scipy.linalg.expm(T * t)
    return float(alpha @ e_Tt @ t_vec)

def main():
    if sys.platform == 'linux':
        if os.getcwd() == '/home/management/projects/elirans/elirans/mom_analysis/code':
            dump_path = '/home/elirans/scratch/mom_match_res'
            dist_path = '/home/elirans/scratch/just_dists'
        else:
            dump_path = '/scratch/eliransc/mom_anal/fit_7_moms_examples'
            dist_path = '/scratch/eliransc/orig_dists'

    else:
        dump_path = r'C:\Users\Eshel\workspace\data\moment_anal\fixed_error_mom_match'
        dist_path = r'C:\Users\Eshel\workspace\data\moment_anal\just_dists'


    mod_num = np.random.randint(1,10000000)


    files = os.listdir(dist_path)
    file_rand = np.random.choice(files).item()
    orig_dist_type = file_rand.split('_')[3]
    print('file dist: ', file_rand)
    try:
        a_orig, T_orig, mm, scv, skew, kurt = pkl.load(open(os.path.join(dist_path, file_rand), 'rb'))

    except:
        os.remove(os.path.join(dist_path, file_rand))
        print('Error loading file: ', file_rand)
        files = os.listdir(dist_path)
        file_rand = np.random.choice(files).item()
        orig_dist_type = file_rand.split('_')[3]
        print('file dist: ', file_rand)
        a_orig, T_orig, mm, scv, skew, kurt = pkl.load(open(os.path.join(dist_path, file_rand), 'rb'))


    orig_ph_size = T_orig.shape[0]
    ph_size = T_orig.shape[0]
    num_rep = 5000
    lr_gamma = 0.9
    init_drop = 0.9
    dist_code = file_rand.split('_')[0]


    num_epochs = 65000

    if ph_size < 30:
        pass
    elif ph_size < 80:
        ph_size = int(ph_size / 2)
    else:
        ph_size = int(ph_size / 2.5)


    min_loss_dict = {2: 1e-7, 3: 1e-7, 4: 1e-7, 5: 1e-7, 6: 1e-7, 7: 1e-7}
    min_loss_dict_errors_fitted = {2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}
    min_loss_dict_errors_not_fitted = 1
    now = time.time()
    torch.set_default_dtype(torch.double)
    # Arrive
    moments = mm[:7]  # torch.tensor([1.00000000e+00, 6.05028076e+00, 1.95626417e+02, 1.05513745e+04,  7.22395741e+05])
    # Service
    # moments = torch.tensor([1.00000000e+00, 2.55408739e+00, 9.81573904e+00, 4.75758155e+01, 2.73684422e+02])
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Original size of PH: ', ph_size)
    print('Original PH type: ', orig_dist_type)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    errs = np.array([0.05, 0.1, 0.25, 0.5, 1.2, 2, 2])



    pairs = []

    for ind in range(500):

        num_moms = np.random.choice([2, 3, 4, 5, 6, 7])
        eps = np.random.choice([-10, -5, -2])
        print(num_moms, eps)

        cat_dim1 = mm[:7].clone()
        cat_dim1[num_moms - 1] = mm[num_moms - 1] * (1 - eps / 100)

        new_eps = eps
        for mom_val in range(num_moms + 1, 8):
            new_eps = new_eps * 1.25
            # print(mom_val, new_eps*1.25)
            cat_dim1[mom_val - 1] = mm[mom_val - 1] * (1 - new_eps / 100)
            print(mom_val, mm[mom_val - 1], mm[mom_val - 1] * (1 - new_eps / 100), new_eps)

        curr_pair = (num_moms, eps)
        print('num_moms: ', num_moms, 'eps: ', eps)
        if  curr_pair in pairs:

            print('Pair already exists, skipping...')
            continue
        else:
            pairs.append(curr_pair)



        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

        while True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weights = torch.ones(cat_dim1.shape[0]).double().to(device)
            weights = weights*torch.tensor([2.5,2.5,2.5,1.5,1,1,1]).to(device)


            elements = ['general', 'cox', 'hyper']
            probabilities = [ 0.05, 0.15, 0.85]  # Must sum to 1

            type_ph = random.choices(elements, weights=probabilities, k=1)[0]
            print('###################',  type_ph)

            if type_ph == 'general':

                m = GeneralPHMatcher(ph_size=ph_size, num_epochs=num_epochs, lr=5e-3, n_replica=num_rep, lr_gamma=lr_gamma, normalize_m1=True,
                                    init_drop=init_drop, weights = weights)
            elif type_ph == 'cox':

                m = CoxianPHMatcher(ph_size=ph_size, num_epochs=num_epochs, lr=5e-3, lr_gamma=lr_gamma, n_replica=num_rep, weights = weights)

            elif type_ph == 'hyper':

                block_sizes = create_block_sizes_new(ph_size)

                m = HyperErlangMatcher(block_sizes=block_sizes, n_replica=num_rep, num_epochs=num_epochs, lr=5e-3,
                                   lr_gamma=lr_gamma, weights = weights)

            if True:
                if type_ph == 'hyper':
                    print('begin with: ', block_sizes)
                else:
                    print('begin with: ', ph_size)

                m.fit(target_ms=cat_dim1, min_loss=min_loss_dict[num_moms],
                      stop=[{"epoch": 500, "keep_fraction": .2},
                            {"epoch": 5000, "keep_fraction": .1},
                            {"epoch": 15000, "keep_fraction": .1}
                            ])
                if True:
                    a, T = m.get_best_after_fit()

                    moment_table = moment_analytics(cat_dim1,
                                                    compute_moments(a.to('cpu'), T.to('cpu'), ph_size, cat_dim1.shape[0]))
                    print(moment_table)
                    curr_score = np.array(moment_table['delta-relative'].abs())
                    if (curr_score < errs).sum() == errs.shape[0]:

                        res_PH = (a, T, moment_table)
                        break
                # except:
                #     print('Error in fitting')


            ph_size += 5

            block_sizes = create_block_sizes_new(ph_size)

            print('new ph size is: ', ph_size)

        file_name = 'mod_num_' + str(mod_num) + '_distcode_' +str(dist_code) + '_mom_' +str(num_moms) + '_eps_' + str(eps)   + '_orig_type_' + orig_dist_type +'_origsize_' + str(orig_ph_size) + '.pkl'

        if True:
            df_res_mom_acc = compute_kl_mom_error_row(res_PH, a_orig, T_orig, mm, num_moms, eps)
            pkl.dump((res_PH, a_orig, T_orig, mm, scv, skew, kurt, df_res_mom_acc),
                     open(os.path.join(dump_path, file_name), 'wb'))
        # except:
        #     print('res_PH not defined')

    # try:
    #     pkl.dump((res_PH, a_orig, T_orig, mm, scv, skew, kurt, df_res_mom_acc), open(os.path.join(dump_path, file_name), 'wb'))
    # except:
    #     print('res_PH not defined')



if __name__ == "__main__":

    main()