import pickle as pkl
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('/home/eliransc/projects/def-dkrass/eliransc/one.deep.moment'))
sys.path.append(os.path.abspath(r"C:\Users\Eshel\workspace\one.deep.moment"))
sys.path.append('../')
from utils import *
from utils_sample_ph import *
import torch



for example in range(500):

    example_num = np.random.randint(0, 1000000)
    if sys.platform == 'linux':

        path_dump = os.path.join('/scratch/eliransc/mom_anal/gg1_util/only_arrivals',str(example_num))
        moms_math_path = '/scratch/eliransc/mom_anal/fit_6_moms_examples'

    else:
        path_dump =  os.path.join(r'C:\Users\Eshel\workspace\data\mom_analysis\gg1_util\only_arrivals',  str(example_num))
        moms_math_path = r'C:\Users\Eshel\workspace\data\fit_6_moms_examples'

    os.mkdir(path_dump)


    files = os.listdir(moms_math_path)
    files_all = [file for file in files if 'moms_5' in file]
    chosen_file = np.random.choice(files_all).item()


    res_PH_arrive = pkl.load(open(os.path.join(moms_math_path,chosen_file), 'rb'))
    res_PH_arrive_fitted = res_PH_arrive[0]


    ## Sample the true moments for the service process

    rand_val = np.random.rand()

    if rand_val < 0.4:
        ph_size = np.random.randint(6, 11)
    elif rand_val < 0.9:
        ph_size = np.random.randint(11, 80)
    else:
        ph_size = np.random.randint(80, 150)

    ## General

    # orig_type = np.choose(['cox', 'hyper', 'general'])
    orig_type = np.random.choice(['cox', 'hyper', 'general']).item()

    if orig_type == 'general':
        a, T, mm = get_PH_general_with_zeros(ph_size)
        mm  = np.array(mm)
    elif orig_type == 'hyper':
        a, T, _, _ = sample_coxian(ph_size, ph_size)
        mm = np.array(compute_first_n_moments(a, T, 10)).ravel()
    else:
        a, T, moms = sample(ph_size)
        mm = np.array(compute_first_n_moments(a, T, 10)).ravel()


    var = mm[1] - mm[0] ** 2
    skew, kurt = compute_skewness_and_kurtosis_from_raw(mm[0], mm[1], mm[2], mm[3])
    skew_ser = skew.item()
    kurt_ser = kurt.item()
    var_ser = var.item()
    ser_data = (var_ser, skew_ser, kurt_ser, mm)

    a_ser_true = a.reshape(1, -1)
    T_ser_true = T
    moms_arrive_true = mm

    a_arrive_true =  res_PH_arrive[1]
    T_arrive_true = res_PH_arrive[2]
    moms_arrive_true = res_PH_arrive[3]


    for rho in np.linspace(0.001, 0.99, 50):

        for num_mom in [1, 2,3, 4, 5, 6]:

            if num_mom == 1:
                a_arrive = a_arrive_true.reshape(1,-1)
                T_arrive = T_arrive_true
            else:

                a_arrive = res_PH_arrive_fitted[num_mom][0].reshape(1, -1)
                T_arrive = res_PH_arrive_fitted[num_mom][1]


            with torch.no_grad():
                a_arrive = np.array(a_arrive.to('cpu'))
                T_arrive = np.array(T_arrive.to('cpu'))

            a_ser = np.array(a_ser_true.reshape(1, -1))
            T_ser = np.array(T_ser_true)

            T_arrive  = T_arrive*rho
            print(num_mom, rho)
            stead = compute_steady(a_arrive, T_arrive, a_ser, T_ser)


            file_name = 'gg1_QBD_orig_rho_' +str(rho)+'_num_moms_' + str(num_mom) + '_example_' + str(example_num) +  '.pkl'

            pkl.dump((stead, res_PH_arrive, ser_data), open(os.path.join(path_dump, file_name),'wb'))





