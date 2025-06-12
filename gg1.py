import pickle as pkl
import numpy as np
import os
import sys
sys.path.append('./')
from utils import *

import torch


def load_PH(path, file_name):

    full_path = os.path.join(path, file_name)
    return pkl.load(open(full_path, 'rb'))


res_PH_ser_fitted = pkl.load(open(os.path.join('./notebooks','ser_fitted.pkl'), 'rb'))
res_PH_arrive_fitted = pkl.load(open(os.path.join('./notebooks','arrive_fitted.pkl'), 'rb'))
a_ser_true, T_ser_true, moms_ser_true = pkl.load( open(os.path.join('./notebooks','ser_true.pkl'), 'rb'))
a_arrive_true, T_arrive_true, moms_arrive_true = pkl.load( open(os.path.join('./notebooks','arrive_true.pkl'), 'rb'))


# path = r'C:\Users\Eshel\workspace\one.deep.moment\optimize'
# res_2_arrive_name = 'a_T_arrive_2_new.pkl'
# res_3_arrive_name = 'a_T_arrive_3_new.pkl'
# res_4_arrive_name = 'a_T_arrive_4_new.pkl'
# res_5_arrive_name =  'a_T_arrive_5_new.pkl'
#
# arrive_orig =  'queue_case_study_arrival_new.pkl' # 'arrive_orig.pkl'
# ser_orig = 'ser_orig.pkl'
#
# res_2_name_ser = 'a_T_ser_2.pkl'
# res_3_name_ser = 'a_T_ser_3.pkl'
# res_4_name_ser = 'a_T_ser_4.pkl'
# res_5_name_ser = 'a_T_ser_5_d.pkl'

# files_name_dict_arrive = {1:arrive_orig, 2: res_2_arrive_name,  3: res_3_arrive_name,  4: res_4_arrive_name, 5: res_5_arrive_name}
# files_name_dict_ser = {1:ser_orig, 2: res_2_name_ser,  3: res_3_name_ser,  4: res_4_name_ser, 5: res_5_name_ser}



for ind in range(500):


    rho = np.random.uniform(0.01,0.98)

    for num_mom in [1, 2,3, 4]:

        if num_mom == 1:
            # a_arrive = a_arrive_true.reshape(1,-1)
            # T_arrive = T_arrive_true
            a_ser = a_ser_true.reshape(1,-1)
            T_ser = T_ser_true
        else:

            # a_arrive = res_PH_arrive_fitted[num_mom][0].reshape(1,-1)
            # T_arrive = res_PH_arrive_fitted[num_mom][1]
            a_ser = res_PH_ser_fitted[num_mom][0].reshape(1,-1)
            T_ser = res_PH_ser_fitted[num_mom][1]

        if num_mom > 1:
            with torch.no_grad():
                # a_arrive = np.array(a_arrive.to('cpu'))
                a_ser = np.array(a_ser.to('cpu'))
                # T_arrive = np.array(T_arrive.to('cpu'))
                T_ser = np.array(T_ser.to('cpu'))

        a_arrive = a_arrive_true.reshape(1,-1)
        T_arrive = T_arrive_true
        # a_ser = a_ser_true.reshape(1, -1)
        # T_ser = T_ser_true

        T_arrive  = T_arrive*rho
        print(num_mom, rho)
        stead = compute_steady(a_arrive, T_arrive, a_ser, T_ser)

        # stead = compute_steady(a_arrive.astype(np.float64)/a_arrive.astype(np.float64).sum(), T_arrive.astype(np.float64), a_ser.astype(np.float64)/a_ser.astype(np.float64).sum(), T_ser.astype(np.float64))
        print(stead[:10],  rho, num_mom)
        file_name = 'gg1_QBD_orig_rho_' +str(rho)+'_num_moms_' + str(num_mom) + '.pkl'
        path_dump = r'./data_new_fitted_ser'
        pkl.dump(stead, open(os.path.join(path_dump, file_name),'wb'))





