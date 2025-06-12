import pickle as pkl
import numpy as np
import os
import sys
sys.path.append('../')
from utils import *

import torch


res_PH_arrive = pkl.load(open(os.path.join(r'C:\Users\Eshel\workspace\data\moment_anal','mod_num_1518325_num_diff_moms_5full.pkl'), 'rb'))
res_PH_arrive_fitted = res_PH_arrive[0]

res_ser = pkl.load(open(os.path.join(r'C:\Users\Eshel\workspace\data\moment_anal','only_ground_truth_2618526.pkl'), 'rb'))
a_ser_true = res_ser[0]
T_ser_true = res_ser[1]
moms_arrive_true = res_ser[2]

# res_PH_ser_fitted = pkl.load(open(os.path.join('./notebooks','ser_fitted.pkl'), 'rb'))
# res_PH_arrive_fitted = pkl.load(open(os.path.join('./notebooks','arrive_fitted.pkl'), 'rb'))
# a_ser_true, T_ser_true, moms_ser_true = pkl.load( open(os.path.join('./notebooks','ser_true.pkl'), 'rb'))
# a_arrive_true, T_arrive_true, moms_arrive_true = pkl.load( open(os.path.join('./notebooks','arrive_true.pkl'), 'rb'))

a_arrive_true =  res_PH_arrive[1]
T_arrive_true = res_PH_arrive[2]
moms_arrive_true = res_PH_arrive[3]





for ind in range(500):


    rho = np.random.uniform(0.01,0.98)

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

        # stead = compute_steady(a_arrive.astype(np.float64)/a_arrive.astype(np.float64).sum(), T_arrive.astype(np.float64), a_ser.astype(np.float64)/a_ser.astype(np.float64).sum(), T_ser.astype(np.float64))
        print(stead[:10],  rho, num_mom)
        file_name = 'gg1_QBD_orig_rho_' +str(rho)+'_num_moms_' + str(num_mom) + '.pkl'
        path_dump = r'../data_new_fitted_arrive'
        pkl.dump(stead, open(os.path.join(path_dump, file_name),'wb'))





