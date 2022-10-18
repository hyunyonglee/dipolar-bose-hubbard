# Copyright 2022 Hyun-Yong Lee

import numpy as np
import model
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms import tebd
import os
import os.path
import sys
import matplotlib.pyplot as plt
import pickle

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

import logging.config
conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
}
logging.config.dictConfig(conf)

L = int(sys.argv[1])
t = float(sys.argv[2])
tp = float(sys.argv[3])
U = float(sys.argv[4])
Ut = float(sys.argv[5])
mu = float(sys.argv[6])
Ncut = int(sys.argv[7])
CHI = int(sys.argv[8])
RM = sys.argv[9]
QN = sys.argv[10]
PATH = sys.argv[11]
BC_MPS = sys.argv[12]
BC = sys.argv[13]
IS = sys.argv[14]
TOL = float(sys.argv[15])
h = float(sys.argv[16])
EXC = sys.argv[17]

model_params = {
    "L": L,
    "t": t,
    "tp": tp,
    "h": h,
    "U": U,
    "Ut": Ut,
    "mu": mu,
    "Ncut": Ncut,
    "bc_MPS": BC_MPS,
    "bc": BC,
    "QN": QN
}

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.DIPOLAR_BOSE_HUBBARD(model_params)

# initial state
if IS == '1-1p':
    product_state = ['1'] * M.lat.N_sites
    product_state[int(L/2)] = '2'

elif IS == '1-2p':
    product_state = ['1'] * M.lat.N_sites
    product_state[int(L/2)] = '2'
    product_state[int(L/2)+1] = '2'
    
elif IS == '1-1h':
    product_state = ['1'] * M.lat.N_sites
    product_state[int(L/2)] = '0'
    
elif IS == '1-2h':
    product_state = ['1'] * M.lat.N_sites
    product_state[int(L/2)] = '0'
    product_state[int(L/2)+1] = '0'

elif IS == 'half':
    product_state = ['1','0'] * int(M.lat.N_sites/2)
    
elif IS == 'half-1p':
    product_state = ['1','0'] * int(M.lat.N_sites/2)
    product_state[int(M.lat.N_sites/2)+1] = '1'

elif IS == 'half-1h':
    product_state = ['1','0'] * int(M.lat.N_sites/2)
    product_state[int(M.lat.N_sites/2)] = '0'

elif IS == '0-1half':
    product_state = ['1','1','0','0'] * int(M.lat.N_sites/4)
    
elif IS == '1-1half':
    product_state = ['2','2','1','1'] * int(M.lat.N_sites/4)
    
elif IS == '1-1half+1':
    product_state = ['2','2','1','1'] * int(M.lat.N_sites/4)
    product_state[2] = '2'
    
elif IS == '1-1half-1':
    product_state = ['2','2','1','1'] * int(M.lat.N_sites/4)
    product_state[1] = '1'
    
elif IS == '1.5+1':
    product_state = ['1','2'] * int(M.lat.N_sites/2)
    product_state[0] = '2'
    
elif IS == '1.5-1':
    product_state = ['1','2'] * int(M.lat.N_sites/2)
    product_state[1] = '1'
    
elif IS == '2+1':
    product_state = ['2'] * M.lat.N_sites
    product_state[int(M.lat.N_sites/2)] = '3'
    
elif IS == '2-1':
    product_state = ['2'] * M.lat.N_sites
    product_state[int(M.lat.N_sites/2)] = '1'
    
elif IS == '2-1half':
    product_state = ['3','3','2','2'] * int(M.lat.N_sites/4)
    
elif IS == '3-1half':
    product_state = ['4','4','3','3'] * int(M.lat.N_sites/4)
    
elif any( IS == frac for frac in ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9',
    '1.1','1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9', 
    '2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8','2.9',
    '3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','4.5','5.5'] ):
    
    n = float(IS)
    dn = np.remainder(n,1)

    product_state = [str(int(np.floor(n)))] * M.lat.N_sites
    Ls = round(dn*M.lat.N_sites)

    a = 0
    I = 0
    for i in range(M.lat.N_sites):

        s = 2*i
        if s < M.lat.N_sites:
            a = 0    
        else:
            a = M.lat.N_sites-1
    
        product_state[s-a] = str(int(product_state[s-a])+1)
        I = I+1
        if I==Ls:
            break

else:
    product_state = [IS] * M.lat.N_sites
    
psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)


if RM == 'random':
    TEBD_params = {'N_steps': 4, 'trunc_params':{'chi_max': 4}, 'verbose': 0}
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()
    psi.canonical_form() 

# dchi = int(CHI/5)
# chi_list = {0: 8, 10: 16, 20: 32, 30: CHI}
# chi_list = {0: 4, 4: 8, 8: 16, 12: 32, 16: 64, 20: CHI}
# for i in range(5):
#     chi_list[i*20] = (i+1)*dchi

if BC_MPS == 'infinite':
    max_sweep = 500
    disable_after = 200
    S_err = TOL
else:
    max_sweep = 500
    disable_after = 30
    S_err = TOL

dmrg_params = {
    # 'mixer': True,  # setting this to True helps to escape local minima
    'mixer' : dmrg.SubspaceExpansion,
    'mixer_params': {
        'amplitude': 1.e-2,
        'decay': 3.0,
        'disable_after': disable_after
    },
    'trunc_params': {
        'chi_max': CHI,
        'svd_min': 1.e-9
    },
    # 'chi_list': chi_list,
    'max_E_err': 1.0e-7,
    'max_S_err': S_err,
    'max_sweeps': max_sweep,
    # 'combine' : True
}

ensure_dir(PATH + "observables/")
ensure_dir(PATH + "entanglement/")
ensure_dir(PATH + "logs/")
ensure_dir(PATH + "mps/")

# ground state
eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
psi.canonical_form() 

N = psi.expectation_value("N")
B = np.abs( psi.expectation_value("B") )
EE = psi.entanglement_entropy()
ES = psi.entanglement_spectrum()

if BC_MPS == 'finite':
    R = L-1
    xi = 0.
    I0 = int(L/3)
    R_CORR = int(L/3)
    
else:
    R = L
    xi = psi.correlation_length()
    I0 = 0
    R_CORR = 100
    
# measuring exciton condensation
hs = []
if BC_MPS == 'finite' and BC == 'periodic':
    for i in range(0,int(L/2-1)): 
        I = 2*i
        hs.append( np.abs( psi.expectation_value_term([('Bd',I+2),('B',I)]) ) )
    hs.append( np.abs( psi.expectation_value_term([('Bd',L-1),('B',L-2 )]) ) )
    for i in range(0,int(L/2-1)):
        I = L-1 - 2*i
        hs.append( np.abs( psi.expectation_value_term([('Bd',I-2),('B',I)]) ) )
    hs.append( np.abs( psi.expectation_value_term([('Bd',0),('B',1)]) ) )

else:
    for i in range(0,R): 
        hs.append( np.abs( psi.expectation_value_term([('Bd',i+1),('B',i)]) ) )
#


# measuring correlation functions
cor_bb = []
cor_dd = []
cor_dd_conn = []
for i in range(R_CORR):

    cor = psi.expectation_value_term([('Bd',I0),('B',I0+1),('B',I0+2+i),('Bd',I0+3+i)])
    bb1 = psi.expectation_value_term([('Bd',I0),('B',I0+1)])
    bb2 = psi.expectation_value_term([('B',I0+2+i),('Bd',I0+3+i)])
    
    cor_bb.append( np.abs( psi.expectation_value_term([('Bd',I0),('B',I0+1+i)]) ) )
    cor_dd.append( np.abs( cor ) )
    cor_dd_conn.append( np.abs( cor - bb1*bb2 ) )   
#


#
if EXC == 'ON':
    dmrg_params['orthogonal_to'] = [psi]
    psi1 = psi.copy()
    eng1 = dmrg.TwoSiteDMRGEngine(psi1, M, dmrg_params)
    E1, psi1 = eng1.run()  # equivalent to dmrg.run() up to the return parameters.
    gap = E1 - E

    with open( PATH + 'mps/exc_t_%.2f_tp_%.2f_U%.2f_Ut%.2f_mu%.2f.pkl' % (t,tp,U,Ut,mu), 'wb') as f:
        pickle.dump(psi1, f)

else:
    gap = 0.
#


file1 = open( PATH + "observables/energy.txt","a")
file1.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(Ut) + " " + repr(mu) + " " + repr(E) + " " + repr( np.mean(N) ) + " " + repr( np.mean(B) ) + " " + repr( np.mean(hs) ) + " " + repr(xi) + " " + repr(gap) + " " + "\n")

file2 = open( PATH + "observables/numbers.txt","a")
file2.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(Ut) + " " + repr(mu) + " " + "  ".join(map(str, N)) + " " + "\n")

file3 = open( PATH + "observables/exciton_density.txt","a")
file3.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(Ut) + " " + repr(mu) + " " + "  ".join(map(str, hs)) + " " + "\n")

file4 = open( PATH + "observables/condensation.txt","a")
file4.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(Ut) + " " + repr(mu) + " " + "  ".join(map(str, B)) + " " + "\n")

file5 = open( PATH + "observables/entanglement_entropy.txt","a")
file5.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(Ut) + " " + repr(mu) + " " + "  ".join(map(str, EE)) + " " + "\n")

file6 = open( PATH + "observables/corr_bb.txt","a")
file6.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(Ut) + " " + repr(mu) + " " + "  ".join(map(str, cor_bb)) + " " + "\n")

file7 = open( PATH + "observables/corr_dd.txt","a")
file7.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(Ut) + " " + repr(mu) + " " + "  ".join(map(str, cor_dd)) + " " + "\n")

file8 = open( PATH + "observables/corr_dd_conn.txt","a")
file8.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(Ut) + " " + repr(mu) + " " + "  ".join(map(str, cor_dd_conn)) + " " + "\n")

file_ES = open( PATH + "entanglement/es_t_%.3f_tp_%.3f_U_%.2f_Ut_%.2f_mu_%.2f.txt" % (t,tp,U,Ut,mu),"a")
for i in range(0,R):
    file_ES.write("  ".join(map(str, ES[i])) + " " + "\n")
file_EE = open( PATH + "entanglement/ee_t_%.3f_tp_%.3f_U_%.2f_Ut_%.2f_mu_%.2f.txt" % (t,tp,U,Ut,mu),"a")
file_EE.write("  ".join(map(str, EE)) + " " + "\n")

file_STAT = open( PATH + "logs/stat_t_%.3f_tp_%.3f_U_%.2f_Ut_%.2f_mu_%.2f.txt" % (t,tp,U,Ut,mu),"a")
file_STAT.write("  ".join(map(str,eng.sweep_stats['E'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['S'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['max_trunc_err'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['norm_err'])) + " " + "\n")

with open( PATH + 'mps/gs_t_%.2f_tp_%.2f_U%.2f_Ut%.2f_mu%.2f.pkl' % (t,tp,U,Ut,mu), 'wb') as f:
    pickle.dump(psi, f)



print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
