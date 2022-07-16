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

os.environ["OMP_NUM_THREADS"] = "68"

L = int(sys.argv[1])
t = float(sys.argv[2])
tp = float(sys.argv[3])
U = float(sys.argv[4])
mu = float(sys.argv[5])
Ncut = int(sys.argv[6])
CHI = int(sys.argv[7])
RM = sys.argv[8]
QN = sys.argv[9]
PATH = sys.argv[10]
BC_MPS = sys.argv[11]
BC = sys.argv[12]
IS = sys.argv[13]

model_params = {
    "L": L,
    "t": t,
    "tp": tp,
    "U": U,
    "mu": mu,
    "Ncut": Ncut,
    "bc_MPS": BC_MPS,
    "bc": BC,
    "QN": QN
}

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.DIPOLAR_BOSE_HUBBARD(model_params)

product_state = [IS] * M.lat.N_sites
psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

if RM == 'random':
    TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': 20}, 'verbose': 0}
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()
    psi.canonical_form() 

dchi = int(CHI/5)
chi_list = {}#{10: 32, 20: 64, 30: CHI}
for i in range(5):
    chi_list[i*20] = (i+1)*dchi

if BC_MPS == 'infinite':
    max_sweep = 1000
else:
    max_sweep = 50

dmrg_params = {
    'mixer': True,  # setting this to True helps to escape local minima
    'mixer_params': {
        'amplitude': 1.e-5,
        'decay': 1.2,
        'disable_after': 150
    },
    'trunc_params': {
        'chi_max': CHI,
        'svd_min': 1.e-9
    },
    'lanczos_params': {
            'N_min': 5,
            'N_max': 20
    },
    'chi_list': chi_list,
    'max_E_err': 1.0e-8,
    'max_S_err': 1.0e-6,
    'max_sweeps': max_sweep,
    'norm_tol' : 1.0e-4
}

ensure_dir(PATH + "observables/")
ensure_dir(PATH + "entanglement/")
ensure_dir(PATH + "logs/")
ensure_dir(PATH + "mps/")

# ground state
eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.

N = psi.expectation_value("N")
B = np.abs( psi.expectation_value("B") )
EE = psi.entanglement_entropy()
ES = psi.entanglement_spectrum()

if BC_MPS == 'finite':
    R = L-1
    xi = 0.
else:
    R = L
    xi = psi.correlation_length()

# measuring exciton condensation
hs = []
for i in range(0,R): 
    hs.append( np.abs( psi.expectation_value_term([('Bd',i+1),('B',i)]) ) )
#

# excited state
if BC_MPS == 'finite':
    dmrg_params['orthogonal_to'] = [psi]
    psi1 = psi.copy()  # MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    eng1 = dmrg.TwoSiteDMRGEngine(psi1, M, dmrg_params)
    E1, psi1 = eng1.run()  # equivalent to dmrg.run() up to the return parameters.

    with open( PATH + 'mps/exs_t_%.2f_U%.2f_mu%.2f.pkl' % (t,U,mu), 'wb') as f:
    	pickle.dump(psi1, f)

else:
    # resume_psi = eng.get_resume_data(sequential_simulations=True)
    # M1 = M.extract_segment(enlarge=10)
    # first, last = M1.lat.segment_first_last
    
    # psi_s = psi.extract_segment(first, last)
    # init_env_data = eng.env.get_initialization_data(first, last)

    # psi1 = psi_s.copy()  # TODO: perturb this a little bit
    # resume_psi1 = {'init_env_data': init_env_data}

    # eng1 = dmrg.TwoSiteDMRGEngine(psi1, M1, dmrg_params, resume_data=resume_psi1)
    # E1, psi1 = eng1.run()
    E1 = E

#
gap = E1 - E

file1 = open( PATH + "observables/energy.txt","a")
file1.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + repr(E) + " " + repr( np.mean(N) ) + " " + repr( np.mean(B) ) + " " + repr( np.mean(hs) ) + " " + repr(xi) + " " + repr(gap) + " " + "\n")

file2 = open( PATH + "observables/numbers.txt","a")
file2.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, N)) + " " + "\n")

file3 = open( PATH + "observables/exciton_density.txt","a")
file3.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, hs)) + " " + "\n")

file4 = open( PATH + "observables/condensation.txt","a")
file4.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, B)) + " " + "\n")

file_ES = open( PATH + "entanglement/es_t_%.2f_tp_%.2f_U_%.2f_mu_%.2f.txt" % (t,tp,U,mu),"a")
for i in range(0,R):
    file_ES.write("  ".join(map(str, ES[i])) + " " + "\n")
file_EE = open( PATH + "entanglement/ee_t_%.2f_tp_%.2f_U_%.2f_mu_%.2f.txt" % (t,tp,U,mu),"a")
file_EE.write("  ".join(map(str, EE)) + " " + "\n")

file_STAT = open( PATH + "logs/stat_t_%.2f_tp_%.2f_U_%.2f_mu_%.2f.txt" % (t,tp,U,mu),"a")
file_STAT.write("  ".join(map(str,eng.sweep_stats['E'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['S'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['max_trunc_err'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['norm_err'])) + " " + "\n")

with open( PATH + 'mps/gs_t_%.2f_tp_%.2f_U%.2f_mu%.2f.pkl' % (t,tp,U,mu), 'wb') as f:
    pickle.dump(psi, f)



print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
