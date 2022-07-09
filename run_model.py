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
U = float(sys.argv[3])
mu = float(sys.argv[4])
Ncut = int(sys.argv[5])
CHI = int(sys.argv[6])
RM = sys.argv[7]
PATH = sys.argv[8]
BC_MPS = "infinite"
BC = "periodic" # "open"

model_params = {
    "L": L,
    "t": t,
    "U": U,
    "mu": mu,
    "Ncut": Ncut,
    "bc_MPS": BC_MPS,
    "bc": BC
}

print("\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
M = model.DIPOLAR_BOSE_HUBBARD(model_params)

product_state = ["1"] * M.lat.N_sites
psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

if RM == 'random':
    TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': 20}, 'verbose': 0}
    eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
    eng.run()
    psi.canonical_form() 

dchi = int(CHI/2)
chi_list = {}
for i in range(2):
    chi_list[i*10] = (i+1)*dchi

dmrg_params = {
    'mixer': True,  # setting this to True helps to escape local minima
    'mixer_params': {
        'amplitude': 1.e-3,
        'decay': 1.2,
        'disable_after': 50
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
    'max_sweeps': 200
}


eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.

N = psi.expectation_value("N")
EE = psi.entanglement_entropy()
ES = psi.entanglement_spectrum()

ensure_dir(PATH + "observables/")
ensure_dir(PATH + "entanglement/")
ensure_dir(PATH + "logs/")
ensure_dir(PATH + "mps/")

file1 = open( PATH + "observables/energy.txt","a")
file1.write(repr(t) + " " + repr(U) + " " + repr(mu) + " " + repr(E) + " " + repr(psi.correlation_length()) + " " + "\n")

file2 = open( PATH + "observables/numbers.txt","a")
file2.write(repr(t) + " " + repr(U) + " " + repr(mu) + " " + "  ".join(map(str, N)) + " " + "\n")

file_ES = open( PATH + "entanglement/es_t_%.1f_U_%.1f_mu_%.1f.txt" % (t,U,mu),"a")
for i in range(0,L):
    file_ES.write("  ".join(map(str, ES[i])) + " " + "\n")
file_EE = open( PATH + "entanglement/ee_t_%.1f_U_%.1f_mu_%.1f.txt" % (t,U,mu),"a")
file_EE.write("  ".join(map(str, EE)) + " " + "\n")

file_STAT = open( PATH + "logs/stat_t_%.1f_U_%.1f_mu_%.1f.txt" % (t,U,mu),"a")
file_STAT.write("  ".join(map(str,eng.sweep_stats['E'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['S'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['max_trunc_err'])) + " " + "\n")
file_STAT.write("  ".join(map(str,eng.sweep_stats['norm_err'])) + " " + "\n")

with open( PATH + 'mps/psi_t_%.1f_U%.1f_mu%.1f.pkl' % (t,U,mu), 'wb') as f:
    pickle.dump(psi, f)


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
