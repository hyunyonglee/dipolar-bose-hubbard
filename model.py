# Copyright 2022 Hyun-Yong Lee

import numpy as np
from tenpy.models.lattice import Site, Chain
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import Config
from tenpy.networks.site import BosonSite  # if you want to use the predefined site
import matplotlib.pyplot as plt
__all__ = ['DIPOLAR_BOSE_HUBBARD']


class DIPOLAR_BOSE_HUBBARD(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "DIPOLAR_BOSE_HUBBARD")
        L = model_params.get('L', 1)
        t = model_params.get('t', 1.)
        tp = model_params.get('tp', 1.)
        U = model_params.get('U', 1.)
        mu = model_params.get('mu', 0.)
        Ncut = model_params.get('Ncut', 2)
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        bc = model_params.get('bc', 'periodic')
        QN = model_params.get('QN', 'N')

        site = BosonSite( Nmax=Ncut, conserve=QN, filling=0.0 )
        site.multiply_operators(['B','B'])
        site.multiply_operators(['Bd','Bd'])

        lat = Chain( L=L, site=site, bc=bc, bc_MPS=bc_MPS)
        
        CouplingModel.__init__(self, lat)

        # 3-site hopping
        self.add_multi_coupling( -t, [('Bd', 0, 0), ('B B', 1, 0), ('Bd', 2, 0)])
        self.add_multi_coupling( -t, [('B', 0, 0), ('Bd Bd', 1, 0), ('B', 2, 0)])

        # 4-site hopping
        self.add_multi_coupling( -tp, [('Bd', 0, 0), ('B', 1, 0), ('B', 2, 0), ('Bd', 3, 0)])
        self.add_multi_coupling( -tp, [('B', 0, 0), ('Bd', 1, 0), ('Bd', 2, 0), ('B', 3, 0)])

        # hubbard
        self.add_onsite( U/2., 0, 'NN')

        # chemical potential
        self.add_onsite( -( mu + U/2. ), 0, 'N')

        
        MPOModel.__init__(self, lat, self.calc_H_MPO())

        