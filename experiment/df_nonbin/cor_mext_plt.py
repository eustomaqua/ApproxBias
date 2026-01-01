# coding: utf-8

import pdb
import numpy as np
import pandas as pd

from experiment.utils_empirical import GraphSetupVer2

from hfm.utils.verifiers import unique_column, DTY_FLT
from hfm.utils.recorders import BLFAIR
from hfm.hfm_df import differentiate_tim, differentiate_val


# ==============================
# Division


# class GraphSetup(GraphSetupVer2):
#     def sub_dat_multivar(self, dframe, nb_set, id_set, tag):
#         pass
