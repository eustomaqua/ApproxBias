# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority vote
#   Measuring fairness via data manifolds (w. extension)
#


import os
import pandas as pd
from hfm.utils.verifiers import DTY_FLT
from experiment.datasets import DATASETS, DATASET_NAMES, PACKAGE_DIR

RAW_EXPT_DIR = os.path.join(PACKAGE_DIR, 'findings')
DAT_EXPT_NMS = ['ricci', 'german', 'adult', 'ppr', 'ppvr', 'tmp']
DAT_EXPT_ORG = ['Ricci', 'Credit', 'Income', 'PPR', 'PPVR',
                'Simulation']


# plt.rcParams['font.family'] = 'Times New Roman'
# DTY_PLT = '.pdf'
# DTY_FLT = 'float'
# PLT_SHOW_OFF = False
# PLT_SPINEOFF = True
# PLT_LOCATION = 'best'
# PLT_FRAMEBOX = True
# PLT_T_LAYOUT = True
# PLT_AX_STYLE = True


class DataSetup:
    def __init__(self, data_type):
        self._data_type = data_type
        self._log_document = data_type

        if data_type == 'ppr':
            self._data_type = DATASET_NAMES[-2]
        elif data_type == 'ppvr':
            self._data_type = DATASET_NAMES[-1]
        elif data_type not in ['ricci', 'german', 'adult']:
            raise ValueError("Wrong dataset `{}`".format(data_type))

        # ['ricci', 'german', 'adult', 'ppr', 'ppvr']
        idx = DATASET_NAMES.index(self._data_type)
        self._dataset = DATASETS[idx]
        self._data_frame = self._dataset.load_raw_dataset()

        if data_type == "ricci":
            self.saIndex = [2]     # 'Race' -2
        elif data_type == "german":
            self.saIndex = [3, 5]  # ['sex', 'age'] [,12]
        elif data_type == "adult":
            self.saIndex = [2, 3]  # ['race','sex'] [7,8]
        elif data_type == "ppr":
            self.saIndex = [0, 2]  # ['sex','race'] [0,3]
        elif data_type == "ppvr":
            self.saIndex = [0, 2]  # ['sex','race'] [0,3]

        self.saValue = self._dataset.get_privileged_group(
            'numerical-binsensitive')
        self.saValue = [0 for sa in self.saValue if sa == 1]

        # @property
        # def data_type(self):
        #     return self._data_type

        @property
        def log_document(self):
            return self._log_document

        # # ----------- mu -----------
        # def prepare_mu_datasets(self, ratio=.5, logger=None):
        #   pass
        # # ----------- tr -----------
        # # ----------- bi -----------
        # def prepare_bi_datasets(self, ratio=.5, logger=None):
        #   pass

        @property
        def dataset(self):
            return self._dataset

        @property
        def data_frame(self):
            return self._data_frame

        @property
        def trial_type(self):
            return self._trial_type


class GraphSetupVer1:
    def __init__(self, gen=False, rep=False, m1=30, m2=10,
                 figname=''):
        # self._nb_iter = nb_iter
        self._gen = gen  # gen_iter
        self._rep = rep  # rep_iter
        self._m1, self._m2 = m1, m2
        self._figname = figname
        self._cmap_name = 'muted'  # 'bright'

    @property
    def figname(self):
        return self._figname

    @property
    def nb_iter(self):
        return self._nb_iter

    @property
    def gen_iter(self):
        return self._gen

    @property
    def rep_iter(self):
        return self._rep

    def schedule_mspaint(self, raw_dframe, tag_col):
        raise NotImplementedError

    # def subdraw_spliting(self):
    #     raise NotImplementedError
    #
    # def subdraw_asawhole(self):
    #     raise NotImplementedError

    def prepare_graph(self):
        raise NotImplementedError

    # PREPARE CSV

    def get_raw_filename(self):
        raise NotImplementedError

    def load_raw_dataset(self, filename, sheetname):
        filepath = os.path.join(
            RAW_EXPT_DIR, 'ver1', filename + '.xlsx')
        dframe = pd.read_excel(filepath, sheetname)
        return dframe

    def recap_sub_data(self, dframe, nb_row=3):
        each = 4 * self._nb_iter + 1
        nb_set = len(dframe) - nb_row + 1
        nb_set = (nb_set + 3 * self._nb_iter) // each
        id_set = [0] + [(
            i * each + self._nb_iter + 1) for i in range(nb_set)]
        id_set = [i + nb_row - 1 for i in id_set]
        return nb_set, id_set  # id_set[-1] for bounding

    def fetch_sub_data(self, df, ind_row, tag_col, ats='ut'):
        data = df[tag_col].iloc[ind_row]
        if ats == 'ua':
            data *= 100.
        return data.values.astype(DTY_FLT)

    # PREPARE JSON
    # TBD

    def draw_sub2_jt(self, joint='and|or'):
        if joint == 'and':
            tmp, suffix = [0, 1, 2, ], 'a'
        elif joint == 'or':
            tmp, suffix = [0, 1, 3, ], 'o'
        elif joint == 'none':
            tmp, suffix = [0, 1, ], 'n'
        else:
            tmp, suffix = [0, 1, 2, 3], 'ao'
        return tmp, suffix


class GraphSetupVer2(GraphSetupVer1):
    def __init__(self, gen=False, rep=False, m1=25, m2=11,
                 n_e=2, figname=''):
        self._gen = gen  # gen_iter
        self._rep = rep  # rep_iter /cvs(plit)
        self._m1, self._m2, self._n_e = m1, m2, n_e
        self._figname = figname
        self._cmap_name = 'muted'
        self._nb_iter = 5  # 1

    def load_raw_dataset(self, filename, sheetname):
        filepath = os.path.join(
            RAW_EXPT_DIR, 'ver2', filename + '.xlsx')
        dframe = pd.read_excel(filepath, sheetname)
        return dframe

    def recap_sub_data(self, dframe, nb_row=4,
                       nc_norm=3, nc_sens=4):
        each_att = nc_sens * self._nb_iter
        each_gen = nc_norm * self._nb_iter
        each_set = each_att * 2 + each_gen + 1
        # each_set = each_att * 2 + nc_norm * self._nb_iter + 1
        nb_set = len(dframe) - nb_row + 1
        nb_set = (nb_set + each_att) // each_set
        # or: nb_set= (nb_set - (each_set-each_att)) //each_set +1

        tmp_diff = each_set - each_att
        id_set = [i * each_set + tmp_diff for i in range(nb_set)]
        id_set = [i + nb_row - 1 for i in [0] + id_set]
        del tmp_diff

        return nb_set, id_set, each_set, each_att, each_gen
