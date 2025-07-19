# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds
#       classifier-related, or ensemble
#


# sklearn
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import neural_network

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier,
    # HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier)
import numpy as np

from experiment.utils.data_classify import EnsembleAlgorithm
# from experiment.ensemble import EnsembleAlgorithm
# from hfm.pkgs.AdaFair import AdaFair
# from experiment.utils.pkgs_AdaFair(_mod) import AdaFair
from lightgbm import LGBMClassifier
from fairgbm import FairGBMClassifier
import sklearn
skl_ver = sklearn.__version__
if skl_ver.startswith('1.3.0'):
    from experiment.utils.pkgs_AdaFair_py36 import AdaFair
elif skl_ver.startswith('1.5.1'):
    pass
del skl_ver


# =====================================
# degree/core
# =====================================


# utils_remark.py
# -------------------------------------

AVAILABLE_ABBR_ENSEM = ['Bagging', 'AdaBoostM1', 'SAMME']
AVAILABLE_ABBR_CLS = [
    'DT', 'NB', 'SVM', 'linSVM', 'MLP',
    'LR1', 'LR2', 'LM1', 'LM2', 'kNNu', 'kNNd', 
]   # ALG_NAMES    # 'lmSGD','LR'

# ENSEM_NAMES = [
#     "bagging", "adaboost", "rforest", "extrats", "gradbst",
# ]


FAIR_INDIVIDUALS = {
    'DT': tree.DecisionTreeClassifier(),
    'NB': naive_bayes.GaussianNB(),
    'LR': linear_model.LogisticRegression(max_iter=500),
    'LR1': linear_model.LogisticRegression(
        penalty='none', max_iter=500),
    'LR2': linear_model.LogisticRegression(
        penalty='l2', max_iter=500),  # default

    'SVM': svm.SVC(),
    'linSVM': svm.LinearSVC(max_iter=5000),
    'kNNu': neighbors.KNeighborsClassifier(
        weights='uniform'),  # default
    'kNNd': neighbors.KNeighborsClassifier(
        weights='distance'),

    'MLP': neural_network.MLPClassifier(max_iter=1000),
    # 'NN': neural_network.MLPClassifier(
    #     solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2)),
    'lmSGD': linear_model.SGDClassifier(),
    'LM1': linear_model.SGDClassifier(penalty='l1'),
    'LM2': linear_model.SGDClassifier(penalty='l2'),  # default
}

INDIVIDUALS = FAIR_INDIVIDUALS
del FAIR_INDIVIDUALS


TREE_ENSEMBLES = {
    'RF': RandomForestClassifier,
    'ET': ExtraTreesClassifier,
    'GradBoost': GradientBoostingClassifier,
}

HOMO_ENSEMBLES = {
    'Bagging': BaggingClassifier,
    'AdaBoost': AdaBoostClassifier,
}  # homogeneous

HETERO_ENSEMBLES = {
    'VotingC': VotingClassifier,
    'StackingC': StackingClassifier,
}  # heterogeneous

ENS_NAMES = [
    'RF', 'ET', 'Bagging', 'AdaBoost',
    'GradBoost',  # 'GradientBoost',
    'VotingC', 'StackingC',
]

CONCISE_INDIVIDUALS = {
    'DT': tree.DecisionTreeClassifier(),
    'NB': naive_bayes.GaussianNB(),
    'LR': linear_model.LogisticRegression(),
    'SVM': svm.SVC(),
    'linSVM': svm.LinearSVC(),
    'kNNu': neighbors.KNeighborsClassifier(weights='uniform'),  # default
    'kNNd': neighbors.KNeighborsClassifier(weights='distance'),
    'MLP': neural_network.MLPClassifier(),
    'lmSGD': linear_model.SGDClassifier(),
}


# ===============================
# Ensembles
#   works for [binary & multiclass]


# -------------------------------
# Training
#   prgm/nucleus/data_classify.py


def achieve_ensemble_from_train_set(name_ens, abbr_cls, nb_cls,
                                    X_trn, y_trn, X_val, X_tst):
    """
    X/y_trn/val/tst: list, np.ndarray, pd.DataFrame? 
    """
    name_cls = INDIVIDUALS[abbr_cls]
    coef, clfs, indices = EnsembleAlgorithm(name_ens, name_cls, nb_cls,
                                            X_trn, y_trn)

    y_insp = [j.predict(X_trn).tolist() for j in clfs]  # inspect
    y_pred = [j.predict(X_tst).tolist() for j in clfs]  # predict
    y_cast = [j.predict(X_val).tolist() for j in clfs] if X_val else []

    # return y_insp, y_cast, y_pred, coef, clfs, indices
    return coef, clfs, indices, y_insp, y_cast, y_pred


# -------------------------------
# Voting
#   prgm/nucleus/ensem_voting.py


def plurality_voting(y, yt):
    vY = np.unique(np.concatenate([[y], yt]))

    vote = [np.sum(
        np.equal(yt, i), axis=0).tolist() for i in vY]
    loca = np.argmax(vote, axis=0)  # vote.argmax(axis=0)
    fens = [vY[i] for i in loca.tolist()]
    return fens


def majority_voting(y, yt):
    vY = np.unique(np.concatenate([[y], yt]))

    vote = [np.sum(
        np.equal(yt, i), axis=0).tolist() for i in vY]
    nb_cls = len(yt)
    half = int(np.ceil(nb_cls / 2.))
    vts = np.array(vote).T  # transpose

    loca = [np.where(j >= half)[0] for j in vts]
    loca = [j[0] if len(j) > 0 else -1 for j in loca]
    fens = [vY[i] if i != -1 else -1 for i in loca]
    return fens


def weighted_voting(y, yt, wgt):
    vY = np.unique(np.concatenate([[y], yt]))

    coef = np.array([wgt]).transpose()
    weig = [np.sum(
        coef * np.equal(yt, i), axis=0).tolist() for i in vY]
    loca = np.array(weig).argmax(axis=0).tolist()
    fens = [vY[i] for i in loca]
    return fens


# -------------------------------
# Pruning


# -------------------------------
# Experiments: Classifier(s)


class ClsfSetup:
    def __init__(self, abbr_cls):
        self._abbr_cls = abbr_cls

    @property
    def abbr_cls(self):
        return self._abbr_cls

    def schedule_content(self):
        raise NotImplementedError

    def prepare_trial(self):
        raise NotImplementedError


class IndividualClsf(ClsfSetup):
    def __init__(self, abbr_cls):
        # self._abbr_cls = abbr_cls
        super().__init__(abbr_cls)
        self._member = INDIVIDUALS[abbr_cls]

    @property
    def abbr_cls(self):
        return self._abbr_cls

    @property
    def member(self):
        return self._member

    @member.setter
    def member(self, value):
        self._member = value


class EnsembleClsf(ClsfSetup):
    def __init__(self, name_ens, abbr_cls, nb_cls, nb_pru=None):
        super().__init__(abbr_cls)
        self._name_ens = name_ens
        self._nb_cls = nb_cls
        self._nb_pru = nb_pru
        if nb_pru is None:
            self._nb_pru = nb_cls
        self._weight = list()

    @property
    def name_ens(self):
        return self._name_ens

    @property
    def nb_cls(self):
        return self._nb_cls

    @property
    def nb_pru(self):
        return self._nb_pru

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value=None):
        self._weight = value


# -------------------------------
#


# -------------------------------
# Experiments: Classifier(s)


# class FairRelativeClsf(EnsembleClsf):
#   def __init__(self, abbr_cls, name_ens=None, nb_cls=0, nb_pru=None):
#     pass


AVAILABLE_CLFS = list(INDIVIDUALS.keys())
# AVAIL_CLFS = list(INDIVIDUALS.keys())
AVAILABLE_ENSF = [
    'bagging', 'AdaBoost',  # 'Bagging@SK', 'AdaBoost@SK',
    'LightGBM', 'FairGBM', 'AdaFair',
    # 'lightGBM', 'fairGBM', 'AdaFair',  # LightGBM, FairGBM
]  # fair ensemble


# class FairRelativeClsf(IndividualClsf):
class RelativeFairClsf(IndividualClsf):
    def __init__(self, abbr_cls, nb_cls=3,
                 constraint_type='FPR,FNR',
                 saIndex=list(), saValue=list()):
        self._abbr_cls = abbr_cls

        '''
    if abbr_cls in AVAILABLE_CLFS:
      self._member = INDIVIDUALS[abbr_cls]
    else:
      self._member = self.prepare_fair_relative(
          # abbr_cls, nb_cls, constraint_type, saIndex, saValues)
          abbr_cls, nb_cls, constraint_type, saIndex, saValue)
    '''
        self._initial_pm = {  # 'abbr_cls': abbr_cls,
            'nb_cls': nb_cls,
            'constraint_type': constraint_type,
            'saIndex': saIndex,
            'saValue': saValue}
        self.initialize_clf(abbr_cls)  # **self._initial_pm)
        # self.initialize_clf(abbr_cls, nb_cls, constraint_type, saIndex, saValue)

        if abbr_cls in AVAILABLE_ENSF:
            self._abbr_cls += '_cls{}'.format(nb_cls)
        if abbr_cls == 'FairGBM':
            self._abbr_cls += '_{}'.format(constraint_type)

    # def initialize_clf(self, abbr_cls, nb_cls, constraint_type,
    #                    saIndex, saValue):
    #   if abbr_cls in AVAILABLE_CLFS:
    #     self._member = INDIVIDUALS[abbr_cls]
    #   else:
    #     self._member = self.prepare_fair_relative(
    #         # abbr_cls, nb_cls, constraint_type, saIndex, saValues)
    #         abbr_cls, nb_cls, constraint_type, saIndex, saValue)
    #   return
    def initialize_clf(self, abbr_cls):
        if abbr_cls in AVAILABLE_CLFS:
            self._member = INDIVIDUALS[abbr_cls]
        else:
            self._member = self.prepare_fair_relative(
                # abbr_cls, nb_cls, constraint_type, saIndex, saValue)
                abbr_cls, **self._initial_pm)
        return

    def prepare_fair_relative(self, abbr_cls, nb_cls=3,
                              constraint_type='FPR,FNR',
                              saIndex=list(), saValue=list()):
        if abbr_cls in ['Bagging@SK', 'bagging',
                        'Bagging']:
            clf = BaggingClassifier(n_estimators=nb_cls)
        elif abbr_cls in ['AdaBoost@SK', 'AdaBoost',
                          'adaboost']:
            clf = AdaBoostClassifier(n_estimators=nb_cls)
        elif abbr_cls in ['lightGBM', 'LightGBM']:
            # clf = lightgbm.LGBMClassifier(n_estimators=nb_cls)
            clf = LGBMClassifier(n_estimators=nb_cls)
        elif abbr_cls in ['fairGBM', 'FairGBM']:
            clf = FairGBMClassifier(n_estimators=nb_cls,
                                    constraint_type=constraint_type)
        elif abbr_cls == 'AdaFair':
            clf = AdaFair(n_estimators=nb_cls,
                          # saIndex=sa_idx, saValues=sa_val)
                          saIndex=saIndex, saValue=saValue)
        return clf


# -------------------------------
#
