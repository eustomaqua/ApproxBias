# coding: utf-8


def test_case():
    from experiment.df_zip.mcvg_sim import ManfCvgPrime, ManfCvgEmpir
    logger = None
    kw = dict(prep='min_max', nb_cv=1)  # 2)
    # kw['prep'] = 'none'
    trial_type = 'mCV_may12_cvg1c'
    data_type = 'ricci'

    # cs = ManfCvgEmpir(trial_type, data_type, **kw)
    cs = ManfCvgPrime(trial_type, data_type, **kw)
    # cs.preparing_curr_dat(logger)
    cs.coding_per_dataset(logger)
    # pdb.set_trace()
    return
