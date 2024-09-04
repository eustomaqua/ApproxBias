# coding: utf-8
#
# Usage: to measure the bias level within one classifier
#


import numpy as np
import pdb
from hfm.hfm_df import bias_degree_bin, bias_degree_nonbin


def test_bias_level():
  res = bias_degree_bin(0, 0)
  ans = bias_degree_bin(0, 1e-7)
  # pdb.set_trace()
  assert res[0] == 0
  assert ans[0] >= 0

  res = bias_degree_nonbin(0, 0)
  ans_1 = bias_degree_nonbin(0, 1e-7)
  ans_2 = bias_degree_nonbin(0, 1e7)
  assert res[0] == 0
  assert ans_1[0] >= 0
  assert ans_2[0] >= 0
  return
