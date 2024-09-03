# coding: utf-8

from utils.verifiers import CONST_ZERO, CONST_DIFF


def test_helper():
  from utils.verifiers import check_zero
  assert check_zero(0) == CONST_ZERO
  assert check_zero(4) == 4

  from utils.verifiers import check_sign
  assert check_sign(7) == 7
  assert check_sign(-1) == -1
  assert check_sign(1e-16) == CONST_ZERO

  from utils.verifiers import check_equal
  assert check_equal(0, CONST_DIFF / 2)
  assert check_equal(1e-8, 1e-9)
  assert not check_equal(1e-5, 1e-6)
