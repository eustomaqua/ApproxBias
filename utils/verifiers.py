# coding: utf-8


# ---------------------
# Constants


CONST_ZERO = 1e-13
CONST_DIFF = 1e-7


# ---------------------
# Helper functions


def check_belong(tmp, *args):
  for v in args:
    if isinstance(tmp, v):
      return True
  return False


def check_zero(tmp, diff=CONST_ZERO):
  if not check_belong(tmp, list, tuple):
    return tmp if tmp != 0. else diff
  tmp = [check_zero(i, diff) for i in tmp]
  return all(tmp)


def check_sign(x, diff=CONST_ZERO):
  if abs(x) > CONST_ZERO:
    return x
  elif x > 0:
    return CONST_ZERO
  elif x < 0:
    return -CONST_ZERO
  return 0.0


def check_equal(tmp_a, tmp_b, diff=CONST_DIFF):
  flag_a = check_belong(tmp_a, list, tuple)
  flag_b = check_belong(tmp_b, list, tuple)
  if not (flag_a or flag_b):
    return True if abs(tmp_a - tmp_b) < diff else False

  if flag_a and check_belong(tmp_b, int, float):
    tmp = [abs(i - tmp_b) < diff for i in tmp_a]
  elif flag_b and check_belong(tmp_a, int, float):
    tmp = [abs(tmp_a - i) < diff for i in tmp_b]
  elif flag_a and flag_b:
    tmp = [abs(i - j) < diff for i, j in zip(tmp_a, tmp_b)]
  return all(tmp)


# ---------------------
#
