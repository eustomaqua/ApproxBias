# coding: utf-8


import time
import functools


# ---------------------
# Timer


def elegant_timer(text):
  def decorator(func):
    def wrapper(*args, **kw):
      since = time.time()
      ans = func(*args, **kw)
      time_elapsed = time.time() - since
      print("Time Cost `{}`: {:.8f} sec".format(
          text, time_elapsed))
      return ans
    return wrapper
  return decorator


def fantasy_timer(func):
  def wrapper(*args, **kw):
    since = time.time()
    ans = func(*args, **kw)
    time_elapsed = time.time() - since
    return ans, time_elapsed
  return wrapper


# ---------------------
# Timer


# ---------------------
#
