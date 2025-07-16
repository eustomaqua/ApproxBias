# coding: utf-8


import time
# import functools


# ---------------------
# Timer


def elegant_timer(text):
    def decorator(func):
        def wrapper(*args, **kw):
            since = time.time()
            ans = func(*args, **kw)
            tim_elapsed = time.time() - since
            print("Time Cost `{}`: {:.8f} sec".format(
                text, tim_elapsed))
            return ans
        return wrapper
    return decorator


def fantasy_timer(func):
    def wrapper(*args, **kw):
        since = time.time()
        ans = func(*args, **kw)
        tim_elapsed = time.time() - since
        return ans, tim_elapsed
    return wrapper


# ---------------------
# Timer


def fantasy_durat_minor(tim_elapsed, verbose=False):
    sec = int(tim_elapsed)
    time_cost = (tim_elapsed - sec) * 1000
    millis = int(tim_elapsed)
    time_cost = (time_cost - millis) * 1000
    format_text = "{:d}s {:d}ms".format(sec, millis)
    if not verbose:
        return "{} {:.2f}μs".format(format_text, time_cost)

    micros = int(time_cost)
    time_cost = (time_cost - micros) * 1000
    nano_s = int(time_cost)
    pico_s = (time_cost - nano_s) * 1000
    format_text = "{} {:d}μs {:d}ns {:.2f}ps".format(
        format_text, micros, nano_s, pico_s)
    return format_text


def fantasy_durat_major(tim_elapsed, verbose=False,
                        abbreviation=True):
    unit_sec = "''" if abbreviation else " sec"
    unit_min = "'" if abbreviation else " min"
    format_text = "{:.2f}{}".format(tim_elapsed, unit_sec)
    if (not verbose) and tim_elapsed < 60:
        return format_text


def fantasy_durat(tim_elapsed):
    time_cost = int(tim_elapsed)
    pass


# ---------------------
#
