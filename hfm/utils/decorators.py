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


# def fantasy_timer(func):
#     # @functools.wraps(func)
#     def wrapper(*args, **kw):
#         since = time.time()
#         ans = func(*args, **kw)
#         tim_elapsed = time.time() - since
#         return ans, tim_elapsed
#     return wrapper


def fantasy_timer(func):
    def wrapper(*args, **kw):
        since = time.perf_counter()
        ans = func(*args, **kw)
        tim_elapsed = time.perf_counter() - since
        return ans, tim_elapsed
    return wrapper


# ---------------------
# Timer


def elegant_durat_core(tim_elapsed, verbose=False):
    second = tim_elapsed % 60
    minute = tim_elapsed // 60
    if not verbose and minute == 0:
        return "{:.2f} sec".format(second)
    tim_elapsed = minute
    minute = tim_elapsed % 60
    hours_ = tim_elapsed // 60
    if not verbose and hours_ == 0:
        return "{:.0f} min {:.2f} sec".format(minute, second)
    return "{:.0f} hrs {:.0f} min {:.2f} sec".format(
        hours_, minute, second)


def elegant_durat(tim_elapsed, verbose=True):
    return "{}, i.e. {:.10f} minutes".format(
        elegant_durat_core(
            tim_elapsed, verbose), tim_elapsed / 60)


def elegant_dated(since, fmt='num'):
    if fmt == 'wks':
        formatter = "%a %y/%m/%d %H:%M:%S"
    elif fmt == 'day':
        formatter = "%A, %B %d, %Y, %H:%M:%S (GMT%z)"
    elif fmt == 'txt':
        formatter = "%d-%b-%Y %H:%M:%S"
    elif fmt == 'num':
        formatter = "%Y-%m-%d %H:%M:%S"
    else:
        raise ValueError("Wrong format for output")
    return time.strftime(formatter, time.localtime(since))


def fantasy_durat_minor(tim_elapsed, verbose=False):
    sec = int(tim_elapsed)
    tim_cost = (tim_elapsed - sec) * 1000
    millis = int(tim_elapsed)
    tim_cost = (tim_cost - millis) * 1000
    format_text = "{:d}s {:d}ms".format(sec, millis)
    if not verbose:
        return "{} {:.2f}μs".format(format_text, tim_cost)

    micros = int(tim_cost)
    tim_cost = (tim_cost - micros) * 1000
    nano_s = int(tim_cost)
    pico_s = (tim_cost - nano_s) * 1000
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

    second = tim_elapsed % 60
    minute = tim_elapsed // 60
    format_text = "{:.2f}{}".format(second, unit_sec)
    if (not verbose) and minute < 60:
        return "{:.0f}{} {}".format(minute, unit_min, format_text)

    tim_elapsed = minute
    minute = tim_elapsed % 60
    hours_ = tim_elapsed // 60
    format_text = "{:.0f}{} {}".format(minute, unit_min, format_text)
    if (not verbose) and hours_ < 24:
        return "{:.0f} hr {}".format(hours_, format_text)

    tim_elapsed = hours_
    hours_ = tim_elapsed % 24
    days__ = tim_elapsed // 24
    format_text = "{:.0f} hr {}".format(hours_, format_text)
    if (not verbose) and days__ < 30:
        return "{:.0f} d {}".format(days__, format_text)

    tim_elapsed = days__  # mo/mos/mth
    days__ = tim_elapsed % 30
    month_ = tim_elapsed // 30
    format_text = "{:.0f} d {}".format(days__, format_text)
    return "{:.0f} mo {}".format(month_, format_text)


def fantasy_durat(tim_elapsed, verbose=True, abbreviation=False):
    tim_cost = int(tim_elapsed)
    if tim_cost == 0:
        return fantasy_durat_minor(tim_elapsed, verbose)

    return "{}\n\t i.e.\t {}\n\t i.e.\t {:.8f} minutes".format(
        fantasy_durat_major(tim_elapsed, verbose, abbreviation),
        elegant_durat_core(tim_elapsed, verbose),
        tim_elapsed / 60)


# ---------------------
#
