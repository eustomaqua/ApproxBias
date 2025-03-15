# coding: utf-8


import logging


# ---------------------
# Printer


def _elegant_helper(text, logger=None):
    if not logger:
        print(text)
        return
    logger.info(text)


def elegant_print(text, logger=None):
    if isinstance(text, str):
        _elegant_helper(text, logger)
        return
    elif isinstance(text, pd.core.series.Series):
        _elegant_helper(text, logger)
        return

    if not isinstance(text, list):
        raise ValueError("Wrong text input.")
    for k in text:
        _elegant_helper(k, logger)


# ---------------------
# Logger


def get_elogger(logname, filename, level=logging.DEBUG):
    logger = logging.getLogger(logname)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s %(name)s | %(message)s",
        # fmt="%(asctime)s - %(name)s | %(message)s",
        datefmt="%a %Y/%m/%d %H:%M:%S")

    fileHandler = logging.FileHandler(filename, mode='a')
    logger.setLevel(level)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # return logger, formatter, fileHandler
    del fileHandler, formatter
    return logger


def rm_ehandler(logger, formatter, fileHandler):
    logger.removeHandler(fileHandler)
    del fileHandler, formatter
    del logger
    return


# ---------------------
