import functools
import logging
import os.path as osp
import sys
import atexit

from termcolor import colored

from mixk.utils.third_party_libs import PathManager


class _ColoredMessage(logging.Formatter):

    def formatMessage(self, record):
        log_msg = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            return colored(text='WARNING', color='red', attrs=['blink']) + ' ' + log_msg
        elif record.levelno in [logging.ERROR, logging.CRITICAL]:
            return colored(text='ERROR', color='red', on_color='on_red', attrs=['bold']) + ' ' + log_msg
        else:
            return log_msg


@functools.lru_cache()
def setup_logger(output: str = None, distributed_rank: int = 0, color: bool = True, name: str = 'mixk'):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    plain_formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S')

    if distributed_rank == 0:
        fmt = _ColoredMessage(
            colored('[%(asctime)s %(name)s]: ', 'green') +
            '%(message)s', datefmt='%m/%d %H:%M:%S') if color else plain_formatter
        logger.addHandler(_master_rank_logger(formatter=fmt))

    if output is not None:
        logger.addHandler(_file_logger(output, distributed_rank, plain_formatter))

    return logger


@functools.lru_cache()
def _master_rank_logger(formatter: logging.Formatter) -> logging.StreamHandler:
    logger = logging.StreamHandler(stream=sys.stdout)
    logger.setLevel(logging.DEBUG)
    logger.setFormatter(formatter)
    return logger


@functools.lru_cache()
def _file_logger(output: str, rank: int, formatter: logging.Formatter) -> logging.StreamHandler:
    filename = output if output.endswith('.log') else osp.join(output, 'mixk.log')
    if rank > 0:
        filename = filename + '.rank_{}.log'.format(rank)

    PathManager.mkdirs(osp.dirname(filename))
    logger = logging.StreamHandler(_cached_log_stream(filename))
    logger.setLevel(logging.DEBUG)
    logger.setFormatter(formatter)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    file_obj = PathManager.open(filename, 'w')
    atexit.register(file_obj.close)
    return file_obj
