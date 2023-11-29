import mixk.utils.distributed_info as dist_info
import torch
import os
import random
from datetime import datetime
from mixk.utils.third_party_libs import PathManager

from mixk.utils.logger import setup_logger
from mixk.utils.collect_running_env import collect_env_info
import argparse
import json
from mixk.utils.config import set_mixk_work_dir, seed_all_rng
import pprint
import sys

random.seed(datetime.now())


def default_argument_parser(epilog=None):
    if epilog is None:
        epilog = f"""
        mixk framework running example:
        1.Run on single machine:
            a. Training on a multiple GPUs
            ${sys.argv[0]} --gpus 8 --config-file cfg.py --load-from /path/weight.pth
            ${sys.argv[0]} --gpus 8 --config-file cfg.py --resume-from /path/weight.pth

            b. Training on a single GPU
            ${sys.argv[0]} --gpus 1 --config-file cfg.py --load-from /path/weight.pth
            or
            ${sys.argv[0]}  --config-file cfg.py --load-from /path/weight.pth

            c. testing on a single GPU or multiple GPUS
            ${sys.argv[0]} --gpus 1 --config-file cfg.py --load-from /path/weight.pth  --eval-only
            ${sys.argv[0]} --gpus 4 --config-file cfg.py --load-from /path/weight.pth  --eval-only

        2.Run on multiple machines:
        (machine0)$ {sys.argv[
            0]} --gpus 8 --node-rank 0 --machines 2 --master-addr 'tcp://127.0.0.1' --master-port  8889 [--other-flags]
        (machine1)$ {sys.argv[
            0]} --gpus 8 --node-rank 1 --machines 2 --master-addr 'tcp://127.0.0.1' --master-port  8889 [--other-flags]
        """  # noqa
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--config-file', metavar='FILE', help='train config file path')
    parser.add_argument('--resume-from', default=None, help='resume from the checkpoint file')
    parser.add_argument('--load-from', default=None, help='load from the checkpoint file')
    parser.add_argument('--eval-only', action='store_true', help='just run evaluation')
    parser.add_argument('--build-submit', action='store_true', help='generate submission results')
    parser.add_argument('--gpus', type=int, default=1, help='the number of gpus on each machine ')
    parser.add_argument('--machines', type=int, default=1, help='the total number of machine to use')
    parser.add_argument('--node-rank', type=int, default=0, help='the rank of current node(unique per machine)')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--master-port',
        default=2**14 + hash(random.randint(0, 2**14)),
        type=int,
        help='it is the free port of mast node(rank 0) and is used for communication in distributed training')
    parser.add_argument(
        '--master-addr', default='tcp://127.0.0.1', type=str, help='the IP address of mast node(rank 0)')

    return parser


def default_setup(args, cfg):
    output_dir = cfg.work_dir
    set_mixk_work_dir(output_dir)
    if output_dir and dist_info.is_main_process():
        PathManager.mkdirs(output_dir)

    rank = dist_info.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank, name='mixk')
    logger.info('Current environment information : \n{}'.format(collect_env_info()))
    logger.info('Command line args: \n {}'.format(args))
    if hasattr(args, 'config_file') and args.config_file != '':
        logger.info('{} file content:\n{}'.format(args.config_file, PathManager.open(args.config_file, 'r').read()))

    logger.info('full config file content: ')
    pprint.pprint({k: v for k, v in cfg.items()})

    if dist_info.is_main_process() and output_dir:
        cfg_path = os.path.join(output_dir, 'config.json')
        with open(cfg_path, 'w') as f:
            f.write(json.dumps({k: v for k, v in cfg.items()}, indent=4, separators=(',', ':')))
            logger.info('full config file saved to {}'.format(cfg_path))

    seed = getattr(cfg, 'seed', None)
    seed_all_rng(seed=None if seed is None else seed + rank)

    if not (hasattr(cfg, 'eval_only') and getattr(cfg, 'eval_only', False)):
        torch.backends.cudnn.benchmark = getattr(cfg, 'CUDNN_BENCHMARK', False)
