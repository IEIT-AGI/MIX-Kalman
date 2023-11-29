from typing import Dict, Optional
from mixk.utils.config import mixkEasyDict
import re


class SamplerAdaptor:
    """modify the parameters of the sampler and the batch_sampler to adapt to
    different sampling methods."""

    @classmethod
    def adaptor(cls, cfg: mixkEasyDict, default_args: Optional[Dict] = None):
        sampler_type = cfg.get('type')
        func_name = '_adaptor_' + cls.run_fun_suffix(sampler_type)
        run_func = getattr(cls, func_name, None)
        if run_func is None:
            return
        else:
            run_func(cfg, default_args)

    @classmethod
    def run_fun_suffix(cls, name: str) -> str:
        r = re.findall('([A-Z][a-z]+)', name)
        func_suffix = '_'.join(r)

        return func_suffix.lower()

    @classmethod
    def _adaptor_token_bucket_sampler(cls, cfg: mixkEasyDict, default_args: Optional[Dict] = None):
        dataset = default_args.pop('dataset')
        default_args['lens'] = dataset.dataset.lens

    @classmethod
    def _adaptor_random_sampler(cls, cfg: mixkEasyDict, default_args: Optional[Dict] = None):
        default_args['data_source'] = default_args.pop('dataset')

    @classmethod
    def _adaptor_distributed_sampler(cls, cfg: mixkEasyDict, default_args: Optional[Dict] = None):
        if 'world_size' in default_args.keys():
            default_args['num_replicas'] = default_args.pop('world_size')

    @classmethod
    def _adaptor_sequential_sampler(cls, cfg: mixkEasyDict, default_args: Optional[Dict] = None):
        if 'dataset' in default_args.keys():
            default_args['data_source'] = default_args.pop('dataset')
