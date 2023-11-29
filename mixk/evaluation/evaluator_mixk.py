import itertools
import json
import logging
import os
import pickle as pkl
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import Dict, Optional

import torch

from ..utils import distributed_info as comm
from ..utils.registry import Registry, build_from_cfg

METRICS = Registry('metric')
DATASET_CONVERTER = Registry('DatasetConverter')
POST_PROCESSOR = Registry('PostProcessor')


def build(cfg, registry, default_args=None):
    """Build a object.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        obj
    """
    if isinstance(cfg, list):
        objs = [build_from_cfg(_cfg, registry, default_args) for _cfg in cfg]
        return objs
    elif isinstance(cfg, Dict):
        return build_from_cfg(cfg, registry, default_args)
    else:
        raise TypeError


def get_predictions_and_labels(func):

    def wrapper(self, *args, **kwargs):
        if self.distributed is False:
            predictions = self._predictions
            labels = getattr(self, '_labels') if hasattr(self, '_labels') else None
        else:

            def get_all_data(data):
                data_list = comm.gather(data, dst_rank=0)
                all_data = list(itertools.chain(*data_list))
                return all_data

            comm.synchronize()
            predictions = get_all_data(self._predictions)
            labels = get_all_data(self._labels) if hasattr(self, '_labels') else None
            if not comm.is_main_process():
                return {}

        kwargs['predictions'] = predictions
        kwargs['labels'] = labels
        return func(self, *args, **kwargs)

    return wrapper


class PostProcessor(metaclass=ABCMeta):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.distributed = comm.get_world_size() > 1

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def convert(self, batch_data, model_outputs):
        pass

    @abstractmethod
    @get_predictions_and_labels
    def process(self, *args, **kwargs):
        pass

    @staticmethod
    def list_to_tensor(list_data: list) -> torch.tensor:
        # tensor_size = (len(list_data), list_data[0].shape[1])
        if not isinstance(list_data[0], dict):
            if isinstance(list_data[0], list) or isinstance(list_data[0], str):
                tensor_data = list_data
                return tensor_data

            if len(list_data[0].shape) == 0:
                tensor_size = (len(list_data), 1)
            elif len(list_data[0].shape) == 1:
                tensor_size = (len(list_data), list_data[0].shape[0])
            else:
                tensor_size = (len(list_data), list_data[0].shape[1])
            tensor_dtype = list_data[0].dtype
            tensor_data = torch.zeros(size=tensor_size, dtype=tensor_dtype)
            for idx, data in enumerate(list_data):
                tensor_data[idx] = data
        else:
            tensor_data = list_data

        return tensor_data


@POST_PROCESSOR.register_module()
class Evaluator(PostProcessor):

    def __init__(self, metrics, dataset_converters):
        super().__init__()
        self._metrics: list = self.build_metrics(metrics)
        self._dataset_converters: list = self.build_dataset_converters(
            dataset_converters=dataset_converters, default_args={'post_process_type': str(self)})
        self._labels: list = []
        self._predictions: list = []

    def reset(self):
        self._labels = []
        self._predictions = []

    @classmethod
    def build_metrics(cls, metrics):
        metrics = metrics if isinstance(metrics, list) else [metrics]
        return build(metrics, METRICS)

    @classmethod
    def build_dataset_converters(cls, dataset_converters, default_args: Optional[Dict]):
        dataset_converters = dataset_converters if isinstance(dataset_converters, list) else [dataset_converters]
        return build(dataset_converters, DATASET_CONVERTER, default_args)

    def convert(self, batch_data, model_outputs):
        for dataset_obj in self._dataset_converters:
            model_outputs, labels = dataset_obj.convert(batch_data, model_outputs)
            if labels is not None:
                self._labels.extend(labels)
            self._predictions.extend(model_outputs)

    @get_predictions_and_labels
    def process(self, *args, **kwargs):
        predictions, labels = kwargs['predictions'], kwargs['labels']
        if len(labels) != 0:
            labels = self.list_to_tensor(labels)
        predictions = self.list_to_tensor(predictions)

        eval_results = {}
        for metric_obj in self._metrics:
            eval_results[str(metric_obj)] = metric_obj.evaluate(predictions, labels)
        self.print_eval_results(eval_results)
        return eval_results

    def print_eval_results(self, eval_results: Dict) -> None:
        for metric_name, metric_value in eval_results.items():
            self.logger.info('{}:    -->    {}'.format(metric_name, metric_value))

    def __str__(self):
        return 'evaluator'


@POST_PROCESSOR.register_module()
class Submitter(PostProcessor):

    def __init__(self,
                 dataset_converters,
                 *,
                 output_dir: str = None,
                 file_name: str = 'submit_result.json',
                 post_process_type: Dict = None):
        super().__init__()

        self._predictions: list = []
        self._file_name = file_name
        self._output_dir = os.path.abspath('./') if output_dir is None else output_dir
        if not os.path.exists(self._output_dir):
            os.mkdir(self._output_dir)
        post_process_type = {'post_process_type': str(self)} if post_process_type is None else post_process_type
        self._save_file_name = os.path.join(self._output_dir, self._file_name)
        self._dataset_converters: list = Evaluator.build_dataset_converters(
            dataset_converters=dataset_converters, default_args=post_process_type)

    def reset(self):
        self._predictions = []

    def convert(self, batch_data, model_outputs):
        for dataset_obj in self._dataset_converters:
            section_predictions = dataset_obj.convert(batch_data, model_outputs)
            self._predictions.extend(section_predictions)

    @get_predictions_and_labels
    def process(self, *args, **kwargs):
        predictions = kwargs['predictions']
        assert len(predictions) > 0, ValueError('predictions are empty!')
        with open(self._save_file_name, 'w') as f:
            f.write(json.dumps(predictions, indent=2))
            self.logger.info('The submit file has been saved to {}'.format(self._save_file_name))

        return None

    def __str__(self):
        return 'submitter'


@POST_PROCESSOR.register_module()
class Predictor(Submitter):

    def __init__(self, dataset_converters, *, output_dir: str = None, file_name: str = 'predict_result.pkl'):
        super().__init__(
            dataset_converters=dataset_converters,
            output_dir=output_dir,
            file_name=file_name,
            post_process_type={'post_process_type': str(self)})

    @get_predictions_and_labels
    def process(self, *args, **kwargs):
        predictions = kwargs['predictions']
        assert len(predictions) > 0, ValueError('predictions are empty!')
        with open(self._save_file_name, 'wb') as f:
            pkl.dump(predictions, f)
            self.logger.info('The prediction file has been saved to path {}:'.format(self._save_file_name))

        return None

    def __str__(self):
        return 'predictor'


def build_post_processor(cfg):
    return build(cfg, POST_PROCESSOR, default_args=None)


def inference_on_dataset(cfg, model, data_loader):
    post_processor = build_post_processor(cfg.post_processor)
    post_processor.reset()

    logger = logging.getLogger(__name__)
    logger.info('Starting inference on {} batch images'.format(len(data_loader)))
    with to_inference(model), torch.no_grad():
        for idx, batch_data in enumerate(data_loader, start=1):
            logger.info('{} running idx: {}/{}'.format(str(post_processor), idx, len(data_loader)))
            outputs = model(batch_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            post_processor.convert(batch_data=batch_data, model_outputs=outputs)

    results = post_processor.process()
    return results if results is not None else {}


@contextmanager
def to_inference(model):
    old_mode = model.training
    model.eval()
    yield
    model.train(old_mode)
