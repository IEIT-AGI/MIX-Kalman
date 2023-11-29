import os.path as osp
from typing import Any

from torch.nn import Module

import mixk.utils.distributed_info as comm
from mixk.utils.checkpoint import Checkpointer


class mixkCheckpointer(Checkpointer):

    def __int__(self,
                model: Module,
                save_dir: str = '',
                *,
                is_save_disk=None,
                is_record_ck: bool = True,
                **other_train_info: object):
        is_master_process = comm.is_main_process()
        if is_save_disk is None:
            is_save_disk = is_master_process
        super().__init__(
            model=model, save_dir=save_dir, is_save_disk=is_save_disk, is_record_ck=is_record_ck, **other_train_info)

    def _load_file_from_path(file_path: str):
        """
        Support multiple formats to load,such as pkl,currently only supports pth
        :return:
        """
        file_format = osp.splitext(file_path)[-1]
        if file_format != '.pth':
            raise Exception('the format of file_path:{} is {},currently only supports pth'.format(
                file_path, file_format))
        checkpoint = super()._load_file_from_path(file_path=file_path)  # load native checkpoint
        if 'model' not in checkpoint:
            checkpoint = {'model': checkpoint}
        return checkpoint

    def _load_model(self, checkpoint: Any):
        """Enhance the  compatibility of the loaded model by ignoring the
        missing key message in checkpoint.

        :param checkpoint:
        :return:
        """
        if checkpoint.get('matching_heuristics', False):
            self._state_dict_to_tensor(checkpoint['model'])
            checkpoint['model'] = self.model.state_dict()

        incompatible_keys = super()._load_model(checkpoint)
        if incompatible_keys is None:
            return None
        else:
            return incompatible_keys
