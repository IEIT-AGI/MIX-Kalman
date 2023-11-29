import json
import os
from mixk.utils.third_party_libs import PathManager
from ..builder import HOOKS
from .log_buffer_mixk import LogBufferWriter
import logging


@HOOKS.register_module()
class JSONLoggerHook(LogBufferWriter):
    """Write training status to a json file.

    It saves training status as one json per line  for easy parsing.

    Examples parsing such a json file:

    .. code-block:: none

        $ cat training_status.json | jq -s '.[0:2]'
        [
            {
                "data_time": 7.34811183065176e-05,
                "epoch": 0,
                "inner-iteration": 4,
                "iter_time": 0.27459389297291636,
                "lr": 1.8885741265344666e-08,
                "total_loss": 1.9011034965515137,
                "vilbert_mutil_loss": 1.9011034965515137
            }
        ,
            {
                "data_time": 6.791087798774242e-05,
                "epoch": 0,
                "inner-iteration": 24,
                "iter_time": 0.27525643550325185,
                "lr": 1.1331444759206801e-07,
                "total_loss": 1.6766178011894226,
                "vilbert_mutil_loss": 1.6766178011894226
            }

        ]

        $ cat training_status.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...
    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._file_handle = PathManager.open(json_file, 'w')
        self._window_size = window_size
        self.logger = logging.getLogger(__name__)

    def close(self):
        if hasattr(self, '_file_handle'):
            self._file_handle.close()

    def process_buffer_data(self):

        if self.log_buffer.by_epoch:
            data = {'epoch': self.log_buffer.epoch, 'inner-iteration': self.log_buffer.iter}
        else:
            data = {'iteration': self.log_buffer.iter}

        data.update(self.log_buffer.latest_with_smoothing_hint(self._window_size))
        self._file_handle.write(json.dumps(data, sort_keys=True) + '\n')
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            self.logger.exception('Exception in JSONLoggerHook class')
            raise
