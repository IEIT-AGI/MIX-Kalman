from ..builder import HOOKS
from .log_buffer_mixk import LogBufferWriter


@HOOKS.register_module()
class TensorboardLoggerHook(LogBufferWriter):
    """Write all scalars to a tensorboard file."""

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory used to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments will be passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir + '/runs/', **kwargs)

    def process_buffer_data(self):
        self._add_scalar()
        self._add_image()
        self._add_histogram()

    def _add_scalar(self):
        for name, value in self.log_buffer.latest_with_smoothing_hint(self._window_size).items():
            self._writer.add_scalar(name, value, self.log_buffer.iter)

    def _add_image(self):
        if len(self.log_buffer.vis_data) >= 1:
            for img_name, img_data, step_num in self.log_buffer.vis_data.images:
                self._writer.add_image(img_name, img_data, step_num)
            self.log_buffer.clear_images()

    def _add_histogram(self):
        if len(self.log_buffer.histograms) >= 1:
            for hist_params in self.log_buffer.histograms.histograms:
                self._writer.add_histogram_raw(**hist_params)
            self.log_buffer.clear_histograms()

    def close(self):
        if hasattr(self, '_writer'):
            self._writer.close()
