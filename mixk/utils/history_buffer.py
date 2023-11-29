from collections import namedtuple

from numpy import mean, median

Scalar = namedtuple('Scalar', ('data', 'iter'))


class HistoryBuffer:

    def __init__(self, capacity: int = 2**20):
        self._capacity = capacity
        self._size = 0
        self._data: list = []
        self._global_mean: float = 0

    def update(self, data: float, iteration: int = None):
        if iteration is None:
            iteration = self._size
        if len(self._data) == self._capacity:
            self._data.pop(0)

        self._size += 1
        self._update_global_mean(value=data)
        self._update_data(data, iteration)

    @property
    def global_mean(self) -> float:
        return self._global_mean

    # @global_mean.setter
    # def global_mean(self, value):
    #     self._global_mean = value

    @property
    def size(self) -> int:
        return self._size

    # @size.setter
    # def size(self, value):
    #     self._size = value

    @property
    def data(self) -> list:
        return self._data

    def _update_global_mean(self, value):
        self._global_mean += (value - self.global_mean) / self._size

    def _update_data(self, data: float, iteration: int):
        self._data.append(Scalar(data, iteration))

    def latest_data(self):
        return self._data[-1][0]

    def latest_window_median(self, window_size: int) -> float:
        latest_window_data = self._get_latest_window_data(window_size)
        return median(latest_window_data)

    def lateset_window_mean(self, window_size: int) -> float:
        latest_window_data = self._get_latest_window_data(window_size)
        return mean(latest_window_data)

    def _get_latest_window_data(self, window_size: int) -> list:
        latest_data = []
        for d in self._data[-window_size:]:
            latest_data.append(d[0])
        return latest_data

    avg = lateset_window_mean
    median = latest_window_median
    values = data
    latest = latest_data
