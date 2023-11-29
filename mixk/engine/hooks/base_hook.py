import enum
from typing import Union


@enum.unique
class PriorityStatus(enum.Enum):
    HIGHEST = 0
    HIGHER = 10
    HIGH = 20
    NORMAL = 30
    LOW = 40
    LOWER = 50
    LOWEST = 60


class HookBase:
    """HookBase is the base class of all hook and is registered in EngineBase class.
    The subclasses of HookBase implement the following six methods according to needed.
    ::
        hook.before_train()
        for iter in range(start_iter,max_iter):
            hook.before_train_iter()
            train.run_train_iter()
            hook.after_train_iter()
        hook.after_epoch()
        hook.after_train()
    """

    def __init__(self):
        self._level = PriorityStatus.NORMAL

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_train_iter(self):
        pass

    def after_train_iter(self):
        pass

    def before_train_epoch(self):
        pass

    def after_train_epoch(self):
        pass

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level: Union[PriorityStatus, str, int]):
        assert isinstance(level, (PriorityStatus, str, int))

        if isinstance(level, PriorityStatus):
            self._level = level
        else:
            level_name = list(PriorityStatus.__members__.keys())
            level_value = list(lv.value for lv in PriorityStatus.__members__.values())
            if level in level_name or level in level_value:
                self._level = PriorityStatus[level.upper()] if isinstance(level_name, str) else PriorityStatus(level)
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f'because level{level} is not a PriorityStatus type,it will be set as default type')
                self._level = PriorityStatus.NORMAL
