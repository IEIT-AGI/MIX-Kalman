import inspect
from typing import Dict, Optional, Any
from tabulate import tabulate


class Registry:
    """Registry class: mapping to the corresponding object based on the
    provided name."""

    def __init__(self, name: str) -> None:
        self._module_name = name
        self._module_dict: Dict[str, object] = {}

    def _register_module(self, module_name: str, module_class: object) -> None:
        assert (inspect.isclass(module_class)), 'module must be a class, but the input module type is {}'.format(
            type(module_class))
        assert (module_name not in self._module_dict), 'A module named {} was already registered in {}'.format(
            module_name, self._module_name)

        self._module_dict[module_name] = module_class

    def register_module(self, name: str = None, module: object = None) -> Optional[object]:

        if module is None:
            # decorator:@xxx.register_module(module=class)
            def _register(class_or_func: object) -> object:
                self._register_module(class_or_func.__name__, class_or_func)
                return class_or_func

            return _register
        else:
            module_name = module.__name__ if name is None else name
            self._register_module(module_name, module)

    def get(self, name: str) -> object:
        obj = self._module_dict.get(name)
        if obj is None:
            raise KeyError('{} is not in the {} registry!'.format(name, self._module_name))
        return obj

    def __contains__(self, name: str) -> bool:
        return name in self._module_dict

    def __repr__(self) -> str:
        registry_module_content = tabulate(
            self._module_dict.items(), headers=['Names', 'Modules'], tablefmt='fancy_grid')
        registry_module_name = 'registry of {}:\n'.format(self._module_name)
        return registry_module_name + registry_module_content

    __str__ = __repr__

    @property
    def module_name(self) -> str:
        return self._module_name

    @property
    def module_dict(self) -> Dict:
        return self._module_dict

    def __len__(self):
        return len(self._module_dict)


def check_var_type(var: Any, var_name: str, desired_var_type: Any) -> None:
    msg = f'{var_name} should be a {desired_var_type.__name__} type ,but got {type(var).__name__} type'
    assert isinstance(var, desired_var_type), TypeError(msg)


def build_from_cfg(cfg: Dict, registry: Registry, default_args: Optional[Dict] = None) -> object:
    check_var_type(cfg, 'cfg', dict)
    check_var_type(registry, 'registry', Registry)
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError('The cfg or default_args  must contain the key type!')

    def collect_all_args():
        args = cfg.copy()
        if default_args is not None:
            for k, v in default_args.items():
                args.setdefault(k, v)
        return args

    args = collect_all_args()
    object_type = args.pop('type')

    if isinstance(object_type, str):
        object_class = registry.get(object_type)
    elif inspect.isclass(object_type):
        object_class = object_type
    else:
        raise TypeError('The type in cfg must be a string or class type, but got{}'.format(type(object_type)))

    try:
        return object_class(**args)
    except Exception as e:
        raise type(e)(f'{object_class.__name__}: {e}')
