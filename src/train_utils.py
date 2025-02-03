import importlib
from typing import Any


def load_object(obj_path: str, default_obj_path: str = '') -> Any:
    """
    Loads an object (e.g., class or function) from a given import path.

    Args:
        obj_path (str): The full import path of the object.
        default_obj_path (str): The default module path to use if not provided in `obj_path`.

    Returns:
        Any: The loaded object.

    Raises:
        AttributeError: If the object cannot be found in the module.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    try:
        return getattr(module_obj, obj_name)
    except AttributeError:
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
