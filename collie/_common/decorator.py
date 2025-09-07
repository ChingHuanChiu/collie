from typing import Tuple, List
from functools import wraps


def type_checker(
    typing: Tuple[type], 
    error_msg: str
):
    """
    A decorator that checks the type of the output of a function.

    Args:
        typing (Tuple[type]): A tuple of types to check against.
        error_msg (str): The error message to be raised if the type does not match.

    Raises:
        TypeError: If the type of the output of the function does not match with given types.
    """
    
    def closure(func):
        @wraps(func)
        def wrapper(*arg, **kwarg):
            result = func(*arg, **kwarg)
            if not isinstance(result, typing):
                raise TypeError(error_msg)
            return result
        return wrapper
    return closure


def dict_key_checker(keys: List[str]):
    """
    A decorator that checks the keys of the output of a function.

    Args:
        keys (List[str]): A list of keys to check against.

    Raises:
        TypeError: If the output of the function is not a dictionary.
        KeyError: If the output of the function does not contain all the keys in the list.
    """
    def closure(func):
        @wraps(func)
        def wrapper(*arg, **kwarg):
            result = func(*arg, **kwarg)
            if not isinstance(result, dict):
                raise TypeError("The output must be a dictionary.")
            all_keys_exist = all(key in result for key in keys)
            if not all_keys_exist:
                raise KeyError(f"The following keys must all exist in the output: {keys}. Output: {result}")
            return result
        return wrapper
    return closure