import pytest
from unittest.mock import Mock
import pandas as pd

from collie._common.decorator import type_checker, dict_key_checker
from collie.core.models import TransformerPayload, TrainerPayload


class TestTypeChecker:
    
    def test_type_checker_with_valid_single_type(self):
        @type_checker((str,), "Must return string")
        def return_string():
            return "hello"
        
        result = return_string()
        assert result == "hello"
    
    def test_type_checker_raises_type_error_for_invalid_type(self):
        @type_checker((str,), "Must return string")
        def return_int():
            return 42
        
        with pytest.raises(TypeError, match="Must return string"):
            return_int()
    
    def test_type_checker_with_multiple_valid_types(self):
        """Test type_checker accepts multiple valid types."""
        @type_checker((str, int), "Must return string or int")
        def return_value(return_type):
            if return_type == "string":
                return "hello"
            return 42
        
        assert return_value("string") == "hello"
        assert return_value("int") == 42
    
    def test_type_checker_with_invalid_type_from_multiple(self):
        """Test type_checker raises error when type not in allowed types."""
        @type_checker((str, int), "Must return string or int")
        def return_list():
            return [1, 2, 3]
        
        with pytest.raises(TypeError, match="Must return string or int"):
            return_list()
    
    def test_type_checker_with_complex_types(self):
        """Test type_checker with Pydantic model types."""
        @type_checker((TransformerPayload,), "Must return TransformerPayload")
        def create_payload():
            return TransformerPayload()
        
        result = create_payload()
        assert isinstance(result, TransformerPayload)
    
    def test_type_checker_with_dataframe(self):
        """Test type_checker with pandas DataFrame."""
        @type_checker((pd.DataFrame,), "Must return DataFrame")
        def create_dataframe():
            return pd.DataFrame({"col": [1, 2, 3]})
        
        result = create_dataframe()
        assert isinstance(result, pd.DataFrame)
    
    def test_type_checker_preserves_function_name(self):
        """Test that decorator preserves function name."""
        @type_checker((str,), "Error message")
        def my_function():
            """Function docstring."""
            return "result"
        
        assert my_function.__name__ == "my_function"
    
    def test_type_checker_preserves_function_docstring(self):
        """Test that decorator preserves function docstring."""
        @type_checker((str,), "Error message")
        def documented_function():
            """This is a documented function."""
            return "result"
        
        assert documented_function.__doc__ == "This is a documented function."
    
    def test_type_checker_with_function_arguments(self):
        """Test type_checker with functions that take arguments."""
        @type_checker((str,), "Must return string")
        def process_data(data, prefix=""):
            return f"{prefix}{data}"
        
        result = process_data("test", prefix="processed_")
        assert result == "processed_test"
    
    def test_type_checker_with_kwargs(self):
        """Test type_checker with keyword arguments."""
        @type_checker((dict,), "Must return dict")
        def create_dict(**kwargs):
            return kwargs
        
        result = create_dict(key1="value1", key2="value2")
        assert result == {"key1": "value1", "key2": "value2"}
    
    def test_type_checker_with_none_type(self):
        """Test type_checker accepting None as valid type."""
        @type_checker((str, type(None)), "Must return string or None")
        def maybe_return_string(should_return):
            if should_return:
                return "value"
            return None
        
        assert maybe_return_string(True) == "value"
        assert maybe_return_string(False) is None
    
    def test_type_checker_with_list_type(self):
        """Test type_checker with list type."""
        @type_checker((list,), "Must return list")
        def return_list():
            return [1, 2, 3]
        
        result = return_list()
        assert result == [1, 2, 3]
    
    def test_type_checker_with_dict_type(self):
        """Test type_checker with dict type."""
        @type_checker((dict,), "Must return dict")
        def return_dict():
            return {"key": "value"}
        
        result = return_dict()
        assert result == {"key": "value"}


class TestDictKeyChecker:
    
    def test_dict_key_checker_with_all_keys_present(self):
        """Test dict_key_checker when all required keys are present."""
        @dict_key_checker(["name", "age"])
        def get_person():
            return {"name": "John", "age": 30, "city": "NYC"}
        
        result = get_person()
        assert result["name"] == "John"
        assert result["age"] == 30
    
    def test_dict_key_checker_raises_type_error_for_non_dict(self):
        """Test dict_key_checker raises TypeError for non-dict return."""
        @dict_key_checker(["key1"])
        def return_string():
            return "not a dict"
        
        with pytest.raises(TypeError, match="The output must be a dictionary"):
            return_string()
    
    def test_dict_key_checker_raises_key_error_for_missing_keys(self):
        """Test dict_key_checker raises KeyError for missing required keys."""
        @dict_key_checker(["name", "age", "email"])
        def get_incomplete_person():
            return {"name": "John", "age": 30}
        
        with pytest.raises(KeyError, match="The following keys must all exist"):
            get_incomplete_person()
    
    def test_dict_key_checker_with_empty_key_list(self):
        """Test dict_key_checker with empty required keys list."""
        @dict_key_checker([])
        def return_any_dict():
            return {"any": "data", "is": "valid"}
        
        result = return_any_dict()
        assert "any" in result
    
    def test_dict_key_checker_with_single_key(self):
        """Test dict_key_checker with single required key."""
        @dict_key_checker(["status"])
        def get_status():
            return {"status": "success", "extra": "data"}
        
        result = get_status()
        assert result["status"] == "success"
    
    def test_dict_key_checker_preserves_function_metadata(self):
        """Test that dict_key_checker preserves function metadata."""
        @dict_key_checker(["key"])
        def my_function():
            """Returns a dict."""
            return {"key": "value"}
        
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "Returns a dict."
    
    def test_dict_key_checker_with_function_arguments(self):
        """Test dict_key_checker with function that takes arguments."""
        @dict_key_checker(["result"])
        def process(value):
            return {"result": value * 2, "metadata": "processed"}
        
        result = process(5)
        assert result["result"] == 10
    
    def test_dict_key_checker_with_nested_dict(self):
        """Test dict_key_checker with nested dictionary."""
        @dict_key_checker(["data", "metadata"])
        def get_nested_data():
            return {
                "data": {"nested": "value"},
                "metadata": {"created": "2025-10-26"}
            }
        
        result = get_nested_data()
        assert "data" in result
        assert "metadata" in result
    
    def test_dict_key_checker_error_message_includes_keys(self):
        """Test that error message includes list of required keys."""
        @dict_key_checker(["key1", "key2", "key3"])
        def missing_keys():
            return {"key1": "value1"}
        
        with pytest.raises(KeyError) as exc_info:
            missing_keys()
        
        error_message = str(exc_info.value)
        assert "key1" in error_message
        assert "key2" in error_message
        assert "key3" in error_message
    
    def test_dict_key_checker_with_numeric_values(self):
        """Test dict_key_checker with numeric dictionary values."""
        @dict_key_checker(["count", "total"])
        def get_stats():
            return {"count": 10, "total": 100, "average": 10.0}
        
        result = get_stats()
        assert result["count"] == 10
        assert result["total"] == 100


class TestCombinedDecorators:
    """Test combining multiple decorators."""
    
    def test_type_and_dict_checker_combined(self):
        """Test using both type_checker and dict_key_checker together."""
        @type_checker((dict,), "Must return dict")
        @dict_key_checker(["status", "data"])
        def get_response():
            return {"status": "success", "data": [1, 2, 3]}
        
        result = get_response()
        assert result["status"] == "success"
        assert isinstance(result["data"], list)
    
    def test_multiple_type_checkers(self):
        """Test stacking multiple type checkers (if applicable)."""
        @type_checker((dict,), "Must return dict")
        def create_config():
            return {"setting": "value"}
        
        result = create_config()
        assert isinstance(result, dict)
    
    def test_decorator_with_class_method(self):
        """Test decorators work with class methods."""
        class DataProcessor:
            @type_checker((dict,), "Must return dict")
            @dict_key_checker(["processed"])
            def process(self, data):
                return {"processed": data, "timestamp": "2025-10-26"}
        
        processor = DataProcessor()
        result = processor.process("test_data")
        assert result["processed"] == "test_data"
    
    def test_decorator_with_static_method(self):
        """Test decorators work with static methods."""
        class Utils:
            @staticmethod
            @type_checker((str,), "Must return string")
            def format_name(name):
                return f"Formatted: {name}"
        
        result = Utils.format_name("Test")
        assert result == "Formatted: Test"
    
    def test_decorator_with_class_method_decorator(self):
        """Test decorators work with @classmethod."""
        class Factory:
            @classmethod
            @type_checker((dict,), "Must return dict")
            def create_config(cls):
                return {"type": cls.__name__}
        
        result = Factory.create_config()
        assert result["type"] == "Factory"