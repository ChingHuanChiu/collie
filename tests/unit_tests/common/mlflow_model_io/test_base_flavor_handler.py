import pytest
from abc import ABC
from unittest.mock import Mock

from collie._common.mlflow_model_io.base_flavor_handler import FlavorHandler


class TestFlavorHandler:
    """Test suite for FlavorHandler abstract base class."""
    
    def test_flavor_handler_is_abstract(self):
        """Test that FlavorHandler is an abstract base class."""
        assert issubclass(FlavorHandler, ABC)
    
    def test_cannot_instantiate_flavor_handler_directly(self):
        """Test that FlavorHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FlavorHandler()
    
    def test_flavor_handler_has_required_methods(self):
        """Test that FlavorHandler defines all required abstract methods."""
        required_methods = ['can_handle', 'flavor', 'log_model', 'load_model']
        
        for method_name in required_methods:
            assert hasattr(FlavorHandler, method_name)
            assert callable(getattr(FlavorHandler, method_name))
    
    def test_concrete_implementation_must_implement_all_methods(self):
        """Test that concrete implementations must implement all abstract methods."""
        # Missing all implementations
        class IncompleteHandler(FlavorHandler):
            pass
        
        with pytest.raises(TypeError):
            IncompleteHandler()
    
    def test_concrete_implementation_with_all_methods(self):
        """Test that a concrete implementation with all methods can be instantiated."""
        class CompleteHandler(FlavorHandler):
            def can_handle(self, model):
                return True
            
            def flavor(self):
                return "test_flavor"
            
            def log_model(self, model, artifact_path, **kwargs):
                pass
            
            def load_model(self, model_uri):
                return Mock()
        
        # Should not raise
        handler = CompleteHandler()
        assert handler is not None
    
    def test_can_handle_signature(self):
        """Test can_handle method signature."""
        class TestHandler(FlavorHandler):
            def can_handle(self, model):
                return isinstance(model, str)
            
            def flavor(self):
                return "test"
            
            def log_model(self, model, artifact_path, **kwargs):
                pass
            
            def load_model(self, model_uri):
                return None
        
        handler = TestHandler()
        assert handler.can_handle("test") is True
        assert handler.can_handle(123) is False
    
    def test_flavor_method_returns_value(self):
        """Test that flavor method returns a value."""
        class TestHandler(FlavorHandler):
            def can_handle(self, model):
                return True
            
            def flavor(self):
                return "custom_flavor"
            
            def log_model(self, model, artifact_path, **kwargs):
                pass
            
            def load_model(self, model_uri):
                return None
        
        handler = TestHandler()
        assert handler.flavor() == "custom_flavor"
    
    def test_log_model_accepts_kwargs(self):
        """Test that log_model accepts keyword arguments."""
        called_kwargs = {}
        
        class TestHandler(FlavorHandler):
            def can_handle(self, model):
                return True
            
            def flavor(self):
                return "test"
            
            def log_model(self, model, artifact_path, **kwargs):
                called_kwargs.update(kwargs)
            
            def load_model(self, model_uri):
                return None
        
        handler = TestHandler()
        handler.log_model("model", "path", extra_param="value", another_param=42)
        
        assert called_kwargs["extra_param"] == "value"
        assert called_kwargs["another_param"] == 42
    
    def test_load_model_returns_model(self):
        """Test that load_model returns a model object."""
        mock_model = Mock()
        
        class TestHandler(FlavorHandler):
            def can_handle(self, model):
                return True
            
            def flavor(self):
                return "test"
            
            def log_model(self, model, artifact_path, **kwargs):
                pass
            
            def load_model(self, model_uri):
                return mock_model
        
        handler = TestHandler()
        loaded = handler.load_model("models:/test/1")
        
        assert loaded is mock_model


class TestFlavorHandlerContract:
    """Test the contract that FlavorHandler implementations must follow."""
    
    def test_handler_contract_can_handle(self):
        """Test can_handle contract."""
        class TestHandler(FlavorHandler):
            def can_handle(self, model):
                return model == "valid_model"
            
            def flavor(self):
                return "test"
            
            def log_model(self, model, artifact_path, **kwargs):
                pass
            
            def load_model(self, model_uri):
                return None
        
        handler = TestHandler()
        
        # Should return boolean
        result = handler.can_handle("valid_model")
        assert isinstance(result, bool)
        assert result is True
        
        result = handler.can_handle("invalid_model")
        assert isinstance(result, bool)
        assert result is False
    
    def test_handler_contract_flavor(self):
        """Test flavor contract returns identifier."""
        class TestHandler(FlavorHandler):
            def can_handle(self, model):
                return True
            
            def flavor(self):
                return "my_ml_framework"
            
            def log_model(self, model, artifact_path, **kwargs):
                pass
            
            def load_model(self, model_uri):
                return None
        
        handler = TestHandler()
        flavor_name = handler.flavor()
        
        # Should return some identifier (string or enum)
        assert flavor_name is not None
        assert flavor_name == "my_ml_framework"
    
    def test_handler_contract_log_model(self):
        """Test log_model contract."""
        logged_data = {}
        
        class TestHandler(FlavorHandler):
            def can_handle(self, model):
                return True
            
            def flavor(self):
                return "test"
            
            def log_model(self, model, artifact_path, **kwargs):
                logged_data['model'] = model
                logged_data['path'] = artifact_path
                logged_data['kwargs'] = kwargs
            
            def load_model(self, model_uri):
                return None
        
        handler = TestHandler()
        handler.log_model("my_model", "artifacts/model", param1="value1")
        
        assert logged_data['model'] == "my_model"
        assert logged_data['path'] == "artifacts/model"
        assert logged_data['kwargs']['param1'] == "value1"
    
    def test_handler_contract_load_model(self):
        """Test load_model contract."""
        class TestHandler(FlavorHandler):
            def can_handle(self, model):
                return True
            
            def flavor(self):
                return "test"
            
            def log_model(self, model, artifact_path, **kwargs):
                pass
            
            def load_model(self, model_uri):
                # Simulate loading
                return {"loaded_from": model_uri}
        
        handler = TestHandler()
        model = handler.load_model("models:/my_model/1")
        
        assert model is not None
        assert model["loaded_from"] == "models:/my_model/1"
