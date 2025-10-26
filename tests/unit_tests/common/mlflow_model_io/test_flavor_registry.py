import pytest
from unittest.mock import Mock, patch, MagicMock

from collie._common.mlflow_model_io.flavor_registry import FlavorRegistry
from collie._common.mlflow_model_io.base_flavor_handler import FlavorHandler
from collie._common.exceptions import ModelFlavorError
from collie.core.enums.ml_models import ModelFlavor


class TestFlavorRegistry:
    """Test suite for FlavorRegistry."""
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_registry_initialization_with_one_framework(self):
        """Test FlavorRegistry initialization with one available framework."""
        registry = FlavorRegistry()
        
        assert registry is not None
        assert len(registry._handlers) > 0
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_registry_raises_error_when_no_frameworks_available(self):
        """Test FlavorRegistry raises error when no frameworks are available."""
        with pytest.raises(ModelFlavorError, match="No model flavor handlers available"):
            FlavorRegistry()
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', True)
    def test_registry_registers_all_available_handlers(self):
        """Test FlavorRegistry registers all available handlers."""
        registry = FlavorRegistry()
        
        # Should have 5 handlers (one for each framework)
        assert len(registry._handlers) == 5
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_registry_registers_only_available_handlers(self):
        """Test FlavorRegistry only registers handlers for available frameworks."""
        registry = FlavorRegistry()
        
        # Should have 2 handlers (sklearn and xgboost)
        assert len(registry._handlers) == 2
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    def test_find_handler_by_model_returns_handler(self):
        """Test find_handler_by_model returns correct handler."""
        registry = FlavorRegistry()
        
        # Create a mock model that sklearn handler can handle
        mock_model = Mock()
        
        with patch.object(registry._handlers[0], 'can_handle', return_value=True):
            handler = registry.find_handler_by_model(mock_model)
            
            assert handler is not None
            assert isinstance(handler, FlavorHandler)
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    def test_find_handler_by_model_returns_none_when_no_match(self):
        """Test find_handler_by_model returns None when no handler matches."""
        registry = FlavorRegistry()
        
        mock_model = Mock()
        
        # Mock all handlers to return False
        for handler in registry._handlers:
            handler.can_handle = Mock(return_value=False)
        
        result = registry.find_handler_by_model(mock_model)
        
        assert result is None
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', True)
    def test_find_handler_by_model_returns_first_match(self):
        """Test find_handler_by_model returns first matching handler."""
        registry = FlavorRegistry()
        
        mock_model = Mock()
        
        # Make first handler return True
        registry._handlers[0].can_handle = Mock(return_value=True)
        registry._handlers[1].can_handle = Mock(return_value=True)
        
        handler = registry.find_handler_by_model(mock_model)
        
        # Should return the first handler
        assert handler is registry._handlers[0]
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    def test_find_handler_by_flavor_returns_handler(self):
        """Test find_handler_by_flavor returns correct handler."""
        registry = FlavorRegistry()
        
        # Mock the flavor method
        expected_handler = registry._handlers[0]
        expected_handler.flavor = Mock(return_value=ModelFlavor.SKLEARN)
        
        handler = registry.find_handler_by_flavor(ModelFlavor.SKLEARN)
        
        assert handler is expected_handler
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    def test_find_handler_by_flavor_returns_none_when_no_match(self):
        """Test find_handler_by_flavor returns None for unknown flavor."""
        registry = FlavorRegistry()
        
        # Mock flavor to return something else
        for handler in registry._handlers:
            handler.flavor = Mock(return_value=ModelFlavor.SKLEARN)
        
        result = registry.find_handler_by_flavor(ModelFlavor.PYTORCH)
        
        assert result is None
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_get_available_flavors_returns_list(self):
        """Test get_available_flavors returns list of available flavors."""
        registry = FlavorRegistry()
        
        flavors = registry.get_available_flavors()
        
        assert isinstance(flavors, list)
        assert len(flavors) == 2
        assert any("sklearn" in str(f).lower() for f in flavors)
        assert any("xgboost" in str(f).lower() for f in flavors)
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    def test_get_handler_info_returns_dict(self):
        """Test get_handler_info returns dictionary with handler information."""
        registry = FlavorRegistry()
        
        info = registry.get_handler_info()
        
        assert isinstance(info, dict)
        assert "total_handlers" in info
        assert "available_flavors" in info
        assert "framework_status" in info
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_get_handler_info_correct_counts(self):
        """Test get_handler_info returns correct handler count."""
        registry = FlavorRegistry()
        
        info = registry.get_handler_info()
        
        # Should have 2 handlers (sklearn and pytorch)
        assert info["total_handlers"] == 2
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_get_handler_info_framework_status(self):
        """Test get_handler_info includes framework availability status."""
        registry = FlavorRegistry()
        
        info = registry.get_handler_info()
        
        assert info["framework_status"]["sklearn"] is True
        assert info["framework_status"]["xgboost"] is True
        assert info["framework_status"]["pytorch"] is False
        assert info["framework_status"]["lightgbm"] is False
        assert info["framework_status"]["transformers"] is False


class TestFlavorRegistryIntegration:
    """Integration tests for FlavorRegistry with actual handlers."""
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    def test_registry_workflow_find_and_use_handler(self):
        """Test typical workflow of finding and using a handler."""
        registry = FlavorRegistry()
        
        # Create mock model
        mock_model = Mock()
        
        # Mock can_handle
        with patch.object(registry._handlers[0], 'can_handle', return_value=True):
            handler = registry.find_handler_by_model(mock_model)
            
            assert handler is not None
            
            # Should be able to get flavor from handler
            handler.flavor = Mock(return_value=ModelFlavor.SKLEARN)
            assert handler.flavor() == ModelFlavor.SKLEARN
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', True)
    def test_registry_handles_multiple_model_types(self):
        """Test registry can handle multiple model types."""
        registry = FlavorRegistry()
        
        sklearn_model = Mock()
        xgboost_model = Mock()
        
        # Configure handlers
        registry._handlers[0].can_handle = lambda m: m is sklearn_model
        registry._handlers[1].can_handle = lambda m: m is xgboost_model
        
        sklearn_handler = registry.find_handler_by_model(sklearn_model)
        xgboost_handler = registry.find_handler_by_model(xgboost_model)
        
        assert sklearn_handler is registry._handlers[0]
        assert xgboost_handler is registry._handlers[1]
        assert sklearn_handler is not xgboost_handler
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    def test_registry_find_by_flavor_string(self):
        """Test finding handler by flavor string."""
        registry = FlavorRegistry()
        
        # Mock flavor to return sklearn
        registry._handlers[0].flavor = Mock(return_value=ModelFlavor.SKLEARN)
        
        handler = registry.find_handler_by_flavor(ModelFlavor.SKLEARN)
        
        assert handler is not None
        assert handler is registry._handlers[0]


class TestFlavorRegistryEdgeCases:
    """Test edge cases and error scenarios."""
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    def test_registry_with_none_model(self):
        """Test registry behavior when passed None as model."""
        registry = FlavorRegistry()
        
        handler = registry.find_handler_by_model(None)
        
        # Should return None or the first handler that can handle None
        # Depending on implementation
        assert handler is None or isinstance(handler, FlavorHandler)
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    def test_registry_with_empty_flavor_string(self):
        """Test registry behavior with empty flavor string."""
        registry = FlavorRegistry()
        
        handler = registry.find_handler_by_flavor("")
        
        assert handler is None
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    def test_get_available_flavors_returns_unique_flavors(self):
        """Test that get_available_flavors doesn't return duplicates."""
        registry = FlavorRegistry()
        
        flavors = registry.get_available_flavors()
        
        # Convert to set and compare lengths
        assert len(flavors) == len(set(flavors))
