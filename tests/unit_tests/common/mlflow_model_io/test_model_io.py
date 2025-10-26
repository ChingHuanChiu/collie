import pytest
from unittest.mock import Mock, patch, MagicMock, call

from collie._common.mlflow_model_io.model_io import MLflowModelIO
from collie._common.mlflow_model_io.flavor_registry import FlavorRegistry
from collie._common.exceptions import ModelFlavorError
from collie.core.enums.ml_models import ModelFlavor


class TestMLflowModelIO:
    """Test suite for MLflowModelIO class."""
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_mlflow_model_io_initialization(self):
        """Test MLflowModelIO can be initialized with MLflow client."""
        mock_client = Mock()
        
        model_io = MLflowModelIO(mock_client)
        
        assert model_io is not None
        assert model_io.client is mock_client
        assert isinstance(model_io.registry, FlavorRegistry)
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_log_model_success(self):
        """Test log_model successfully logs a model."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        mock_model = Mock()
        mock_handler = Mock()
        mock_handler.flavor.return_value = ModelFlavor.SKLEARN
        
        with patch.object(model_io.registry, 'find_handler_by_model', return_value=mock_handler):
            with patch('mlflow.log_param') as mock_log_param:
                model_io.log_model(mock_model, "model_name")
                
                mock_handler.log_model.assert_called_once_with(
                    mock_model,
                    "model_name",
                    registered_model_name=None
                )
                mock_log_param.assert_called_once()
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_log_model_with_registered_name(self):
        """Test log_model with registered_model_name parameter."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        mock_model = Mock()
        mock_handler = Mock()
        mock_handler.flavor.return_value = ModelFlavor.SKLEARN
        
        with patch.object(model_io.registry, 'find_handler_by_model', return_value=mock_handler):
            with patch('mlflow.log_param'):
                model_io.log_model(
                    mock_model,
                    "model_name",
                    registered_model_name="my_registered_model"
                )
                
                mock_handler.log_model.assert_called_once_with(
                    mock_model,
                    "model_name",
                    registered_model_name="my_registered_model"
                )
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_log_model_with_extra_kwargs(self):
        """Test log_model passes extra kwargs to handler."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        mock_model = Mock()
        mock_handler = Mock()
        mock_handler.flavor.return_value = ModelFlavor.SKLEARN
        
        with patch.object(model_io.registry, 'find_handler_by_model', return_value=mock_handler):
            with patch('mlflow.log_param'):
                model_io.log_model(
                    mock_model,
                    "model_name",
                    extra_param="value",
                    another_param=42
                )
                
                mock_handler.log_model.assert_called_once_with(
                    mock_model,
                    "model_name",
                    registered_model_name=None,
                    extra_param="value",
                    another_param=42
                )
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_log_model_raises_error_for_unsupported_model(self):
        """Test log_model raises ValueError for unsupported model type."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        mock_model = Mock()
        
        with patch.object(model_io.registry, 'find_handler_by_model', return_value=None):
            with pytest.raises(ValueError, match="Unsupported model type"):
                model_io.log_model(mock_model, "model_name")
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_log_model_logs_flavor_parameter(self):
        """Test log_model logs the model flavor as MLflow parameter."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        mock_model = Mock()
        mock_handler = Mock()
        mock_handler.flavor.return_value = ModelFlavor.SKLEARN
        
        with patch.object(model_io.registry, 'find_handler_by_model', return_value=mock_handler):
            with patch('mlflow.log_param') as mock_log_param:
                model_io.log_model(mock_model, "model_name")
                
                mock_log_param.assert_called_once_with("model_flavor", ModelFlavor.SKLEARN)
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_load_model_success(self):
        """Test load_model successfully loads a model."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        mock_loaded_model = Mock()
        mock_handler = Mock()
        mock_handler.load_model.return_value = mock_loaded_model
        
        with patch.object(model_io.registry, 'find_handler_by_flavor', return_value=mock_handler):
            result = model_io.load_model(ModelFlavor.SKLEARN, "models:/my_model/1")
            
            assert result is mock_loaded_model
            mock_handler.load_model.assert_called_once_with("models:/my_model/1")
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_load_model_raises_error_for_unsupported_flavor(self):
        """Test load_model raises ValueError for unsupported flavor."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        with patch.object(model_io.registry, 'find_handler_by_flavor', return_value=None):
            with pytest.raises(ValueError, match="Unsupported model flavor"):
                model_io.load_model("unknown_flavor", "models:/my_model/1")
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_load_model_with_different_flavors(self):
        """Test load_model can load models with different flavors."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        sklearn_model = Mock()
        xgboost_model = Mock()
        
        sklearn_handler = Mock()
        sklearn_handler.load_model.return_value = sklearn_model
        
        xgboost_handler = Mock()
        xgboost_handler.load_model.return_value = xgboost_model
        
        def mock_find_handler(flavor):
            if flavor == ModelFlavor.SKLEARN:
                return sklearn_handler
            elif flavor == ModelFlavor.XGBOOST:
                return xgboost_handler
            return None
        
        with patch.object(model_io.registry, 'find_handler_by_flavor', side_effect=mock_find_handler):
            result1 = model_io.load_model(ModelFlavor.SKLEARN, "models:/sklearn/1")
            result2 = model_io.load_model(ModelFlavor.XGBOOST, "models:/xgboost/1")
            
            assert result1 is sklearn_model
            assert result2 is xgboost_model


class TestMLflowModelIOIntegration:
    """Integration tests for MLflowModelIO."""
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_end_to_end_log_and_load_workflow(self):
        """Test end-to-end workflow of logging and loading a model."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        # Log model
        mock_model = Mock()
        mock_handler = Mock()
        mock_handler.flavor.return_value = ModelFlavor.SKLEARN
        mock_handler.load_model.return_value = mock_model
        
        with patch.object(model_io.registry, 'find_handler_by_model', return_value=mock_handler):
            with patch.object(model_io.registry, 'find_handler_by_flavor', return_value=mock_handler):
                with patch('mlflow.log_param'):
                    # Log the model
                    model_io.log_model(mock_model, "my_model")
                    
                    # Load the model
                    loaded_model = model_io.load_model(ModelFlavor.SKLEARN, "models:/my_model/1")
                    
                    assert loaded_model is mock_model
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_model_io_handles_multiple_frameworks(self):
        """Test MLflowModelIO can handle multiple frameworks."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        # Should have registry with 2 handlers
        assert len(model_io.registry._handlers) == 2
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_model_io_with_mlflow_client(self):
        """Test MLflowModelIO works with MlflowClient instance."""
        from mlflow.tracking import MlflowClient
        
        # Can use real or mock client
        mock_client = Mock(spec=MlflowClient)
        model_io = MLflowModelIO(mock_client)
        
        assert model_io.client is mock_client


class TestMLflowModelIOEdgeCases:
    """Test edge cases and error scenarios."""
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_log_model_with_none_model(self):
        """Test log_model behavior with None model."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        with patch.object(model_io.registry, 'find_handler_by_model', return_value=None):
            with pytest.raises(ValueError):
                model_io.log_model(None, "model_name")
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_load_model_with_invalid_uri(self):
        """Test load_model with invalid model URI."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        mock_handler = Mock()
        mock_handler.load_model.side_effect = Exception("Invalid URI")
        
        with patch.object(model_io.registry, 'find_handler_by_flavor', return_value=mock_handler):
            with pytest.raises(Exception):
                model_io.load_model(ModelFlavor.SKLEARN, "invalid_uri")
    
    @patch('collie._common.mlflow_model_io.flavor_registry.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.flavor_registry.XGBOOST_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.PYTORCH_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.LIGHTGBM_AVAILABLE', False)
    @patch('collie._common.mlflow_model_io.flavor_registry.TRANSFORMERS_AVAILABLE', False)
    def test_log_model_with_empty_name(self):
        """Test log_model with empty model name."""
        mock_client = Mock()
        model_io = MLflowModelIO(mock_client)
        
        mock_model = Mock()
        mock_handler = Mock()
        mock_handler.flavor.return_value = ModelFlavor.SKLEARN
        
        with patch.object(model_io.registry, 'find_handler_by_model', return_value=mock_handler):
            with patch('mlflow.log_param'):
                # Should still work, might be handled by MLflow
                model_io.log_model(mock_model, "")
                
                mock_handler.log_model.assert_called_once()
