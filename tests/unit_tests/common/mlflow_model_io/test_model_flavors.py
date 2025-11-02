import pytest
from unittest.mock import Mock, patch, MagicMock

from collie._common.mlflow_model_io.base_flavor_handler import FlavorHandler
from collie._common.exceptions import ModelFlavorError
from collie.core.enums.ml_models import ModelFlavor


class TestSklearnFlavorHandler:
    """Test suite for SklearnFlavorHandler."""
    
    @patch('collie._common.mlflow_model_io.model_flavors.SKLEARN_AVAILABLE', True)
    def test_sklearn_handler_imports(self):
        """Test SklearnFlavorHandler can be imported when sklearn is available."""
        from collie._common.mlflow_model_io.model_flavors import SklearnFlavorHandler
        
        handler = SklearnFlavorHandler()
        assert handler is not None
        assert isinstance(handler, FlavorHandler)
    
    @patch('collie._common.mlflow_model_io.model_flavors.SKLEARN_AVAILABLE', True)
    def test_sklearn_can_handle_sklearn_model(self):
        """Test can_handle returns True for sklearn models."""
        from collie._common.mlflow_model_io.model_flavors import SklearnFlavorHandler
        import sklearn.base
        
        # Create a real sklearn-like model
        class MockSklearnModel(sklearn.base.BaseEstimator):
            pass
        
        mock_model = MockSklearnModel()
        handler = SklearnFlavorHandler()
        
        # This should return True because mock_model is instance of BaseEstimator
        assert handler.can_handle(mock_model) is True
    
    @patch('collie._common.mlflow_model_io.model_flavors.SKLEARN_AVAILABLE', True)
    def test_sklearn_flavor_returns_correct_enum(self):
        """Test flavor() returns correct ModelFlavor enum."""
        from collie._common.mlflow_model_io.model_flavors import SklearnFlavorHandler
        
        handler = SklearnFlavorHandler()
        assert handler.flavor() == ModelFlavor.SKLEARN
    
    @patch('collie._common.mlflow_model_io.model_flavors.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.model_flavors.mlflow.sklearn.log_model')
    def test_sklearn_log_model_success(self, mock_log_model):
        """Test log_model successfully logs sklearn model."""
        from collie._common.mlflow_model_io.model_flavors import SklearnFlavorHandler
        
        handler = SklearnFlavorHandler()
        mock_model = Mock()
        
        handler.log_model(mock_model, "model_path")
        
        mock_log_model.assert_called_once_with(sk_model=mock_model, artifact_path="model_path")
    
    @patch('collie._common.mlflow_model_io.model_flavors.SKLEARN_AVAILABLE', False)
    def test_sklearn_log_model_raises_when_unavailable(self):
        """Test log_model raises error when sklearn is unavailable."""
        from collie._common.mlflow_model_io.model_flavors import SklearnFlavorHandler
        
        handler = SklearnFlavorHandler()
        mock_model = Mock()
        
        with pytest.raises(ModelFlavorError, match="scikit-learn is not available"):
            handler.log_model(mock_model, "model_path")
    
    @patch('collie._common.mlflow_model_io.model_flavors.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.model_flavors.mlflow.sklearn.load_model')
    def test_sklearn_load_model_success(self, mock_load_model):
        """Test load_model successfully loads sklearn model."""
        from collie._common.mlflow_model_io.model_flavors import SklearnFlavorHandler
        
        mock_loaded_model = Mock()
        mock_load_model.return_value = mock_loaded_model
        
        handler = SklearnFlavorHandler()
        result = handler.load_model("models:/my_model/1")
        
        assert result is mock_loaded_model
        mock_load_model.assert_called_once_with("models:/my_model/1")


class TestXGBoostFlavorHandler:
    """Test suite for XGBoostFlavorHandler."""
    
    @patch('collie._common.mlflow_model_io.model_flavors.XGBOOST_AVAILABLE', True)
    def test_xgboost_handler_imports(self):
        """Test XGBoostFlavorHandler can be imported when xgboost is available."""
        from collie._common.mlflow_model_io.model_flavors import XGBoostFlavorHandler
        
        handler = XGBoostFlavorHandler()
        assert handler is not None
    
    @patch('collie._common.mlflow_model_io.model_flavors.XGBOOST_AVAILABLE', True)
    def test_xgboost_flavor_returns_correct_enum(self):
        """Test flavor() returns correct ModelFlavor enum."""
        from collie._common.mlflow_model_io.model_flavors import XGBoostFlavorHandler
        
        handler = XGBoostFlavorHandler()
        assert handler.flavor() == ModelFlavor.XGBOOST
    
    @patch('collie._common.mlflow_model_io.model_flavors.XGBOOST_AVAILABLE', False)
    def test_xgboost_log_model_raises_when_unavailable(self):
        """Test log_model raises error when xgboost is unavailable."""
        from collie._common.mlflow_model_io.model_flavors import XGBoostFlavorHandler
        
        handler = XGBoostFlavorHandler()
        mock_model = Mock()
        
        with pytest.raises(ModelFlavorError, match="XGBoost is not available"):
            handler.log_model(mock_model, "model_path")
    
    @patch('collie._common.mlflow_model_io.model_flavors.XGBOOST_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.model_flavors.mlflow.xgboost.log_model')
    def test_xgboost_log_model_with_kwargs(self, mock_log_model):
        """Test log_model passes kwargs correctly."""
        from collie._common.mlflow_model_io.model_flavors import XGBoostFlavorHandler
        
        handler = XGBoostFlavorHandler()
        mock_model = Mock()
        
        handler.log_model(mock_model, "model_path", extra_param="value")
        
        mock_log_model.assert_called_once_with(
            xgb_model=mock_model, 
            artifact_path="model_path", 
            extra_param="value"
        )


class TestPyTorchFlavorHandler:
    """Test suite for PyTorchFlavorHandler."""
    
    @patch('collie._common.mlflow_model_io.model_flavors.PYTORCH_AVAILABLE', True)
    def test_pytorch_handler_imports(self):
        """Test PyTorchFlavorHandler can be imported when pytorch is available."""
        from collie._common.mlflow_model_io.model_flavors import PyTorchFlavorHandler
        
        handler = PyTorchFlavorHandler()
        assert handler is not None
    
    @patch('collie._common.mlflow_model_io.model_flavors.PYTORCH_AVAILABLE', True)
    def test_pytorch_flavor_returns_correct_enum(self):
        """Test flavor() returns correct ModelFlavor enum."""
        from collie._common.mlflow_model_io.model_flavors import PyTorchFlavorHandler
        
        handler = PyTorchFlavorHandler()
        assert handler.flavor() == ModelFlavor.PYTORCH
    
    @patch('collie._common.mlflow_model_io.model_flavors.PYTORCH_AVAILABLE', False)
    def test_pytorch_load_model_raises_when_unavailable(self):
        """Test load_model raises error when pytorch is unavailable."""
        from collie._common.mlflow_model_io.model_flavors import PyTorchFlavorHandler
        
        handler = PyTorchFlavorHandler()
        
        with pytest.raises(ModelFlavorError, match="PyTorch is not available"):
            handler.load_model("models:/my_model/1")


class TestLightGBMFlavorHandler:
    """Test suite for LightGBMFlavorHandler."""
    
    @patch('collie._common.mlflow_model_io.model_flavors.LIGHTGBM_AVAILABLE', True)
    def test_lightgbm_handler_imports(self):
        """Test LightGBMFlavorHandler can be imported when lightgbm is available."""
        from collie._common.mlflow_model_io.model_flavors import LightGBMFlavorHandler
        
        handler = LightGBMFlavorHandler()
        assert handler is not None
    
    @patch('collie._common.mlflow_model_io.model_flavors.LIGHTGBM_AVAILABLE', True)
    def test_lightgbm_flavor_returns_correct_enum(self):
        """Test flavor() returns correct ModelFlavor enum."""
        from collie._common.mlflow_model_io.model_flavors import LightGBMFlavorHandler
        
        handler = LightGBMFlavorHandler()
        assert handler.flavor() == ModelFlavor.LIGHTGBM
    
    @patch('collie._common.mlflow_model_io.model_flavors.LIGHTGBM_AVAILABLE', False)
    def test_lightgbm_log_model_raises_when_unavailable(self):
        """Test log_model raises error when lightgbm is unavailable."""
        from collie._common.mlflow_model_io.model_flavors import LightGBMFlavorHandler
        
        handler = LightGBMFlavorHandler()
        mock_model = Mock()
        
        with pytest.raises(ModelFlavorError, match="LightGBM is not available"):
            handler.log_model(mock_model, "model_path")


class TestTransformersFlavorHandler:
    """Test suite for TransformersFlavorHandler."""
    
    @patch('collie._common.mlflow_model_io.model_flavors.TRANSFORMERS_AVAILABLE', True)
    def test_transformers_handler_imports(self):
        """Test TransformersFlavorHandler can be imported when transformers is available."""
        from collie._common.mlflow_model_io.model_flavors import TransformersFlavorHandler
        
        handler = TransformersFlavorHandler()
        assert handler is not None
    
    @patch('collie._common.mlflow_model_io.model_flavors.TRANSFORMERS_AVAILABLE', True)
    def test_transformers_flavor_returns_correct_enum(self):
        """Test flavor() returns correct ModelFlavor enum."""
        from collie._common.mlflow_model_io.model_flavors import TransformersFlavorHandler
        
        handler = TransformersFlavorHandler()
        assert handler.flavor() == ModelFlavor.TRANSFORMERS
    
    @patch('collie._common.mlflow_model_io.model_flavors.TRANSFORMERS_AVAILABLE', False)
    def test_transformers_load_model_raises_when_unavailable(self):
        """Test load_model raises error when transformers is unavailable."""
        from collie._common.mlflow_model_io.model_flavors import TransformersFlavorHandler
        
        handler = TransformersFlavorHandler()
        
        with pytest.raises(ModelFlavorError, match="Transformers is not available"):
            handler.load_model("models:/my_model/1")


class TestFlavorHandlerErrorHandling:
    """Test error handling in flavor handlers."""
    
    @patch('collie._common.mlflow_model_io.model_flavors.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.model_flavors.mlflow.sklearn.log_model')
    def test_log_model_exception_wrapped(self, mock_log_model):
        """Test that exceptions during log_model are wrapped in ModelFlavorError."""
        from collie._common.mlflow_model_io.model_flavors import SklearnFlavorHandler
        
        mock_log_model.side_effect = Exception("MLflow error")
        
        handler = SklearnFlavorHandler()
        mock_model = Mock()
        
        with pytest.raises(ModelFlavorError, match="Failed to log sklearn model"):
            handler.log_model(mock_model, "model_path")
    
    @patch('collie._common.mlflow_model_io.model_flavors.SKLEARN_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.model_flavors.mlflow.sklearn.load_model')
    def test_load_model_exception_wrapped(self, mock_load_model):
        """Test that exceptions during load_model are wrapped in ModelFlavorError."""
        from collie._common.mlflow_model_io.model_flavors import SklearnFlavorHandler
        
        mock_load_model.side_effect = Exception("Load error")
        
        handler = SklearnFlavorHandler()
        
        with pytest.raises(ModelFlavorError, match="Failed to load sklearn model"):
            handler.load_model("models:/my_model/1")
    
    @patch('collie._common.mlflow_model_io.model_flavors.XGBOOST_AVAILABLE', True)
    @patch('collie._common.mlflow_model_io.model_flavors.mlflow.xgboost.log_model')
    def test_error_includes_model_details(self, mock_log_model):
        """Test that error messages include model details."""
        from collie._common.mlflow_model_io.model_flavors import XGBoostFlavorHandler
        
        mock_log_model.side_effect = Exception("Save failed")
        
        handler = XGBoostFlavorHandler()
        mock_model = Mock(__class__=Mock(__name__="XGBClassifier"))
        
        with pytest.raises(ModelFlavorError) as exc_info:
            handler.log_model(mock_model, "artifacts/model")
        
        # Error should contain details
        assert exc_info.value.details is not None


class TestFlavorAvailabilityFlags:
    """Test framework availability flags."""
    
    def test_availability_flags_are_boolean(self):
        """Test that all availability flags are boolean."""
        from collie._common.mlflow_model_io.model_flavors import (
            SKLEARN_AVAILABLE,
            XGBOOST_AVAILABLE,
            PYTORCH_AVAILABLE,
            LIGHTGBM_AVAILABLE,
            TRANSFORMERS_AVAILABLE
        )
        
        assert isinstance(SKLEARN_AVAILABLE, bool)
        assert isinstance(XGBOOST_AVAILABLE, bool)
        assert isinstance(PYTORCH_AVAILABLE, bool)
        assert isinstance(LIGHTGBM_AVAILABLE, bool)
        assert isinstance(TRANSFORMERS_AVAILABLE, bool)
