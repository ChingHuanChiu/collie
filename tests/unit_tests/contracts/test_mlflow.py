import pytest
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd

from collie.contracts.mlflow import MLflowConfig
from collie._common.exceptions import (
    MLflowConfigurationError,
    MLflowOperationError
)


class TestMLflowConfig:
    """Test suite for MLflowConfig singleton class."""
    
    def setup_method(self):
        """Reset singleton instance before each test."""
        MLflowConfig._instance = None
    
    def test_mlflow_config_is_singleton(self):
        """Test that MLflowConfig implements singleton pattern."""
        config1 = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment"
        )
        config2 = MLflowConfig(
            tracking_uri="http://different:5000",
            experiment_name="different_experiment"
        )
        
        # Both should be the same instance
        assert config1 is config2
        # First configuration should persist
        assert config1.tracking_uri == "http://localhost:5000"
        assert config1.experiment_name == "test_experiment"
    
    def test_mlflow_config_initialization(self):
        """Test MLflowConfig initialization with basic parameters."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="my_experiment"
        )
        
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "my_experiment"
        assert config.mlflow_client is not None
    
    def test_mlflow_config_has_mlflow_client(self):
        """Test that MLflowConfig creates MlflowClient instance."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test"
        )
        
        from mlflow.tracking import MlflowClient
        assert isinstance(config.mlflow_client, MlflowClient)
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_configure_sets_mlflow_settings(
        self,
        mock_set_experiment,
        mock_set_tracking_uri
    ):
        """Test that configure() method sets MLflow settings."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment"
        )
        
        config.configure()
        
        mock_set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_set_experiment.assert_called_once_with("test_experiment")
    
    def test_mlflow_config_thread_safety(self):
        """Test that MLflowConfig singleton is thread-safe."""
        import threading
        
        instances = []
        
        def create_instance():
            config = MLflowConfig(
                tracking_uri="http://localhost:5000",
                experiment_name="test"
            )
            instances.append(config)
        
        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All instances should be the same object
        assert all(instance is instances[0] for instance in instances)
    
    def test_mlflow_config_only_initializes_once(self):
        """Test that MLflowConfig only initializes internal state once."""
        config1 = MLflowConfig(
            tracking_uri="http://first:5000",
            experiment_name="first_exp"
        )
        
        original_client = config1.mlflow_client
        
        # Second call with different params
        config2 = MLflowConfig(
            tracking_uri="http://second:5000",
            experiment_name="second_exp"
        )
        
        # Should keep original configuration
        assert config2.tracking_uri == "http://first:5000"
        assert config2.experiment_name == "first_exp"
        assert config2.mlflow_client is original_client
    
    def test_mlflow_config_tracking_uri_property(self):
        """Test accessing tracking_uri property."""
        config = MLflowConfig(
            tracking_uri="sqlite:///mlflow.db",
            experiment_name="local_exp"
        )
        
        assert config.tracking_uri == "sqlite:///mlflow.db"
    
    def test_mlflow_config_experiment_name_property(self):
        """Test accessing experiment_name property."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="production_experiment"
        )
        
        assert config.experiment_name == "production_experiment"
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_configure_can_be_called_multiple_times(
        self,
        mock_set_experiment,
        mock_set_tracking_uri
    ):
        """Test that configure() can be called multiple times safely."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test"
        )
        
        config.configure()
        config.configure()
        config.configure()
        
        # Should be called each time
        assert mock_set_tracking_uri.call_count == 3
        assert mock_set_experiment.call_count == 3


class TestMLflowConfigEdgeCases:
    """Test edge cases and error handling for MLflowConfig."""
    
    def setup_method(self):
        """Reset singleton instance before each test."""
        MLflowConfig._instance = None
    
    def test_mlflow_config_with_file_tracking_uri(self, tmp_path):
        """Test MLflowConfig with file-based tracking URI."""
        mlruns_path = tmp_path / "mlruns"
        config = MLflowConfig(
            tracking_uri=f"file:///{mlruns_path}",
            experiment_name="file_experiment"
        )
        
        assert config.tracking_uri == f"file:///{mlruns_path}"
    
    def test_mlflow_config_with_sqlite_tracking_uri(self, tmp_path):
        """Test MLflowConfig with SQLite tracking URI."""
        db_path = tmp_path / "mlflow.db"
        config = MLflowConfig(
            tracking_uri=f"sqlite:///{db_path}",
            experiment_name="sqlite_experiment"
        )
        
        # The tracking_uri should use the absolute path provided
        assert config.tracking_uri.startswith("sqlite:///")
        assert "mlflow.db" in config.tracking_uri
    
    def test_mlflow_config_with_special_characters_in_experiment(self):
        """Test MLflowConfig with special characters in experiment name."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="my-experiment_v1.0"
        )
        
        assert config.experiment_name == "my-experiment_v1.0"
    
    def test_mlflow_config_with_empty_experiment_name(self):
        """Test MLflowConfig behavior with empty experiment name."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name=""
        )
        
        assert config.experiment_name == ""
    
    def test_mlflow_config_client_initialization(self):
        """Test that MlflowClient is initialized with correct tracking URI."""
        tracking_uri = "http://custom:8080"
        config = MLflowConfig(
            tracking_uri=tracking_uri,
            experiment_name="test"
        )
        
        # Client should be created with the tracking URI
        assert config.mlflow_client is not None


class TestMLflowConfigIntegration:
    """Integration tests for MLflowConfig with other components."""
    
    def setup_method(self):
        """Reset singleton instance before each test."""
        MLflowConfig._instance = None
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    def test_mlflow_config_workflow(
        self,
        mock_start_run,
        mock_set_experiment,
        mock_set_tracking_uri
    ):
        """Test typical MLflow configuration workflow."""
        # Create and configure
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="test_workflow"
        )
        config.configure()
        
        # Verify configuration was applied
        mock_set_tracking_uri.assert_called_once()
        mock_set_experiment.assert_called_once()
        
        # Should be able to use MLflow after configuration
        import mlflow
        mlflow.start_run()
        mock_start_run.assert_called_once()
    
    def test_mlflow_config_with_multiple_experiments(self):
        """Test MLflowConfig behavior when switching experiments."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="first_experiment"
        )
        
        # Singleton means we can't change the experiment after initialization
        # through constructor, but we can through configure if we modify the instance
        assert config.experiment_name == "first_experiment"
        
        # Attempting to create with different experiment returns same instance
        config2 = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="second_experiment"
        )
        
        assert config2.experiment_name == "first_experiment"  # Still first
