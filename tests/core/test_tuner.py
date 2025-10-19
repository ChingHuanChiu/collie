import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from typing import Dict, Any

from collie.contracts.event import Event, EventType, PipelineContext
from collie.core.models import TunerPayload, TunerArtifact
from collie.core.tuner.tuner import Tuner
from collie._common.exceptions import TunerError


class TestTuner:
    """Test cases for Tuner component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tuner = Tuner(
            description="Test tuner",
            tags={"env": "test"}
        )
    
    def test_tuner_initialization(self):
        """Test Tuner initialization."""
        assert self.tuner.description == "Test tuner"
        assert self.tuner.tags == {"env": "test"}
    
    def test_tuner_initialization_defaults(self):
        """Test Tuner initialization with defaults."""
        tuner = Tuner()
        assert tuner.description is None
        assert tuner.tags == {"component": "Tuner"}
    
    @patch.object(Tuner, 'start_run')
    @patch.object(Tuner, '_handle')
    @patch.object(Tuner, 'mlflow')
    def test_run_success(self, mock_mlflow, mock_handle, mock_start_run):
        """Test successful tuner run."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.artifact_uri = "test://artifacts"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create test hyperparameters
        hyperparams = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 10
        }
        train_data = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        
        payload = TunerPayload(
            hyperparameters=hyperparams,
            train_data=train_data
        )
        
        # Setup event
        context = PipelineContext()
        input_event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = self.tuner.run(input_event)
        
        # Verify
        assert result.type == EventType.TUNING_DONE
        assert isinstance(result.payload, TunerPayload)
        assert result.context == input_event.context
        
        # Verify MLflow run was started correctly
        mock_start_run.assert_called_once_with(
            tags={"env": "test"},
            run_name="Tuner",
            log_system_metrics=True,
            nested=True,
            description="Test tuner"
        )
        
        # Verify hyperparameters were logged
        expected_hyperparams = payload.model_dump()
        mock_mlflow.log_dict.assert_called_once_with(
            dictionary=expected_hyperparams,
            artifact_file=TunerArtifact().hyperparameters
        )
        
        # Verify handle was called
        mock_handle.assert_called_once_with(input_event)
        
        # Verify hyperparameters URI was set in context
        expected_uri = f"test://artifacts/{TunerArtifact().hyperparameters}"
        assert result.context.get("hyperparameters_uri") == expected_uri
    
    @patch.object(Tuner, 'start_run')
    @patch.object(Tuner, '_handle')
    def test_run_with_exception(self, mock_handle, mock_start_run):
        """Test tuner run with exception."""
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Make handle raise exception
        mock_handle.side_effect = Exception("Test error")
        
        # Create test event
        payload = TunerPayload(hyperparameters={"lr": 0.01})
        input_event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=PipelineContext()
        )
        
        # Execute and verify exception
        with pytest.raises(TunerError, match="Tuner failed with error: Test error"):
            self.tuner.run(input_event)
    
    def test_tuner_payload_validation(self):
        """Test _tuner_payload method."""
        # Create valid payload
        hyperparams = {"learning_rate": 0.01, "batch_size": 32}
        train_data = pd.DataFrame({"col": [1, 2, 3]})
        
        payload = TunerPayload(
            hyperparameters=hyperparams,
            train_data=train_data
        )
        event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=PipelineContext()
        )
        
        result = self.tuner._tuner_payload(event)
        assert isinstance(result, TunerPayload)
        assert result == payload
        assert result.hyperparameters == hyperparams
    
    def test_tuner_payload_invalid_type(self):
        """Test _tuner_payload with invalid payload type."""
        # Create event with wrong payload type
        event = Event(
            type=EventType.DATA_READY,
            payload="invalid_payload",  # Wrong type
            context=PipelineContext()
        )
        
        # This should raise a type validation error due to the @type_checker decorator
        with pytest.raises(Exception):  # The specific exception depends on the decorator implementation
            self.tuner._tuner_payload(event)
    
    def test_artifact_path_generation(self):
        """Test artifact path generation."""
        mock_run = Mock()
        mock_run.info.artifact_uri = "test://artifacts/run123"
        
        path = self.tuner.artifact_path(mock_run)
        expected_path = f"test://artifacts/run123/{TunerArtifact().hyperparameters}"
        
        assert path == expected_path
    
    @patch.object(Tuner, 'start_run')
    @patch.object(Tuner, '_handle')
    @patch.object(Tuner, 'mlflow')
    def test_run_hyperparameters_only(self, mock_mlflow, mock_handle, mock_start_run):
        """Test tuner run with hyperparameters only."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.artifact_uri = "test://artifacts"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create payload with hyperparameters only
        hyperparams = {"learning_rate": 0.001, "dropout": 0.2}
        payload = TunerPayload(hyperparameters=hyperparams)
        
        # Setup event
        context = PipelineContext()
        input_event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = self.tuner.run(input_event)
        
        # Verify
        assert result.type == EventType.TUNING_DONE
        assert result.payload.hyperparameters == hyperparams
        assert result.payload.train_data is None
        assert result.payload.validation_data is None
        assert result.payload.test_data is None
    
    @patch.object(Tuner, 'start_run')
    @patch.object(Tuner, '_handle')
    @patch.object(Tuner, 'mlflow')
    def test_run_with_all_datasets(self, mock_mlflow, mock_handle, mock_start_run):
        """Test tuner run with all datasets."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.artifact_uri = "test://artifacts"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create test data
        hyperparams = {"learning_rate": 0.01}
        train_data = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        val_data = pd.DataFrame({"feature1": [4, 5], "target": [1, 0]})
        test_data = pd.DataFrame({"feature1": [6, 7], "target": [0, 1]})
        
        payload = TunerPayload(
            hyperparameters=hyperparams,
            train_data=train_data,
            validation_data=val_data,
            test_data=test_data
        )
        
        # Setup event
        context = PipelineContext()
        input_event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = self.tuner.run(input_event)
        
        # Verify all data is preserved
        assert result.payload.hyperparameters == hyperparams
        assert result.payload.train_data.equals(train_data)
        assert result.payload.validation_data.equals(val_data)
        assert result.payload.test_data.equals(test_data)
        
        # Verify hyperparameters were logged correctly
        expected_hyperparams = payload.model_dump()
        mock_mlflow.log_dict.assert_called_once_with(
            dictionary=expected_hyperparams,
            artifact_file=TunerArtifact().hyperparameters
        )
    
    @patch.object(Tuner, 'start_run')
    @patch.object(Tuner, '_handle')
    @patch.object(Tuner, 'mlflow')
    def test_exception_handling_preserves_original_exception(self, mock_mlflow, mock_handle, mock_start_run):
        """Test that exceptions are properly chained."""
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create original exception
        original_error = ValueError("Original tuning error")
        mock_handle.side_effect = original_error
        
        # Create test event
        payload = TunerPayload(hyperparameters={"lr": 0.01})
        input_event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=PipelineContext()
        )
        
        # Execute and verify exception chaining
        with pytest.raises(TunerError) as exc_info:
            self.tuner.run(input_event)
        
        # Verify exception message and chaining
        assert "Tuner failed with error: Original tuning error" in str(exc_info.value)
        assert exc_info.value.__cause__ == original_error
    
    def test_model_dump_includes_all_fields(self):
        """Test that model_dump includes all payload fields."""
        hyperparams = {"lr": 0.01, "batch_size": 16}
        train_data = pd.DataFrame({"col": [1, 2]})
        val_data = pd.DataFrame({"col": [3, 4]})
        
        payload = TunerPayload(
            hyperparameters=hyperparams,
            train_data=train_data,
            validation_data=val_data
        )
        
        dumped = payload.model_dump()
        
        assert "hyperparameters" in dumped
        assert "train_data" in dumped
        assert "validation_data" in dumped
        assert "test_data" in dumped
        assert dumped["hyperparameters"] == hyperparams
