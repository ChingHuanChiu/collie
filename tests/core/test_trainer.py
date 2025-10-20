import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from collie.contracts.event import Event, EventType, PipelineContext
from collie.core.models import TrainerPayload, TrainerArtifact
from collie.core.trainer.trainer import Trainer
from collie._common.exceptions import TrainerError


class TestTrainer:
    """Test cases for Trainer component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = Trainer(
            description="Test trainer",
            tags={"env": "test"}
        )
    
    def test_trainer_initialization(self):
        """Test Trainer initialization."""
        assert self.trainer.description == "Test trainer"
        assert self.trainer.tags == {"env": "test"}
    
    def test_trainer_initialization_defaults(self):
        """Test Trainer initialization with defaults."""
        trainer = Trainer()
        assert trainer.description is None
        assert trainer.tags == {"component": "Trainer"}
    
    @patch.object(Trainer, 'start_run')
    @patch.object(Trainer, '_handle')
    @patch.object(Trainer, 'log_model')
    def test_run_success(self, mock_log_model, mock_handle, mock_start_run):
        """Test successful trainer run."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create test model
        mock_model = Mock()
        payload = TrainerPayload(
            model=mock_model,
            train_loss=0.5,
            val_loss=0.3
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
        result = self.trainer.run(input_event)
        
        # Verify
        assert result.type == EventType.TRAINING_DONE
        assert isinstance(result.payload, TrainerPayload)
        assert result.context == input_event.context
        
        # Verify MLflow run was started correctly
        mock_start_run.assert_called_once_with(
            run_name="Trainer",
            tags={"env": "test"},
            log_system_metrics=True,
            nested=True,
            description="Test trainer"
        )
        
        # Verify model was logged
        mock_log_model.assert_called_once_with(
            model=mock_model,
            name=TrainerArtifact().model
        )
        
        # Verify handle was called
        mock_handle.assert_called_once_with(input_event)
        
        # Verify model URI was set in context
        expected_uri = f"runs:/test-run-id/{TrainerArtifact().model}"
        assert result.context.get("model_uri") == expected_uri
    
    @patch.object(Trainer, 'start_run')
    @patch.object(Trainer, '_handle')
    def test_run_with_exception(self, mock_handle, mock_start_run):
        """Test trainer run with exception."""
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Make handle raise exception
        mock_handle.side_effect = Exception("Test error")
        
        # Create test event
        payload = TrainerPayload()
        input_event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=PipelineContext()
        )
        
        # Execute and verify exception
        with pytest.raises(TrainerError, match="Trainer failed with error: Test error"):
            self.trainer.run(input_event)
    
    def test_trainer_payload_validation(self):
        """Test _trainer_payload method."""
        # Create valid payload
        mock_model = Mock()
        payload = TrainerPayload(
            model=mock_model,
            train_loss=0.2
        )
        event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=PipelineContext()
        )
        
        result = self.trainer._trainer_payload(event)
        assert isinstance(result, TrainerPayload)
        assert result == payload
        assert result.model == mock_model
        assert result.train_loss == 0.2
    
    def test_trainer_payload_invalid_type(self):
        """Test _trainer_payload with invalid payload type."""
        # Create event with wrong payload type
        event = Event(
            type=EventType.DATA_READY,
            payload="invalid_payload",  # Wrong type
            context=PipelineContext()
        )
        
        # This should raise a type validation error due to the @type_checker decorator
        with pytest.raises(Exception):  # The specific exception depends on the decorator implementation
            self.trainer._trainer_payload(event)
    
    def test_artifact_uri_generation(self):
        """Test artifact URI generation."""
        mock_run = Mock()
        mock_run.info.run_id = "test-run-id-123"
        
        uri = self.trainer.artifact_uri(mock_run)
        expected_uri = f"runs:/test-run-id-123/{TrainerArtifact().model}"
        
        assert uri == expected_uri
    
    @patch.object(Trainer, 'start_run')
    @patch.object(Trainer, '_handle')
    @patch.object(Trainer, 'log_model')
    def test_run_minimal_payload(self, mock_log_model, mock_handle, mock_start_run):
        """Test trainer run with minimal payload."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create minimal payload
        payload = TrainerPayload()  # All None values
        
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
        result = self.trainer.run(input_event)
        
        # Verify
        assert result.type == EventType.TRAINING_DONE
        assert result.payload.model is None
        assert result.payload.train_loss is None
        assert result.payload.val_loss is None
        
        # Verify model was logged (even if None)
        mock_log_model.assert_called_once_with(
            model=None,
            name=TrainerArtifact().model
        )
    
    @patch.object(Trainer, 'start_run')
    @patch.object(Trainer, '_handle')
    @patch.object(Trainer, 'log_model')
    def test_run_with_loss_metrics(self, mock_log_model, mock_handle, mock_start_run):
        """Test trainer run with loss metrics."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create payload with losses
        mock_model = Mock()
        payload = TrainerPayload(
            model=mock_model,
            train_loss=0.8,
            val_loss=0.6
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
        result = self.trainer.run(input_event)
        
        # Verify payload values are preserved
        assert result.payload.train_loss == 0.8
        assert result.payload.val_loss == 0.6
        assert result.payload.model == mock_model
    
    @patch.object(Trainer, 'start_run')
    @patch.object(Trainer, '_handle')
    @patch.object(Trainer, 'log_model')
    def test_exception_handling_preserves_original_exception(self, mock_log_model, mock_handle, mock_start_run):
        """Test that exceptions are properly chained."""
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create original exception
        original_error = ValueError("Original error message")
        mock_handle.side_effect = original_error
        
        # Create test event
        payload = TrainerPayload()
        input_event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=PipelineContext()
        )
        
        # Execute and verify exception chaining
        with pytest.raises(TrainerError) as exc_info:
            self.trainer.run(input_event)
        
        # Verify exception message and chaining
        assert "Trainer failed with error: Original error message" in str(exc_info.value)
        assert exc_info.value.__cause__ == original_error
