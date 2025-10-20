import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from typing import Any

from collie.contracts.event import Event, EventType, PipelineContext, EventHandler
from collie.core.models import TransformerPayload, TransformerArtifact
from collie.core.transform.transform import Transformer
from collie._common.exceptions import TransformerError


class TestTransformer:
    """Test cases for Transformer component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = Transformer(
            description="Test transformer",
            tags={"env": "test"}
        )
    
    def test_transformer_initialization(self):
        """Test Transformer initialization."""
        assert self.transformer.description == "Test transformer"
        assert self.transformer.tags == {"env": "test"}
    
    def test_transformer_initialization_defaults(self):
        """Test Transformer initialization with defaults."""
        transformer = Transformer()
        assert transformer.description is None
        assert transformer.tags == {"component": "Transformer"}
    
    @patch.object(Transformer, 'start_run')
    @patch.object(Transformer, '_handle')
    @patch.object(Transformer, 'log_pd_data')
    def test_run_success(self, mock_log_pd_data, mock_handle, mock_start_run):
        """Test successful transformer run."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.artifact_uri = "test://artifacts"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create test data
        train_data = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        payload = TransformerPayload(train_data=train_data)
        
        # Setup event
        context = PipelineContext()
        input_event = Event(
            type=EventType.INITIALIZE,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = self.transformer.run(input_event)
        
        # Verify
        assert result.type == EventType.DATA_READY
        assert isinstance(result.payload, TransformerPayload)
        assert result.context == input_event.context
        
        # Verify MLflow run was started correctly
        mock_start_run.assert_called_once_with(
            tags={"env": "test"},
            run_name="Transformer",
            log_system_metrics=True,
            nested=True,
            description="Test transformer"
        )
        
        # Verify handle was called
        mock_handle.assert_called_once_with(input_event)
    
    @patch.object(Transformer, 'start_run')
    @patch.object(Transformer, '_handle')
    def test_run_with_exception(self, mock_handle, mock_start_run):
        """Test transformer run with exception."""
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Make handle raise exception
        mock_handle.side_effect = Exception("Test error")
        
        # Create test event
        payload = TransformerPayload()
        input_event = Event(
            type=EventType.INITIALIZE,
            payload=payload,
            context=PipelineContext()
        )
        
        # Execute and verify exception
        with pytest.raises(TransformerError, match="Transformer failed with error: Test error"):
            self.transformer.run(input_event)
    
    def test_transformer_payload_validation(self):
        """Test _transformer_payload method."""
        # Create valid payload
        payload = TransformerPayload(
            train_data=pd.DataFrame({"col": [1, 2, 3]})
        )
        event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=PipelineContext()
        )
        
        result = self.transformer._transformer_payload(event)
        assert isinstance(result, TransformerPayload)
        assert result == payload
    
    def test_transformer_payload_invalid_type(self):
        """Test _transformer_payload with invalid payload type."""
        # Create event with wrong payload type
        event = Event(
            type=EventType.DATA_READY,
            payload="invalid_payload",  # Wrong type
            context=PipelineContext()
        )
        
        # This should raise a type validation error due to the @type_checker decorator
        with pytest.raises(Exception):  # The specific exception depends on the decorator implementation
            self.transformer._transformer_payload(event)
    
    def test_artifact_uri_generation(self):
        """Test artifact URI generation."""
        mock_run = Mock()
        mock_run.info.artifact_uri = "test://artifacts"
        
        # Test different data types
        train_uri = self.transformer.artifact_uri(mock_run, "train_data")
        val_uri = self.transformer.artifact_uri(mock_run, "validation_data")
        test_uri = self.transformer.artifact_uri(mock_run, "test_data")
        
        artifact = TransformerArtifact()
        expected_train = f"test://artifacts/{artifact.train_data}"
        expected_val = f"test://artifacts/{artifact.validation_data}"
        expected_test = f"test://artifacts/{artifact.test_data}"
        
        assert train_uri == expected_train
        assert val_uri == expected_val
        assert test_uri == expected_test
    
    @patch.object(Transformer, 'start_run')
    @patch.object(Transformer, '_handle')
    @patch.object(Transformer, 'log_pd_data')
    def test_run_with_multiple_datasets(self, mock_log_pd_data, mock_handle, mock_start_run):
        """Test transformer run with all datasets."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.artifact_uri = "test://artifacts"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create test data
        train_data = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        val_data = pd.DataFrame({"feature1": [4, 5], "target": [1, 0]})
        test_data = pd.DataFrame({"feature1": [6, 7], "target": [0, 1]})
        
        payload = TransformerPayload(
            train_data=train_data,
            validation_data=val_data,
            test_data=test_data
        )
        
        # Setup event
        context = PipelineContext()
        input_event = Event(
            type=EventType.INITIALIZE,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = self.transformer.run(input_event)
        
        # Verify all datasets were logged
        assert mock_log_pd_data.call_count == 3
        
        # Verify context was updated with URIs
        artifact = TransformerArtifact()
        expected_train_uri = f"test://artifacts/{artifact.train_data}"
        expected_val_uri = f"test://artifacts/{artifact.validation_data}"
        expected_test_uri = f"test://artifacts/{artifact.test_data}"
        
        assert result.context.get("train_data_uri") == expected_train_uri
        assert result.context.get("validation_data_uri") == expected_val_uri
        assert result.context.get("test_data_uri") == expected_test_uri
    
    @patch.object(Transformer, 'start_run')
    @patch.object(Transformer, '_handle')
    @patch.object(Transformer, 'log_pd_data')
    def test_run_with_none_datasets(self, mock_log_pd_data, mock_handle, mock_start_run):
        """Test transformer run with None datasets."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.artifact_uri = "test://artifacts"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create payload with None values
        payload = TransformerPayload()
        
        # Setup event
        context = PipelineContext()
        input_event = Event(
            type=EventType.INITIALIZE,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = self.transformer.run(input_event)
        
        # Verify no datasets were logged (all were None)
        mock_log_pd_data.assert_not_called()
        
        # Verify no URIs were set in context
        assert result.context.get("train_data_uri") is None
        assert result.context.get("validation_data_uri") is None
        assert result.context.get("test_data_uri") is None
