import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from collie.contracts.event import Event, EventType, PipelineContext
from collie.core.models import PusherPayload
from collie.core.pusher.pusher import Pusher
from collie.core.enums.ml_models import MLflowModelStage
from collie._common.exceptions import PusherError


class TestPusher:
    """Test cases for Pusher component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pusher = Pusher(
            target_stage=MLflowModelStage.PRODUCTION,
            archive_existing_versions=True,
            description="Test pusher",
            tags={"env": "test"}
        )
        self.pusher.registered_model_name = "test_model"
    
    def test_pusher_initialization(self):
        """Test Pusher initialization."""
        assert self.pusher.target_stage == MLflowModelStage.PRODUCTION
        assert self.pusher.archive_existing_versions is True
        assert self.pusher.description == "Test pusher"
        assert self.pusher.tags == {"env": "test"}
        assert self.pusher.registered_model_name == "test_model"
    
    def test_pusher_initialization_defaults(self):
        """Test Pusher initialization with defaults."""
        pusher = Pusher()
        assert pusher.target_stage is None
        assert pusher.archive_existing_versions is True
        assert pusher.description is None
        assert pusher.tags == {"component": "Pusher"}
    
    def test_registered_model_name_property(self):
        """Test registered_model_name property getter and setter."""
        pusher = Pusher()
        
        # Test getter when not set
        with pytest.raises(PusherError, match="Registered model name is not set"):
            _ = pusher.registered_model_name
        
        # Test setter and getter
        pusher.registered_model_name = "my_model"
        assert pusher.registered_model_name == "my_model"
    
    @patch.object(Pusher, 'start_run')
    @patch.object(Pusher, '_handle')
    @patch.object(Pusher, '_get_version_to_transition')
    @patch.object(Pusher, 'transition_model_version')
    def test_run_success_with_promotion(self, mock_transition, mock_get_version, mock_handle, mock_start_run):
        """Test successful pusher run with model promotion."""
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        mock_get_version.return_value = "3"
        
        # Create test payload
        payload = PusherPayload(model_uri="runs:/123/model")
        
        # Setup event with passing evaluation
        context = PipelineContext()
        context.set("pass_evaluation", True)
        input_event = Event(
            type=EventType.EVALUATION_DONE,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = self.pusher.run(input_event)
        
        # Verify
        assert result.type == EventType.PUSHER_DONE
        assert isinstance(result.payload, PusherPayload)
        assert result.context == input_event.context
        
        # Verify MLflow run was started correctly
        mock_start_run.assert_called_once_with(
            tags={"env": "test"},
            run_name="Pusher",
            log_system_metrics=False,
            nested=True,
            description="Test pusher"
        )
        
        # Verify model transition was called
        mock_get_version.assert_called_once()
        mock_transition.assert_called_once_with(
            registered_model_name="test_model",
            version="3",
            desired_stage=MLflowModelStage.PRODUCTION,
            archive_existing_versions_at_stage=True
        )
        
        # Verify handle was called
        mock_handle.assert_called_once_with(input_event)
    
    @patch.object(Pusher, 'start_run')
    @patch.object(Pusher, '_handle')
    @patch.object(Pusher, 'transition_model_version')
    def test_run_success_no_promotion(self, mock_transition, mock_handle, mock_start_run):
        """Test successful pusher run without model promotion."""
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create test payload
        payload = PusherPayload(model_uri="runs:/123/model")
        
        # Setup event with failing evaluation
        context = PipelineContext()
        context.set("pass_evaluation", False)
        input_event = Event(
            type=EventType.EVALUATION_DONE,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = self.pusher.run(input_event)
        
        # Verify
        assert result.type == EventType.PUSHER_DONE
        assert isinstance(result.payload, PusherPayload)
        
        # Verify model transition was NOT called
        mock_transition.assert_not_called()
    
    @patch.object(Pusher, 'start_run')
    @patch.object(Pusher, '_handle')
    @patch.object(Pusher, 'transition_model_version')
    def test_run_no_target_stage(self, mock_transition, mock_handle, mock_start_run):
        """Test pusher run with no target stage set."""
        # Setup pusher without target stage
        pusher = Pusher(target_stage=None)
        pusher.registered_model_name = "test_model"
        
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create test payload
        payload = PusherPayload(model_uri="runs:/123/model")
        
        # Setup event with passing evaluation
        context = PipelineContext()
        context.set("pass_evaluation", True)
        input_event = Event(
            type=EventType.EVALUATION_DONE,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = pusher.run(input_event)
        
        # Verify model transition was NOT called (no target stage)
        mock_transition.assert_not_called()
        assert result.type == EventType.PUSHER_DONE
    
    @patch.object(Pusher, 'start_run')
    @patch.object(Pusher, '_handle')
    def test_run_with_exception(self, mock_handle, mock_start_run):
        """Test pusher run with exception."""
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Make handle raise exception
        mock_handle.side_effect = Exception("Test error")
        
        # Create test event
        payload = PusherPayload(model_uri="runs:/123/model")
        input_event = Event(
            type=EventType.EVALUATION_DONE,
            payload=payload,
            context=PipelineContext()
        )
        
        # Execute and verify exception
        with pytest.raises(PusherError, match="Pusher failed with error: Test error"):
            self.pusher.run(input_event)
    
    @patch.object(Pusher, 'get_latest_version')
    def test_get_version_to_transition_success(self, mock_get_latest):
        """Test _get_version_to_transition with existing version."""
        mock_get_latest.return_value = "5"
        
        result = self.pusher._get_version_to_transition()
        
        assert result == "5"
        mock_get_latest.assert_called_once_with(
            "test_model",
            stages=[MLflowModelStage.STAGING, MLflowModelStage.PRODUCTION]
        )
    
    @patch.object(Pusher, 'get_latest_version')
    def test_get_version_to_transition_no_version(self, mock_get_latest):
        """Test _get_version_to_transition with no existing version."""
        mock_get_latest.return_value = None
        
        with pytest.raises(PusherError, match="No model versions found in stages"):
            self.pusher._get_version_to_transition()
    
    @patch.object(Pusher, 'get_latest_version')
    def test_get_version_to_transition_custom_stages(self, mock_get_latest):
        """Test _get_version_to_transition with custom stages."""
        mock_get_latest.return_value = "2"
        custom_stages = [MLflowModelStage.STAGING]
        
        result = self.pusher._get_version_to_transition(stages=custom_stages)
        
        assert result == "2"
        mock_get_latest.assert_called_once_with(
            "test_model",
            stages=custom_stages
        )
    
    def test_get_pusher_payload_validation(self):
        """Test _get_pusher_payload method."""
        payload = PusherPayload(model_uri="runs:/123/model")
        event = Event(
            type=EventType.PUSHER_DONE,
            payload=payload,
            context=PipelineContext()
        )
        
        result = self.pusher._get_pusher_payload(event)
        assert isinstance(result, PusherPayload)
        assert result == payload
        assert result.model_uri == "runs:/123/model"
    
    def test_get_pusher_payload_invalid_type(self):
        """Test _get_pusher_payload with invalid payload type."""
        event = Event(
            type=EventType.PUSHER_DONE,
            payload="invalid_payload",  # Wrong type
            context=PipelineContext()
        )
        
        # This should raise a type validation error due to the @type_checker decorator
        with pytest.raises(Exception):
            self.pusher._get_pusher_payload(event)
    
    @patch.object(Pusher, 'start_run')
    @patch.object(Pusher, '_handle')
    @patch.object(Pusher, '_get_version_to_transition')
    @patch.object(Pusher, 'transition_model_version')
    def test_run_full_workflow(self, mock_transition, mock_get_version, mock_handle, mock_start_run):
        """Test complete pusher workflow."""
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        mock_get_version.return_value = "1"
        
        # Create test payload
        payload = PusherPayload(model_uri="runs:/456/model")
        
        # Setup event context with all necessary data
        context = PipelineContext()
        context.set("pass_evaluation", True)
        context.set("model_uri", "runs:/456/model")
        context.set("evaluator_report_uri", "test://artifacts/report.json")
        
        input_event = Event(
            type=EventType.EVALUATION_DONE,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = self.pusher.run(input_event)
        
        # Verify complete workflow
        assert result.type == EventType.PUSHER_DONE
        assert result.payload.model_uri == "runs:/456/model"
        assert result.context.get("pass_evaluation") is True
        assert result.context.get("model_uri") == "runs:/456/model"
        assert result.context.get("evaluator_report_uri") == "test://artifacts/report.json"
        
        # Verify all components were called
        mock_handle.assert_called_once_with(input_event)
        mock_get_version.assert_called_once()
        mock_transition.assert_called_once_with(
            registered_model_name="test_model",
            version="1",
            desired_stage=MLflowModelStage.PRODUCTION,
            archive_existing_versions_at_stage=True
        )
    
    @patch.object(Pusher, 'start_run')
    @patch.object(Pusher, '_handle')
    def test_run_with_different_archive_setting(self, mock_handle, mock_start_run):
        """Test pusher with archive_existing_versions=False."""
        # Setup pusher with archive=False
        pusher = Pusher(
            target_stage=MLflowModelStage.STAGING,
            archive_existing_versions=False
        )
        pusher.registered_model_name = "test_model"
        
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create test event
        payload = PusherPayload(model_uri="runs:/123/model")
        context = PipelineContext()
        context.set("pass_evaluation", False)  # Won't promote anyway
        input_event = Event(
            type=EventType.EVALUATION_DONE,
            payload=payload,
            context=context
        )
        
        mock_handle.return_value = input_event
        
        # Execute
        result = pusher.run(input_event)
        
        # Verify initialization values are preserved
        assert pusher.target_stage == MLflowModelStage.STAGING
        assert pusher.archive_existing_versions is False
        assert result.type == EventType.PUSHER_DONE
