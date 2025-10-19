import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from collie.contracts.event import Event, EventType, PipelineContext
from collie.core.models import EvaluatorPayload, EvaluatorArtifact
from collie.core.evaluator.evaluator import Evaluator
from collie.core.enums.ml_models import MLflowModelStage
from collie._common.exceptions import EvaluatorError


class TestEvaluator:
    """Test cases for Evaluator component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = Evaluator(
            target_stage=MLflowModelStage.PRODUCTION,
            description="Test evaluator",
            tags={"env": "test"}
        )
        self.evaluator.registered_model_name = "test_model"
    
    def test_evaluator_initialization(self):
        """Test Evaluator initialization."""
        assert self.evaluator.target_stage == MLflowModelStage.PRODUCTION
        assert self.evaluator.description == "Test evaluator"
        assert self.evaluator.tags == {"env": "test"}
        assert self.evaluator.registered_model_name == "test_model"
    
    def test_evaluator_initialization_defaults(self):
        """Test Evaluator initialization with defaults."""
        evaluator = Evaluator()
        assert evaluator.target_stage == MLflowModelStage.STAGING
        assert evaluator.description is None
        assert evaluator.tags == {"component": "Evaluator"}
        assert evaluator.model_uri is None
        assert evaluator.metrics is None
    
    def test_registered_model_name_property(self):
        """Test registered_model_name property getter and setter."""
        evaluator = Evaluator()
        
        # Test getter when not set
        with pytest.raises(EvaluatorError, match="Registered model name is not set"):
            _ = evaluator.registered_model_name
        
        # Test setter and getter
        evaluator.registered_model_name = "my_model"
        assert evaluator.registered_model_name == "my_model"
    
    @patch.object(Evaluator, 'start_run')
    @patch.object(Evaluator, '_handle')
    @patch.object(Evaluator, '_log_metrics')
    @patch.object(Evaluator, '_log_summary')
    @patch.object(Evaluator, '_transition_model_version')
    def test_run_success(self, mock_transition, mock_log_summary, mock_log_metrics, mock_handle, mock_start_run):
        """Test successful evaluator run."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.artifact_uri = "test://artifacts"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Create test metrics
        metrics = [
            {"accuracy": 0.95},
            {"precision": 0.92},
            {"recall": 0.89}
        ]
        payload = EvaluatorPayload(
            metrics=metrics,
            is_better_than_production=True
        )
        
        # Setup event
        context = PipelineContext()
        context.set("model_uri", "runs:/123/model")
        input_event = Event(
            type=EventType.TRAINING_DONE,
            payload=payload,
            context=context
        )
        
        # Setup handle return
        mock_handle.return_value = input_event
        
        # Execute
        result = self.evaluator.run(input_event)
        
        # Verify
        assert result.type == EventType.EVALUATION_DONE
        assert isinstance(result.payload, EvaluatorPayload)
        assert result.context == input_event.context
        
        # Verify MLflow run was started correctly
        mock_start_run.assert_called_once_with(
            tags={"env": "test"},
            run_name="Evaluator",
            log_system_metrics=True,
            description="Test evaluator",
            nested=True
        )
        
        # Verify methods were called
        mock_handle.assert_called_once_with(input_event)
        mock_log_metrics.assert_called_once()
        mock_log_summary.assert_called_once_with(payload)
        mock_transition.assert_called_once_with(payload, input_event)
        
        # Verify evaluator state
        assert self.evaluator.metrics == metrics
        assert self.evaluator.model_uri == "runs:/123/model"
        
        # Verify context was updated
        expected_uri = f"test://artifacts/{EvaluatorArtifact().report}"
        assert result.context.get("evaluator_report_uri") == expected_uri
    
    @patch.object(Evaluator, 'start_run')
    @patch.object(Evaluator, '_handle')
    def test_run_with_exception(self, mock_handle, mock_start_run):
        """Test evaluator run with exception."""
        # Setup mocks
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Make handle raise exception
        mock_handle.side_effect = Exception("Test error")
        
        # Create test event
        payload = EvaluatorPayload(metrics=[{"accuracy": 0.8}], is_better_than_production=False)
        input_event = Event(
            type=EventType.TRAINING_DONE,
            payload=payload,
            context=PipelineContext()
        )
        
        # Execute and verify exception
        with pytest.raises(EvaluatorError, match="Evaluator failed: Test error"):
            self.evaluator.run(input_event)
    
    def test_experiment_is_better_static_method(self):
        """Test experiment_is_better static method."""
        # Test when experiment is better
        payload_better = EvaluatorPayload(
            metrics=[{"accuracy": 0.95}],
            is_better_than_production=True
        )
        assert Evaluator.experiment_is_better(payload_better) is True
        
        # Test when experiment is not better
        payload_worse = EvaluatorPayload(
            metrics=[{"accuracy": 0.85}],
            is_better_than_production=False
        )
        assert Evaluator.experiment_is_better(payload_worse) is False
    
    @patch.object(Evaluator, 'register_model')
    @patch.object(Evaluator, 'transition_model_version')
    @patch.object(Evaluator, '_next_model_version')
    @patch.object(Evaluator, 'mlflow')
    def test_transition_model_version_better_experiment(self, mock_mlflow, mock_next_version, mock_transition, mock_register):
        """Test model transition when experiment is better."""
        # Setup mocks
        mock_next_version.return_value = "2"
        
        # Create payload indicating better performance
        payload = EvaluatorPayload(
            metrics=[{"accuracy": 0.95}],
            is_better_than_production=True
        )
        
        # Setup event and evaluator state
        context = PipelineContext()
        event = Event(type=EventType.EVALUATION_DONE, payload=payload, context=context)
        self.evaluator.model_uri = "runs:/123/model"
        
        # Execute
        self.evaluator._transition_model_version(payload, event)
        
        # Verify model registration and transition
        mock_register.assert_called_once_with(
            model_name="test_model",
            model_uri="runs:/123/model"
        )
        mock_transition.assert_called_once_with(
            registered_model_name="test_model",
            version="2",
            desired_stage=MLflowModelStage.PRODUCTION,
            archive_existing_versions_at_stage=True
        )
        
        # Verify context and logging
        assert event.context.get("pass_evaluation") is True
        mock_mlflow.log_param.assert_called_with("promotion_reason", "experiment_is_better")
    
    @patch.object(Evaluator, 'mlflow')
    def test_transition_model_version_worse_experiment(self, mock_mlflow):
        """Test model transition when experiment is worse."""
        # Create payload indicating worse performance
        payload = EvaluatorPayload(
            metrics=[{"accuracy": 0.75}],
            is_better_than_production=False
        )
        
        # Setup event
        context = PipelineContext()
        event = Event(type=EventType.EVALUATION_DONE, payload=payload, context=context)
        
        # Execute
        self.evaluator._transition_model_version(payload, event)
        
        # Verify context and logging
        assert event.context.get("pass_evaluation") is False
        mock_mlflow.log_param.assert_called_with("promotion_reason", "experiment_not_better")
    
    @patch.object(Evaluator, 'get_latest_version')
    def test_next_model_version_with_existing_versions(self, mock_get_latest):
        """Test _next_model_version with existing versions."""
        mock_get_latest.return_value = "5"
        
        result = self.evaluator._next_model_version()
        
        assert result == "6"
        mock_get_latest.assert_called_once_with(
            "test_model",
            stages=[MLflowModelStage.STAGING, MLflowModelStage.PRODUCTION]
        )
    
    @patch.object(Evaluator, 'get_latest_version')
    def test_next_model_version_no_existing_versions(self, mock_get_latest):
        """Test _next_model_version with no existing versions."""
        mock_get_latest.return_value = None
        
        result = self.evaluator._next_model_version()
        
        assert result == "1"
    
    def test_flatten_metrics(self):
        """Test _flatten_metrics method."""
        metrics = [
            {"accuracy": 0.95, "precision": 0.92},
            {"recall": 0.89, "f1_score": 0.90},
            {"auc": 0.87}
        ]
        
        result = self.evaluator._flatten_metrics(metrics)
        
        expected = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.90,
            "auc": 0.87
        }
        
        assert result == expected
    
    @patch.object(Evaluator, 'mlflow')
    def test_log_summary(self, mock_mlflow):
        """Test _log_summary method."""
        metrics = [
            {"accuracy": 0.95, "precision": 0.92},
            {"recall": 0.89}
        ]
        payload = EvaluatorPayload(
            metrics=metrics,
            is_better_than_production=True
        )
        
        self.evaluator._log_summary(payload)
        
        expected_summary = {
            "total_metrics": 3,
            "is_better": True,
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.89
            }
        }
        
        mock_mlflow.log_dict.assert_called_once_with(
            dictionary=expected_summary,
            artifact_file="report.json"
        )
    
    @patch.object(Evaluator, 'mlflow')
    def test_log_metrics(self, mock_mlflow):
        """Test _log_metrics method."""
        metrics = [
            {"accuracy": 0.95, "precision": 0.92},
            {"recall": 0.89}
        ]
        self.evaluator.metrics = metrics
        
        self.evaluator._log_metrics()
        
        # Verify all metrics were logged individually
        expected_calls = [
            (("accuracy", 0.95),),
            (("precision", 0.92),),
            (("recall", 0.89),)
        ]
        
        assert mock_mlflow.log_metric.call_count == 3
        actual_calls = [call[0] for call in mock_mlflow.log_metric.call_args_list]
        assert set(actual_calls) == set(expected_calls)
    
    def test_get_evaluator_payload_validation(self):
        """Test _get_evaluator_payload method."""
        payload = EvaluatorPayload(
            metrics=[{"accuracy": 0.95}],
            is_better_than_production=True
        )
        event = Event(
            type=EventType.EVALUATION_DONE,
            payload=payload,
            context=PipelineContext()
        )
        
        result = self.evaluator._get_evaluator_payload(event)
        assert isinstance(result, EvaluatorPayload)
        assert result == payload
    
    def test_get_evaluator_payload_invalid_type(self):
        """Test _get_evaluator_payload with invalid payload type."""
        event = Event(
            type=EventType.EVALUATION_DONE,
            payload="invalid_payload",  # Wrong type
            context=PipelineContext()
        )
        
        # This should raise a type validation error due to the @type_checker decorator
        with pytest.raises(Exception):
            self.evaluator._get_evaluator_payload(event)
    
    def test_artifact_uri_generation(self):
        """Test artifact URI generation."""
        mock_run = Mock()
        mock_run.info.artifact_uri = "test://artifacts/run123"
        
        uri = self.evaluator.artifact_uri(mock_run)
        expected_uri = f"test://artifacts/run123/{EvaluatorArtifact().report}"
        
        assert uri == expected_uri
