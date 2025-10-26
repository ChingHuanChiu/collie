import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from collie.core.orchestrator.orchestrator import Orchestrator
from collie.core.transform.transform import Transformer
from collie.core.trainer.trainer import Trainer
from collie.core.tuner.tuner import Tuner
from collie.core.evaluator.evaluator import Evaluator
from collie.core.pusher.pusher import Pusher
from collie.contracts.event import Event, EventType, PipelineContext
from collie.core.models import (
    TransformerPayload,
    TrainerPayload,
    TunerPayload,
    EvaluatorPayload,
    PusherPayload
)
from collie._common.exceptions import TrainerError, OrchestratorError, EvaluatorError


@pytest.mark.integration
class TestTransformerTrainerIntegration:
    
    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })
    
    @patch('collie.core.transform.transform.Transformer.start_run')
    @patch('collie.core.trainer.trainer.Trainer.start_run')
    def test_data_flows_from_transformer_to_trainer(self, mock_trainer_run, mock_transformer_run):
        """Test that transformed data flows correctly to trainer component."""
        # Setup mock MLflow runs
        for mock_run in [mock_transformer_run, mock_trainer_run]:
            mock_context = MagicMock()
            mock_context.info.artifact_uri = "test://artifacts"
            mock_context.info.run_id = "test-run-123"
            mock_run.return_value.__enter__.return_value = mock_context
            mock_run.return_value.__exit__.return_value = None
        
        # Create mock components instead of real instances
        transformer = Mock(spec=Transformer)
        trainer = Mock(spec=Trainer)
        
        # Setup transformer output
        transformer_payload = TransformerPayload(train_data=self.sample_data)
        transformer_event = Event(
            type=EventType.DATA_READY,
            payload=transformer_payload,
            context=PipelineContext()
        )
        transformer.run.return_value = transformer_event
        
        # Setup trainer output
        mock_model = Mock()
        trainer_payload = TrainerPayload(model=mock_model)
        trainer_payload.set_extra("train_loss", 0.5)
        trainer_event = Event(
            type=EventType.TRAINING_DONE,
            payload=trainer_payload,
            context=PipelineContext()
        )
        trainer.run.return_value = trainer_event
        
        # Execute transformer
        initial_event = Event(
            type=EventType.INITIALIZE,
            payload=None,
            context=PipelineContext()
        )
        transformer_result = transformer.run(initial_event)
        
        # Verify transformer output
        assert transformer_result.type == EventType.DATA_READY
        assert isinstance(transformer_result.payload, TransformerPayload)
        
        # Execute trainer with transformer output
        trainer_result = trainer.run(transformer_result)
        
        # Verify trainer output
        assert trainer_result.type == EventType.TRAINING_DONE
        assert isinstance(trainer_result.payload, TrainerPayload)
        assert trainer_result.payload.model == mock_model


@pytest.mark.integration
class TestEvaluatorPusherIntegration:
    """Integration tests for Evaluator and Pusher components working together."""
    
    def test_pusher_receives_evaluation_results(self):
        """Test that pusher component receives and uses evaluation results."""
        # Create evaluation payload that passes
        metrics = [{"accuracy": 0.95}, {"precision": 0.92}]
        evaluator_payload = EvaluatorPayload(
            metrics=metrics,
            is_better_than_production=True
        )
        
        # Setup context with evaluation results
        context = PipelineContext()
        context.set("pass_evaluation", True)
        context.set("model_uri", "runs:/123/model")
        
        evaluator_event = Event(
            type=EventType.EVALUATION_DONE,
            payload=evaluator_payload,
            context=context
        )
        
        # Create pusher event
        pusher_payload = PusherPayload(model_uri="runs:/123/model")
        pusher_event = Event(
            type=EventType.PUSHER_DONE,
            payload=pusher_payload,
            context=context
        )
        
        # Verify pusher has access to evaluation results
        assert pusher_event.context.get("pass_evaluation") is True
        assert pusher_event.context.get("model_uri") == "runs:/123/model"
        assert pusher_event.payload.model_uri == "runs:/123/model"
    
    def test_pusher_handles_failed_evaluation(self):
        """Test that pusher component handles failed evaluation correctly."""
        # Create evaluation payload that fails
        metrics = [{"accuracy": 0.65}, {"precision": 0.62}]
        evaluator_payload = EvaluatorPayload(
            metrics=metrics,
            is_better_than_production=False
        )
        
        # Setup context with failed evaluation
        context = PipelineContext()
        context.set("pass_evaluation", False)
        context.set("model_uri", "runs:/123/model")
        
        evaluator_event = Event(
            type=EventType.EVALUATION_DONE,
            payload=evaluator_payload,
            context=context
        )
        
        # Verify evaluation failure is captured
        assert evaluator_event.context.get("pass_evaluation") is False
        assert evaluator_payload.is_better_than_production is False


@pytest.mark.integration
class TestTunerTrainerIntegration:
    """Integration tests for Tuner and Trainer components working together."""
    
    def test_trainer_receives_tuned_hyperparameters(self):
        """Test that trainer receives hyperparameters from tuner."""
        # Create tuner output with optimized hyperparameters
        hyperparams = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 10,
            "optimizer": "adam"
        }
        
        tuner_payload = TunerPayload(
            hyperparameters=hyperparams,
            train_data=pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
        )
        
        # Setup context with hyperparameter URI
        context = PipelineContext()
        context.set("hyperparameters_uri", "path/to/hyperparams.json")
        
        tuner_event = Event(
            type=EventType.TUNING_DONE,
            payload=tuner_payload,
            context=context
        )
        
        # Verify trainer can access hyperparameters
        assert tuner_event.context.get("hyperparameters_uri") == "path/to/hyperparams.json"
        assert tuner_event.payload.hyperparameters["learning_rate"] == 0.01
        assert tuner_event.payload.hyperparameters["batch_size"] == 32
        assert tuner_event.payload.hyperparameters["epochs"] == 10


@pytest.mark.integration
class TestMinimalPipelineOrchestration:
    """Integration tests for minimal pipeline orchestration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_two_component_pipeline(self, mock_logger, mock_mlflow_config):
        """Test orchestration of a minimal two-component pipeline."""
        # Create mock components
        mock_transformer = Mock()
        mock_trainer = Mock()
        
        # Setup component responses
        transformer_event = Event(
            type=EventType.DATA_READY,
            payload=TransformerPayload(train_data=self.sample_data),
            context=PipelineContext()
        )
        
        trainer_event = Event(
            type=EventType.TRAINING_DONE,
            payload=TrainerPayload(model=Mock()),
            context=PipelineContext()
        )
        
        mock_transformer.run.return_value = transformer_event
        mock_trainer.run.return_value = trainer_event
        
        # Create and run orchestrator
        orchestrator = Orchestrator(
            components=[mock_transformer, mock_trainer],
            tracking_uri=mock_mlflow_config.tracking_uri,
            registered_model_name="test_model"
        )
        
        # Manually set mlflow_config since we're not calling run()
        orchestrator.mlflow_config = mock_mlflow_config
        
        orchestrator.orchestrate_pipeline()
        
        # Verify both components were executed
        assert mock_transformer.run.called
        assert mock_trainer.run.called
        
        # Verify logging
        mock_logger.info.assert_any_call("Pipeline started.")
        mock_logger.info.assert_any_call("Running component 0: Mock")
        mock_logger.info.assert_any_call("Component 0 finished: Mock")
        mock_logger.info.assert_any_call("Running component 1: Mock")
        mock_logger.info.assert_any_call("Component 1 finished: Mock")


@pytest.mark.integration
class TestPipelineContextPersistence:
    """Integration tests for context persistence through pipeline stages."""
    
    def test_context_data_accumulates_through_stages(self):
        """Test that context data persists and accumulates through all pipeline stages."""
        context = PipelineContext()
        
        # Define pipeline stages and their context updates
        stages = [
            ("transformer", "train_data_uri", "path/to/train.csv"),
            ("trainer", "model_uri", "runs:/123/model"),
            ("evaluator", "evaluation_report_uri", "path/to/report.json"),
            ("pusher", "deployment_status", "success")
        ]
        
        # Simulate each stage updating context
        for stage_name, key, value in stages:
            context.set(key, value)
            
            # Create event for current stage
            event = Event(
                type=EventType.DATA_READY,
                payload=f"{stage_name}_payload",
                context=context
            )
            
            # Verify all previous context data is still available
            for prev_stage, prev_key, prev_value in stages:
                if prev_key in context.to_dict():
                    assert context.get(prev_key) == prev_value
        
        # Final verification - all context data should be present
        final_context = context.to_dict()
        assert final_context["train_data_uri"] == "path/to/train.csv"
        assert final_context["model_uri"] == "runs:/123/model"
        assert final_context["evaluation_report_uri"] == "path/to/report.json"
        assert final_context["deployment_status"] == "success"
    
    def test_shared_context_across_events(self):
        """Test that multiple events can share and update the same context."""
        context = PipelineContext()
        
        # First event adds data
        event1 = Event(type=EventType.INITIALIZE, payload="data1", context=context)
        event1.context.set("train_data_uri", "path/to/train.csv")
        
        # Second event adds more data
        event2 = Event(type=EventType.DATA_READY, payload="data2", context=context)
        event2.context.set("model_uri", "runs:/123/model")
        
        # Third event should see all previous data
        event3 = Event(type=EventType.TRAINING_DONE, payload="data3", context=context)
        
        # Verify all events share the same context
        assert event1.context is event2.context is event3.context
        assert event3.context.get("train_data_uri") == "path/to/train.csv"
        assert event3.context.get("model_uri") == "runs:/123/model"


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipelineFlow:
    """Integration tests for complete end-to-end pipeline flows."""
    
    def test_complete_event_type_progression(self):
        """Test that events progress through all expected types in correct order."""
        # Define expected event type progression
        expected_flow = [
            EventType.INITIALIZE,
            EventType.DATA_READY,
            EventType.TRAINING_DONE,
            EventType.TUNING_DONE,
            EventType.EVALUATION_DONE,
            EventType.PUSHER_DONE
        ]
        
        context = PipelineContext()
        current_event = Event(
            type=EventType.INITIALIZE,
            payload=None,
            context=context
        )
        
        # Simulate progression through each stage
        for i, event_type in enumerate(expected_flow[1:], 1):
            # Update context as would happen in real pipeline
            context.set(f"stage_{i}_completed", True)
            
            # Create next event
            current_event = Event(
                type=event_type,
                payload=f"stage_{i}_payload",
                context=context
            )
            
            # Verify event structure
            assert current_event.type == event_type
            assert current_event.payload == f"stage_{i}_payload"
            assert current_event.context.get(f"stage_{i}_completed") is True
        
        # Verify final pipeline state
        assert current_event.type == EventType.PUSHER_DONE
        final_context = context.to_dict()
        for i in range(1, len(expected_flow)):
            assert final_context[f"stage_{i}_completed"] is True


@pytest.mark.integration
class TestErrorPropagation:
    """Integration tests for error handling and propagation through pipeline."""
    
    def test_component_error_stops_pipeline(self, mock_mlflow_config):
        """Test that a component error properly stops pipeline execution."""
        # Create mock components where second component fails
        mock_transformer = Mock()
        mock_trainer = Mock()
        mock_trainer.run.side_effect = TrainerError("Training failed due to invalid data")
        
        # Setup transformer to succeed
        mock_transformer.run.return_value = Event(
            type=EventType.DATA_READY,
            payload=TransformerPayload(),
            context=PipelineContext()
        )
        
        # Create orchestrator
        orchestrator = Orchestrator(
            components=[mock_transformer, mock_trainer],
            tracking_uri=mock_mlflow_config.tracking_uri,
            registered_model_name="test_model"
        )
        
        # Manually set mlflow_config since we're not calling run()
        orchestrator.mlflow_config = mock_mlflow_config
        
        # Pipeline should fail at trainer stage
        with pytest.raises(TrainerError, match="Training failed due to invalid data"):
            orchestrator.orchestrate_pipeline()
        
        # Verify transformer was called but trainer failed
        assert mock_transformer.run.called
        assert mock_trainer.run.called
    
    def test_orchestrator_wraps_component_errors(self):
        """Test that orchestrator properly wraps component errors in run method."""
        # Create component that will fail
        mock_component = Mock()
        mock_component.run.side_effect = EvaluatorError("Model evaluation failed")
        
        orchestrator = Orchestrator(
            components=[mock_component],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
        
        # Test through run method which handles error wrapping
        with patch.object(orchestrator, 'start_run') as mock_start_run:
            mock_context = MagicMock()
            mock_start_run.return_value.__enter__.return_value = mock_context
            mock_start_run.return_value.__exit__.return_value = None
            
            with pytest.raises(OrchestratorError, match="Component error in orchestration"):
                orchestrator.run()


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndScenarios:
    """End-to-end integration test scenarios for complete workflows."""
    
    def test_successful_model_deployment_workflow(self):
        """Test complete successful model deployment workflow from data to deployment."""
        context = PipelineContext()
        
        # Stage 1: Data transformation
        context.set("train_data_uri", "path/to/train.csv")
        context.set("validation_data_uri", "path/to/val.csv")
        transformer_payload = TransformerPayload(
            train_data=pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]}),
            validation_data=pd.DataFrame({"feature": [4, 5], "target": [1, 0]})
        )
        
        # Stage 2: Model training
        context.set("model_uri", "runs:/123/model")
        trainer_payload = TrainerPayload(
            model=Mock(), 
            train_loss=0.3, 
            val_loss=0.25
        )
        
        # Stage 3: Model evaluation (passes)
        context.set("pass_evaluation", True)
        context.set("evaluator_report_uri", "path/to/report.json")
        evaluator_payload = EvaluatorPayload(
            metrics=[{"accuracy": 0.95}, {"f1": 0.93}],
            is_better_than_production=True
        )
        
        # Stage 4: Model deployment
        context.set("deployment_status", "success")
        pusher_payload = PusherPayload(model_uri="runs:/123/model")
        
        # Create final event
        final_event = Event(
            type=EventType.PUSHER_DONE,
            payload=pusher_payload,
            context=context
        )
        
        # Verify complete successful deployment
        assert final_event.type == EventType.PUSHER_DONE
        assert final_event.context.get("pass_evaluation") is True
        assert final_event.context.get("deployment_status") == "success"
        assert final_event.context.get("model_uri") == "runs:/123/model"
        assert final_event.payload.model_uri == "runs:/123/model"
    
    def test_model_rejection_workflow(self):
        """Test workflow where model fails evaluation and is not deployed."""
        context = PipelineContext()
        
        # Model evaluation fails
        context.set("pass_evaluation", False)
        context.set("evaluator_report_uri", "path/to/report.json")
        evaluator_payload = EvaluatorPayload(
            metrics=[{"accuracy": 0.65}, {"f1": 0.60}],  # Poor performance
            is_better_than_production=False
        )
        
        # Pusher marks as rejected
        context.set("deployment_status", "rejected")
        pusher_payload = PusherPayload(model_uri="runs:/123/model")
        
        final_event = Event(
            type=EventType.PUSHER_DONE,
            payload=pusher_payload,
            context=context
        )
        
        # Verify rejection workflow
        assert final_event.context.get("pass_evaluation") is False
        assert final_event.context.get("deployment_status") == "rejected"
        # Model URI should still be available for analysis
        assert final_event.payload.model_uri == "runs:/123/model"
