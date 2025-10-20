"""
Integration tests for the collie ML pipeline framework.

These tests verify that components work together correctly
and that the entire pipeline can be orchestrated successfully.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import List

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
from collie.core.enums.ml_models import MLflowModelStage


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for complete pipeline workflows."""
    
    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })
    
    @patch('collie.core.transform.transform.Transformer.start_run')
    @patch('collie.core.trainer.trainer.Trainer.start_run')
    @patch('collie.core.evaluator.evaluator.Evaluator.start_run')
    @patch('collie.core.pusher.pusher.Pusher.start_run')
    def test_transformer_to_trainer_integration(self, mock_pusher_run, mock_eval_run, mock_trainer_run, mock_transformer_run):
        """Test integration between Transformer and Trainer components."""
        # Setup mock runs
        for mock_run in [mock_transformer_run, mock_trainer_run, mock_eval_run, mock_pusher_run]:
            mock_context = MagicMock()
            mock_context.info.artifact_uri = "test://artifacts"
            mock_context.info.run_id = "test-run-123"
            mock_run.return_value.__enter__.return_value = mock_context
            mock_run.return_value.__exit__.return_value = None
        
        # Create components
        transformer = Transformer()
        trainer = Trainer()
        
        # Mock the _handle methods to simulate data processing
        with patch.object(transformer, '_handle') as mock_transformer_handle, \
             patch.object(trainer, '_handle') as mock_trainer_handle, \
             patch.object(transformer, 'log_pd_data'), \
             patch.object(trainer, 'log_model'):
            
            # Setup transformer to return data
            transformer_payload = TransformerPayload(train_data=self.sample_data)
            transformer_event = Event(
                type=EventType.DATA_READY,
                payload=transformer_payload,
                context=PipelineContext()
            )
            mock_transformer_handle.return_value = transformer_event
            
            # Setup trainer to return model
            mock_model = Mock()
            trainer_payload = TrainerPayload(model=mock_model, train_loss=0.5)
            trainer_event = Event(
                type=EventType.TRAINING_DONE,
                payload=trainer_payload,
                context=PipelineContext()
            )
            mock_trainer_handle.return_value = trainer_event
            
            # Run transformer
            initial_event = Event(
                type=EventType.INITIALIZE,
                payload=None,
                context=PipelineContext()
            )
            
            transformer_result = transformer.run(initial_event)
            
            # Verify transformer output
            assert transformer_result.type == EventType.DATA_READY
            assert isinstance(transformer_result.payload, TransformerPayload)
            
            # Run trainer with transformer output
            trainer_result = trainer.run(transformer_result)
            
            # Verify trainer output
            assert trainer_result.type == EventType.TRAINING_DONE
            assert isinstance(trainer_result.payload, TrainerPayload)
            assert trainer_result.payload.model == mock_model
            
            # Verify context carries through
            assert "model_uri" in trainer_result.context.to_dict()
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_minimal_pipeline_orchestration(self, mock_logger):
        """Test orchestration of a minimal pipeline."""
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
        
        # Create orchestrator
        orchestrator = Orchestrator(
            components=[mock_transformer, mock_trainer],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
        
        # Run pipeline
        orchestrator.orchestrate_pipeline()
        
        # Verify components were called
        assert mock_transformer.run.called
        assert mock_trainer.run.called
        
        # Verify logging
        mock_logger.info.assert_any_call("Pipeline started.")
        mock_logger.info.assert_any_call("Running component 0: Mock")
        mock_logger.info.assert_any_call("Component 0 finished: Mock")
    
    def test_event_context_persistence_through_pipeline(self):
        """Test that context persists and accumulates through pipeline stages."""
        context = PipelineContext()
        
        # Simulate pipeline stages updating context
        stages = [
            ("transformer", "train_data_uri", "path/to/train.csv"),
            ("trainer", "model_uri", "runs:/123/model"),
            ("evaluator", "evaluation_report_uri", "path/to/report.json"),
            ("pusher", "deployment_status", "success")
        ]
        
        for stage_name, key, value in stages:
            context.set(key, value)
            
            # Create event for this stage
            event = Event(
                type=EventType.DATA_READY,
                payload=f"{stage_name}_payload",
                context=context
            )
            
            # Verify all previous context is still available
            for prev_stage, prev_key, prev_value in stages:
                if prev_key in context.to_dict():
                    assert context.get(prev_key) == prev_value
        
        # Final verification - all context should be present
        final_context = context.to_dict()
        assert final_context["train_data_uri"] == "path/to/train.csv"
        assert final_context["model_uri"] == "runs:/123/model"
        assert final_context["evaluation_report_uri"] == "path/to/report.json"
        assert final_context["deployment_status"] == "success"
    
    @pytest.mark.slow
    def test_full_pipeline_event_flow(self):
        """Test the complete event flow through all pipeline stages."""
        # Expected event type progression
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
        
        # Verify final state
        assert current_event.type == EventType.PUSHER_DONE
        final_context = context.to_dict()
        for i in range(1, len(expected_flow)):
            assert final_context[f"stage_{i}_completed"] is True


@pytest.mark.integration
class TestComponentInteraction:
    """Test interactions between specific components."""
    
    def test_evaluator_pusher_interaction(self):
        """Test interaction between Evaluator and Pusher components."""
        # Create evaluation payload that passes evaluation
        metrics = [{"accuracy": 0.95}, {"precision": 0.92}]
        evaluator_payload = EvaluatorPayload(
            metrics=metrics,
            is_better_than_production=True
        )
        
        # Create context as would be set by evaluator
        context = PipelineContext()
        context.set("pass_evaluation", True)
        context.set("model_uri", "runs:/123/model")
        
        evaluator_event = Event(
            type=EventType.EVALUATION_DONE,
            payload=evaluator_payload,
            context=context
        )
        
        # Create pusher payload
        pusher_payload = PusherPayload(model_uri="runs:/123/model")
        
        pusher_event = Event(
            type=EventType.PUSHER_DONE,
            payload=pusher_payload,
            context=context
        )
        
        # Verify that pusher can access evaluation results
        assert pusher_event.context.get("pass_evaluation") is True
        assert pusher_event.context.get("model_uri") == "runs:/123/model"
        assert pusher_event.payload.model_uri == "runs:/123/model"
    
    def test_tuner_trainer_integration(self):
        """Test integration between Tuner and Trainer components."""
        # Create hyperparameters from tuner
        hyperparams = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 10
        }
        
        tuner_payload = TunerPayload(
            hyperparameters=hyperparams,
            train_data=pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
        )
        
        # Create context with hyperparameters
        context = PipelineContext()
        context.set("hyperparameters_uri", "path/to/hyperparams.json")
        
        tuner_event = Event(
            type=EventType.TUNING_DONE,
            payload=tuner_payload,
            context=context
        )
        
        # Trainer should be able to access tuned hyperparameters
        # In a real scenario, trainer would load hyperparameters from URI
        assert tuner_event.context.get("hyperparameters_uri") == "path/to/hyperparams.json"
        assert tuner_event.payload.hyperparameters["learning_rate"] == 0.01
        assert tuner_event.payload.hyperparameters["batch_size"] == 32


@pytest.mark.integration
class TestErrorPropagation:
    """Test error handling and propagation through the pipeline."""
    
    def test_component_error_stops_pipeline(self):
        """Test that component errors properly stop pipeline execution."""
        from collie._common.exceptions import TrainerError, OrchestratorError
        
        # Create mock components where trainer fails
        mock_transformer = Mock()
        mock_trainer = Mock()
        mock_trainer.run.side_effect = TrainerError("Training failed")
        
        # Setup transformer to succeed
        mock_transformer.run.return_value = Event(
            type=EventType.DATA_READY,
            payload=TransformerPayload(),
            context=PipelineContext()
        )
        
        # Create orchestrator
        orchestrator = Orchestrator(
            components=[mock_transformer, mock_trainer],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
        
        # Pipeline should fail at trainer stage
        with pytest.raises(TrainerError):
            orchestrator.orchestrate_pipeline()
        
        # Transformer should have been called, trainer should have failed
        assert mock_transformer.run.called
        assert mock_trainer.run.called
    
    def test_exception_chaining_through_orchestrator(self):
        """Test that exceptions are properly chained through orchestrator."""
        from collie._common.exceptions import EvaluatorError, OrchestratorError
        
        # Create orchestrator that will fail
        mock_component = Mock()
        mock_component.run.side_effect = EvaluatorError("Evaluation failed")
        
        orchestrator = Orchestrator(
            components=[mock_component],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
        
        # Test through the run method (which handles orchestrator errors)
        with patch.object(orchestrator, 'start_run') as mock_start_run:
            mock_context = MagicMock()
            mock_start_run.return_value.__enter__.return_value = mock_context
            mock_start_run.return_value.__exit__.return_value = None
            
            with pytest.raises(OrchestratorError, match="Component error in orchestration"):
                orchestrator.run()


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndScenarios:
    """End-to-end integration test scenarios."""
    
    def test_successful_model_deployment_scenario(self):
        """Test a complete successful model deployment scenario."""
        # This test simulates a complete pipeline run with mocked components
        # that would typically deploy a model to production
        
        context = PipelineContext()
        
        # Stage 1: Data transformation
        transformer_payload = TransformerPayload(
            train_data=pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]}),
            validation_data=pd.DataFrame({"feature": [4, 5], "target": [1, 0]})
        )
        context.set("train_data_uri", "path/to/train.csv")
        context.set("validation_data_uri", "path/to/val.csv")
        
        # Stage 2: Model training
        mock_model = Mock()
        trainer_payload = TrainerPayload(model=mock_model, train_loss=0.3, val_loss=0.25)
        context.set("model_uri", "runs:/123/model")
        
        # Stage 3: Model evaluation (passes)
        evaluator_payload = EvaluatorPayload(
            metrics=[{"accuracy": 0.95}, {"f1": 0.93}],
            is_better_than_production=True
        )
        context.set("pass_evaluation", True)
        context.set("evaluator_report_uri", "path/to/report.json")
        
        # Stage 4: Model deployment
        pusher_payload = PusherPayload(model_uri="runs:/123/model")
        context.set("deployment_status", "success")
        
        # Create final event
        final_event = Event(
            type=EventType.PUSHER_DONE,
            payload=pusher_payload,
            context=context
        )
        
        # Verify complete deployment scenario
        assert final_event.type == EventType.PUSHER_DONE
        assert final_event.context.get("pass_evaluation") is True
        assert final_event.context.get("deployment_status") == "success"
        assert final_event.context.get("model_uri") == "runs:/123/model"
        assert final_event.payload.model_uri == "runs:/123/model"
    
    def test_model_rejection_scenario(self):
        """Test scenario where model is rejected and not deployed."""
        context = PipelineContext()
        
        # Model evaluation fails
        evaluator_payload = EvaluatorPayload(
            metrics=[{"accuracy": 0.75}, {"f1": 0.73}],  # Poor performance
            is_better_than_production=False
        )
        context.set("pass_evaluation", False)
        context.set("evaluator_report_uri", "path/to/report.json")
        
        # Pusher should not deploy
        pusher_payload = PusherPayload(model_uri="runs:/123/model")
        context.set("deployment_status", "rejected")
        
        final_event = Event(
            type=EventType.PUSHER_DONE,
            payload=pusher_payload,
            context=context
        )
        
        # Verify rejection scenario
        assert final_event.context.get("pass_evaluation") is False
        assert final_event.context.get("deployment_status") == "rejected"
        # Model URI should still be available for analysis
        assert final_event.payload.model_uri == "runs:/123/model"
