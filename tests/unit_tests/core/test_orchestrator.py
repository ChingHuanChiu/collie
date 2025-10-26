"""
Unit Tests for Orchestrator Component

This module contains unit tests for the Orchestrator class,
which manages the execution flow of pipeline components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from collie.core.orchestrator.orchestrator import Orchestrator
from collie.contracts.event import Event, EventType, PipelineContext
from collie.core.transform.transform import Transformer
from collie.core.trainer.trainer import Trainer
from collie.core.evaluator.evaluator import Evaluator
from collie.core.pusher.pusher import Pusher
from collie._common.exceptions import OrchestratorError, TrainerError


@pytest.mark.unit
@pytest.mark.orchestrator
class TestOrchestratorInitialization:
    
    def test_initialization_with_all_parameters(self):
        """Test Orchestrator initialization with all parameters specified."""
        mock_transformer = Mock(spec=Transformer)
        mock_trainer = Mock(spec=Trainer)
        components = [mock_transformer, mock_trainer]
        
        orchestrator = Orchestrator(
            components=components,
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model",
            mlflow_tags={"env": "test", "version": "1.0"},
            experiment_name="test_experiment",
            description="Test orchestrator description"
        )
        
        assert orchestrator.components == components
        assert orchestrator.tracking_uri == "sqlite:///test.db"
        assert orchestrator.registered_model_name == "test_model"
        assert orchestrator.mlflow_tags == {"env": "test", "version": "1.0"}
        assert orchestrator.experiment_name == "test_experiment"
        assert orchestrator.description == "Test orchestrator description"
    
    def test_initialization_with_minimal_parameters(self):
        """Test Orchestrator initialization with only required parameters."""
        mock_component = Mock()
        
        orchestrator = Orchestrator(
            components=[mock_component],
            tracking_uri="sqlite:///minimal.db",
            registered_model_name="minimal_model"
        )
        
        assert len(orchestrator.components) == 1
        assert orchestrator.tracking_uri == "sqlite:///minimal.db"
        assert orchestrator.registered_model_name == "minimal_model"
        assert orchestrator.mlflow_tags is None
        assert orchestrator.experiment_name is None
        assert orchestrator.description is None
    
    def test_initialization_with_empty_components_list(self):
        """Test Orchestrator initialization with empty components list."""
        orchestrator = Orchestrator(
            components=[],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
        
        assert orchestrator.components == []


@pytest.mark.unit
@pytest.mark.orchestrator
class TestOrchestratorPipelineExecution:
    """Unit tests for Orchestrator pipeline execution."""
    
    def setup_method(self):
        """Set up test fixtures for pipeline execution tests."""
        self.mock_transformer = Mock(spec=Transformer)
        self.mock_trainer = Mock(spec=Trainer)
        self.mock_evaluator = Mock(spec=Evaluator)
        self.mock_pusher = Mock(spec=Pusher)
        
        self.components = [
            self.mock_transformer,
            self.mock_trainer,
            self.mock_evaluator,
            self.mock_pusher
        ]
        
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_calls_all_components(self, mock_logger, mock_mlflow_config):
        """Test that orchestrate_pipeline calls all components in order."""
        orchestrator = Orchestrator(
            components=self.components,
            tracking_uri=mock_mlflow_config.tracking_uri,
            registered_model_name="test_model",
            mlflow_tags={"env": "test"},
            experiment_name=mock_mlflow_config.experiment_name
        )
        
        # Manually set mlflow_config since we're not calling run()
        orchestrator.mlflow_config = mock_mlflow_config
        
        # Setup mock events
        for component in self.components:
            mock_event = Mock(spec=Event)
            mock_event.type = EventType.DATA_READY
            component.run.return_value = mock_event
        
        # Execute pipeline
        orchestrator.orchestrate_pipeline()
        
        # Verify all components were called
        for component in self.components:
            component.run.assert_called_once()
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_logging(self, mock_logger, mock_mlflow_config):
        """Test that orchestrate_pipeline logs execution properly."""
        orchestrator = Orchestrator(
            components=self.components,
            tracking_uri=mock_mlflow_config.tracking_uri,
            registered_model_name="test_model",
            experiment_name=mock_mlflow_config.experiment_name
        )
        
        # Manually set mlflow_config since we're not calling run()
        orchestrator.mlflow_config = mock_mlflow_config
        
        # Setup mock events
        for component in self.components:
            component.run.return_value = Mock(spec=Event)
        
        # Execute pipeline
        orchestrator.orchestrate_pipeline()
        
        # Verify logging
        mock_logger.info.assert_any_call("Pipeline started.")
        
        for i, component in enumerate(self.components):
            component_name = type(component).__name__
            mock_logger.info.assert_any_call(f"Running component {i}: {component_name}")
            mock_logger.info.assert_any_call(f"Component {i} finished: {component_name}")
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_sets_mlflow_config(self, mock_logger, mock_mlflow_config):
        """Test that orchestrate_pipeline sets MLflow config on components."""
        orchestrator = Orchestrator(
            components=self.components,
            tracking_uri=mock_mlflow_config.tracking_uri,
            registered_model_name="test_model",
            experiment_name=mock_mlflow_config.experiment_name
        )
        
        # Manually set mlflow_config since we're not calling run()
        orchestrator.mlflow_config = mock_mlflow_config
        
        # Setup mock events
        for component in self.components:
            component.run.return_value = Mock(spec=Event)
        
        # Execute pipeline
        orchestrator.orchestrate_pipeline()
        
        # Verify mlflow_config was set on each component
        for component in self.components:
            assert hasattr(component, 'mlflow_config')
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_sets_registered_model_name(self, mock_logger, mock_mlflow_config):
        """Test that orchestrate_pipeline sets registered_model_name on components."""
        orchestrator = Orchestrator(
            components=self.components,
            tracking_uri=mock_mlflow_config.tracking_uri,
            registered_model_name="test_model",
            experiment_name=mock_mlflow_config.experiment_name
        )
        
        # Manually set mlflow_config since we're not calling run()
        orchestrator.mlflow_config = mock_mlflow_config
        
        # Setup components with _registered_model_name attribute
        for component in self.components:
            component._registered_model_name = None
            component.run.return_value = Mock(spec=Event)
        
        # Execute pipeline
        orchestrator.orchestrate_pipeline()
        
        # Verify registered_model_name was set
        for component in self.components:
            assert component.registered_model_name == "test_model"
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_event_flow(self, mock_logger, mock_mlflow_config):
        """Test that events flow correctly between components."""
        orchestrator = Orchestrator(
            components=self.components,
            tracking_uri=mock_mlflow_config.tracking_uri,
            registered_model_name="test_model",
            experiment_name=mock_mlflow_config.experiment_name
        )
        
        # Manually set mlflow_config since we're not calling run()
        orchestrator.mlflow_config = mock_mlflow_config
        
        # Setup different events for each component
        events = [
            Mock(type=EventType.DATA_READY),
            Mock(type=EventType.TRAINING_DONE),
            Mock(type=EventType.EVALUATION_DONE),
            Mock(type=EventType.PUSHER_DONE)
        ]
        
        for i, component in enumerate(self.components):
            component.run.return_value = events[i]
        
        # Execute pipeline
        orchestrator.orchestrate_pipeline()
        
        # Verify first component receives initialize event
        first_call_event = self.components[0].run.call_args[0][0]
        assert first_call_event.type == EventType.INITIALIZE
        
        # Verify each subsequent component receives previous component's output
        for i in range(1, len(self.components)):
            received_event = self.components[i].run.call_args[0][0]
            expected_event = self.components[i - 1].run.return_value
            assert received_event == expected_event


@pytest.mark.unit
@pytest.mark.orchestrator
class TestOrchestratorRunMethod:
    """Unit tests for Orchestrator run method."""
    
    def setup_method(self):
        """Set up test fixtures for run method tests."""
        self.mock_component = Mock()
        self.orchestrator = Orchestrator(
            components=[self.mock_component],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
    
    @patch.object(Orchestrator, 'start_run')
    @patch.object(Orchestrator, 'orchestrate_pipeline')
    def test_run_method_successful_execution(self, mock_orchestrate, mock_start_run):
        """Test successful execution of run method."""
        # Setup mocks
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context
        mock_start_run.return_value.__exit__.return_value = None
        mock_orchestrate.return_value = "pipeline_result"
        
        # Execute
        result = self.orchestrator.run()
        
        # Verify
        assert result == "pipeline_result"
        mock_start_run.assert_called_once()
        mock_orchestrate.assert_called_once()
    
    @patch.object(Orchestrator, 'start_run')
    @patch.object(Orchestrator, 'orchestrate_pipeline')
    def test_run_method_with_tags_and_description(self, mock_orchestrate, mock_start_run):
        """Test run method includes tags and description in MLflow run."""
        # Create orchestrator with tags and description
        orchestrator = Orchestrator(
            components=[self.mock_component],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model",
            mlflow_tags={"env": "production", "version": "2.0"},
            description="Production pipeline"
        )
        
        # Setup mocks
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context
        mock_start_run.return_value.__exit__.return_value = None
        
        # Execute
        orchestrator.run()
        
        # Verify MLflow run was started with correct parameters
        mock_start_run.assert_called_once_with(
            tags={"env": "production", "version": "2.0"},
            run_name="Orchestrator",
            description="Production pipeline"
        )
    
    @patch.object(Orchestrator, 'start_run')
    @patch.object(Orchestrator, 'orchestrate_pipeline')
    def test_run_method_handles_component_error(self, mock_orchestrate, mock_start_run, mock_mlflow_config):
        """Test run method properly handles component errors."""
        # Setup mocks
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context
        mock_start_run.return_value.__exit__.return_value = None
        
        # Simulate component error
        mock_orchestrate.side_effect = TrainerError("Training failed")
        
        # Execute and verify error handling
        with pytest.raises(OrchestratorError, match=r"Component error in orchestration:.*Training failed"):
            self.orchestrator.run()
    
    @patch.object(Orchestrator, 'start_run')
    @patch.object(Orchestrator, 'orchestrate_pipeline')
    def test_run_method_handles_unexpected_error(self, mock_orchestrate, mock_start_run):
        """Test run method properly handles unexpected errors."""
        # Setup mocks
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context
        mock_start_run.return_value.__exit__.return_value = None
        
        # Simulate unexpected error
        mock_orchestrate.side_effect = ValueError("Unexpected error occurred")
        
        # Execute and verify error handling
        with pytest.raises(OrchestratorError, match="Unexpected orchestration error: Unexpected error occurred"):
            self.orchestrator.run()


@pytest.mark.unit
@pytest.mark.orchestrator
class TestOrchestratorUtilityMethods:
    """Unit tests for Orchestrator utility methods."""
    
    def test_initialize_event_creation(self):
        """Test that initialize_event creates proper Event object."""
        orchestrator = Orchestrator(
            components=[],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
        
        event = orchestrator.initialize_event()
        
        assert isinstance(event, Event)
        assert event.type == EventType.INITIALIZE
        assert event.payload is None
        assert isinstance(event.context, PipelineContext)
        assert event.context.data == {}
