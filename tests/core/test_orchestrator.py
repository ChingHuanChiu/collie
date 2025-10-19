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


class TestOrchestrator:
    """Test cases for Orchestrator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock components
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
        
        self.orchestrator = Orchestrator(
            components=self.components,
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model",
            mlflow_tags={"env": "test"},
            experiment_name="test_experiment",
            description="Test orchestrator"
        )
    
    def test_orchestrator_initialization(self):
        """Test Orchestrator initialization."""
        assert self.orchestrator.components == self.components
        assert self.orchestrator.tracking_uri == "sqlite:///test.db"
        assert self.orchestrator.registered_model_name == "test_model"
        assert self.orchestrator.mlflow_tags == {"env": "test"}
        assert self.orchestrator.experiment_name == "test_experiment"
        assert self.orchestrator.description == "Test orchestrator"
    
    def test_orchestrator_initialization_minimal(self):
        """Test Orchestrator initialization with minimal parameters."""
        minimal_orchestrator = Orchestrator(
            components=[self.mock_transformer],
            tracking_uri="sqlite:///minimal.db",
            registered_model_name="minimal_model"
        )
        
        assert len(minimal_orchestrator.components) == 1
        assert minimal_orchestrator.tracking_uri == "sqlite:///minimal.db"
        assert minimal_orchestrator.registered_model_name == "minimal_model"
        assert minimal_orchestrator.mlflow_tags is None
        assert minimal_orchestrator.experiment_name is None
        assert minimal_orchestrator.description is None
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_success(self, mock_logger):
        """Test successful pipeline orchestration."""
        # Setup mock components to return events
        mock_events = []
        for i, component in enumerate(self.components):
            mock_event = Mock(spec=Event)
            mock_event.type = EventType.DATA_READY
            mock_events.append(mock_event)
            component.run.return_value = mock_event
        
        # Execute
        self.orchestrator.orchestrate_pipeline()
        
        # Verify logger calls
        mock_logger.info.assert_any_call("Pipeline started.")
        
        # Verify each component was called with proper logging
        for i, component in enumerate(self.components):
            component_name = type(component).__name__
            mock_logger.info.assert_any_call(f"Running component {i}: {component_name}")
            mock_logger.info.assert_any_call(f"Component {i} finished: {component_name}")
            
            # Verify component was called
            component.run.assert_called_once()
            
            # Verify mlflow_config was set
            assert hasattr(component, 'mlflow_config')
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_with_registered_model_name(self, mock_logger):
        """Test pipeline orchestration sets registered_model_name on applicable components."""
        # Create components that have _registered_model_name attribute
        for component in self.components:
            component._registered_model_name = None
        
        # Setup return events
        for component in self.components:
            mock_event = Mock(spec=Event)
            component.run.return_value = mock_event
        
        # Execute
        self.orchestrator.orchestrate_pipeline()
        
        # Verify registered_model_name was set on all components
        for component in self.components:
            assert component.registered_model_name == "test_model"
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_event_flow(self, mock_logger):
        """Test that events flow correctly between components."""
        # Setup mock events with different types to simulate pipeline flow
        events = [
            Mock(type=EventType.INITIALIZE),
            Mock(type=EventType.DATA_READY),
            Mock(type=EventType.TRAINING_DONE),
            Mock(type=EventType.EVALUATION_DONE),
            Mock(type=EventType.PUSHER_DONE)
        ]
        
        # Setup components to return next event in sequence
        for i, component in enumerate(self.components):
            component.run.return_value = events[i + 1] if i + 1 < len(events) else events[-1]
        
        # Execute
        self.orchestrator.orchestrate_pipeline()
        
        # Verify event flow
        # First component should receive initialize event
        initial_event = self.components[0].run.call_args[0][0]
        assert initial_event.type == EventType.INITIALIZE
        
        # Each subsequent component should receive output of previous
        for i in range(1, len(self.components)):
            received_event = self.components[i].run.call_args[0][0]
            expected_event = self.components[i - 1].run.return_value
            assert received_event == expected_event
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_components_without_registered_model_name(self, mock_logger):
        """Test pipeline with components that don't have _registered_model_name attribute."""
        # Create a component without _registered_model_name
        mock_component_no_model_name = Mock()
        # Don't set _registered_model_name attribute
        
        components = [mock_component_no_model_name]
        orchestrator = Orchestrator(
            components=components,
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
        
        # Setup return event
        mock_event = Mock(spec=Event)
        mock_component_no_model_name.run.return_value = mock_event
        
        # Execute - should not raise error
        orchestrator.orchestrate_pipeline()
        
        # Verify component was still called
        mock_component_no_model_name.run.assert_called_once()
        
        # Verify mlflow_config was still set
        assert hasattr(mock_component_no_model_name, 'mlflow_config')
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_empty_components(self, mock_logger):
        """Test pipeline orchestration with empty components list."""
        empty_orchestrator = Orchestrator(
            components=[],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
        
        # Execute - should not raise error
        empty_orchestrator.orchestrate_pipeline()
        
        # Verify pipeline started log
        mock_logger.info.assert_called_with("Pipeline started.")
    
    @patch('collie.core.orchestrator.orchestrator.logger')
    def test_orchestrate_pipeline_single_component(self, mock_logger):
        """Test pipeline orchestration with single component."""
        single_orchestrator = Orchestrator(
            components=[self.mock_transformer],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
        
        # Setup return event
        mock_event = Mock(spec=Event)
        self.mock_transformer.run.return_value = mock_event
        
        # Execute
        single_orchestrator.orchestrate_pipeline()
        
        # Verify logging
        mock_logger.info.assert_any_call("Pipeline started.")
        mock_logger.info.assert_any_call("Running component 0: Mock")
        mock_logger.info.assert_any_call("Component 0 finished: Mock")
        
        # Verify component was called
        self.mock_transformer.run.assert_called_once()


class TestOrchestratorBase:
    """Test cases for OrchestratorBase functionality through Orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_component = Mock()
        self.orchestrator = Orchestrator(
            components=[self.mock_component],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model"
        )
    
    @patch.object(Orchestrator, 'start_run')
    @patch.object(Orchestrator, 'orchestrate_pipeline')
    def test_run_method_success(self, mock_orchestrate, mock_start_run):
        """Test successful run method execution."""
        # Setup mocks
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context
        mock_start_run.return_value.__exit__.return_value = None
        mock_orchestrate.return_value = "pipeline_result"
        
        # Execute
        result = self.orchestrator.run()
        
        # Verify
        assert result == "pipeline_result"
        
        # Verify MLflow run was started with correct parameters
        mock_start_run.assert_called_once_with(
            tags=None,  # No tags set in this test
            run_name="Orchestrator",
            description=None  # No description set in this test
        )
        
        # Verify orchestrate_pipeline was called
        mock_orchestrate.assert_called_once()
    
    @patch.object(Orchestrator, 'start_run')
    @patch.object(Orchestrator, 'orchestrate_pipeline')
    def test_run_method_with_tags_and_description(self, mock_orchestrate, mock_start_run):
        """Test run method with tags and description."""
        # Setup orchestrator with tags and description
        orchestrator = Orchestrator(
            components=[self.mock_component],
            tracking_uri="sqlite:///test.db",
            registered_model_name="test_model",
            mlflow_tags={"env": "production", "version": "1.0"},
            description="Production pipeline run"
        )
        
        # Setup mocks
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context
        mock_start_run.return_value.__exit__.return_value = None
        mock_orchestrate.return_value = "tagged_result"
        
        # Execute
        result = orchestrator.run()
        
        # Verify
        assert result == "tagged_result"
        
        # Verify MLflow run was started with correct parameters
        mock_start_run.assert_called_once_with(
            tags={"env": "production", "version": "1.0"},
            run_name="Orchestrator",
            description="Production pipeline run"
        )
    
    @patch.object(Orchestrator, 'start_run')
    @patch.object(Orchestrator, 'orchestrate_pipeline')
    def test_run_method_component_error_handling(self, mock_orchestrate, mock_start_run):
        """Test run method handles component errors correctly."""
        # Setup mocks
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context
        mock_start_run.return_value.__exit__.return_value = None
        
        # Make orchestrate_pipeline raise a component error
        mock_orchestrate.side_effect = TrainerError("Training failed")
        
        # Execute and verify exception handling
        with pytest.raises(OrchestratorError, match="Component error in orchestration: Training failed"):
            self.orchestrator.run()
    
    @patch.object(Orchestrator, 'start_run')
    @patch.object(Orchestrator, 'orchestrate_pipeline')
    def test_run_method_unexpected_error_handling(self, mock_orchestrate, mock_start_run):
        """Test run method handles unexpected errors correctly."""
        # Setup mocks
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context
        mock_start_run.return_value.__exit__.return_value = None
        
        # Make orchestrate_pipeline raise an unexpected error
        mock_orchestrate.side_effect = ValueError("Unexpected error")
        
        # Execute and verify exception handling
        with pytest.raises(OrchestratorError, match="Unexpected orchestration error: Unexpected error"):
            self.orchestrator.run()
    
    def test_initialize_event(self):
        """Test initialize_event method."""
        event = self.orchestrator.initialize_event()
        
        assert isinstance(event, Event)
        assert event.type == EventType.INITIALIZE
        assert event.payload is None
        assert isinstance(event.context, PipelineContext)
    
    @patch.object(Orchestrator, 'start_run')
    @patch.object(Orchestrator, 'orchestrate_pipeline')
    def test_run_method_mlflow_config_setup(self, mock_orchestrate, mock_start_run):
        """Test that run method sets up MLflow config correctly."""
        # Setup mocks
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context
        mock_start_run.return_value.__exit__.return_value = None
        mock_orchestrate.return_value = "result"
        
        # Execute
        self.orchestrator.run()
        
        # Verify MLflow config was set
        assert hasattr(self.orchestrator, 'mlflow_config')
        assert self.orchestrator.mlflow_config.tracking_uri == "sqlite:///test.db"
        assert self.orchestrator.mlflow_config.experiment_name is None  # Not set in this test
