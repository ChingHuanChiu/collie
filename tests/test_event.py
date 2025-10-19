import pytest
from unittest.mock import Mock, patch
from typing import Any

from collie.contracts.event import Event, EventType, PipelineContext, EventHandler
from collie.core.models import TransformerPayload


class TestPipelineContext:
    """Test cases for PipelineContext class."""
    
    def test_pipeline_context_initialization_empty(self):
        """Test PipelineContext initialization with no data."""
        context = PipelineContext()
        assert context.data == {}
    
    def test_pipeline_context_initialization_with_data(self):
        """Test PipelineContext initialization with data."""
        initial_data = {"key1": "value1", "key2": 42}
        context = PipelineContext(data=initial_data)
        assert context.data == initial_data
    
    def test_get_existing_key(self):
        """Test getting an existing key from context."""
        context = PipelineContext({"test_key": "test_value"})
        result = context.get("test_key")
        assert result == "test_value"
    
    def test_get_non_existing_key_default_none(self):
        """Test getting a non-existing key returns None by default."""
        context = PipelineContext()
        result = context.get("non_existing_key")
        assert result is None
    
    def test_get_non_existing_key_custom_default(self):
        """Test getting a non-existing key with custom default."""
        context = PipelineContext()
        result = context.get("non_existing_key", "default_value")
        assert result == "default_value"
    
    def test_set_new_key(self):
        """Test setting a new key-value pair."""
        context = PipelineContext()
        context.set("new_key", "new_value")
        assert context.data["new_key"] == "new_value"
    
    def test_set_update_existing_key(self):
        """Test updating an existing key."""
        context = PipelineContext({"existing_key": "old_value"})
        context.set("existing_key", "new_value")
        assert context.data["existing_key"] == "new_value"
    
    def test_to_dict(self):
        """Test converting context to dictionary."""
        initial_data = {"key1": "value1", "key2": 42}
        context = PipelineContext(data=initial_data)
        result = context.to_dict()
        assert result == initial_data
        assert result is context.data  # Should return the same object


class TestEventType:
    """Test cases for EventType enum."""
    
    def test_event_type_enum_members(self):
        """Test that EventType has all expected members."""
        expected_members = {
            'INITIALIZE', 'DATA_READY', 'TRAINING_DONE', 
            'TUNING_DONE', 'EVALUATION_DONE', 'PUSHER_DONE', 'ERROR'
        }
        actual_members = {member.name for member in EventType}
        assert actual_members == expected_members
    
    def test_event_type_auto_values(self):
        """Test that EventType values are auto-generated."""
        # Since we use auto(), values should be unique integers
        values = [member.value for member in EventType]
        assert len(values) == len(set(values))  # All values should be unique
        assert all(isinstance(value, int) for value in values)


class TestEvent:
    """Test cases for Event class."""
    
    def test_event_creation_minimal(self):
        """Test creating an Event with minimal parameters."""
        payload = "test_payload"
        event = Event(payload=payload)
        
        assert event.type is None
        assert event.payload == payload
        assert isinstance(event.context, PipelineContext)
        assert event.context.data == {}
    
    def test_event_creation_complete(self):
        """Test creating an Event with all parameters."""
        payload = TransformerPayload()
        context = PipelineContext({"key": "value"})
        
        event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=context
        )
        
        assert event.type == EventType.DATA_READY
        assert event.payload == payload
        assert event.context == context
    
    def test_event_creation_with_custom_context(self):
        """Test creating an Event with custom context."""
        payload = "test_payload"
        custom_context = PipelineContext({"custom_key": "custom_value"})
        
        event = Event(payload=payload, context=custom_context)
        
        assert event.context == custom_context
        assert event.context.get("custom_key") == "custom_value"
    
    def test_event_model_config_allows_arbitrary_types(self):
        """Test that Event can handle arbitrary types due to model_config."""
        # This should work because of ConfigDict(arbitrary_types_allowed=True)
        mock_payload = Mock()
        context = PipelineContext()
        
        event = Event(
            type=EventType.INITIALIZE,
            payload=mock_payload,
            context=context
        )
        
        assert event.payload == mock_payload
        assert event.context == context
    
    def test_event_equality(self):
        """Test Event equality comparison."""
        payload1 = "test_payload"
        payload2 = "test_payload"
        context1 = PipelineContext({"key": "value"})
        context2 = PipelineContext({"key": "value"})
        
        event1 = Event(
            type=EventType.DATA_READY,
            payload=payload1,
            context=context1
        )
        
        event2 = Event(
            type=EventType.DATA_READY,
            payload=payload2,
            context=context2
        )
        
        # Note: Events might not be equal due to context being different objects
        # This tests the structure, not necessarily equality
        assert event1.type == event2.type
        assert event1.payload == event2.payload
        assert event1.context.to_dict() == event2.context.to_dict()


class ConcreteEventHandler(EventHandler):
    """Concrete implementation of EventHandler for testing."""
    
    def _handle(self, event: Event) -> Event:
        """Simple implementation that returns the event unchanged."""
        return event


class TestEventHandler:
    """Test cases for EventHandler abstract base class."""
    
    def test_event_handler_is_abstract(self):
        """Test that EventHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EventHandler()
    
    def test_concrete_event_handler_instantiation(self):
        """Test that concrete implementation can be instantiated."""
        handler = ConcreteEventHandler()
        assert isinstance(handler, EventHandler)
    
    def test_concrete_event_handler_handle_method(self):
        """Test that concrete implementation's _handle method works."""
        handler = ConcreteEventHandler()
        payload = "test_payload"
        context = PipelineContext({"test": "value"})
        
        event = Event(
            type=EventType.INITIALIZE,
            payload=payload,
            context=context
        )
        
        result = handler._handle(event)
        
        assert result == event
        assert result.type == EventType.INITIALIZE
        assert result.payload == payload
        assert result.context == context
    
    def test_event_handler_with_complex_payload(self):
        """Test EventHandler with complex payload types."""
        handler = ConcreteEventHandler()
        
        # Test with TransformerPayload
        import pandas as pd
        from collie.core.models import TransformerPayload
        
        payload = TransformerPayload(
            train_data=pd.DataFrame({"col": [1, 2, 3]})
        )
        
        event = Event(
            type=EventType.DATA_READY,
            payload=payload
        )
        
        result = handler._handle(event)
        
        assert result.payload == payload
        assert result.payload.train_data.equals(payload.train_data)


class TestEventIntegration:
    """Integration tests for Event system components."""
    
    def test_full_event_workflow(self):
        """Test a complete event workflow with context updates."""
        # Initialize context
        context = PipelineContext()
        
        # Create initial event
        initial_payload = "initial_data"
        event = Event(
            type=EventType.INITIALIZE,
            payload=initial_payload,
            context=context
        )
        
        # Simulate component processing
        context.set("step1_completed", True)
        context.set("model_uri", "runs:/123/model")
        
        # Update event
        updated_event = Event(
            type=EventType.DATA_READY,
            payload="processed_data",
            context=context
        )
        
        # Verify state
        assert updated_event.type == EventType.DATA_READY
        assert updated_event.payload == "processed_data"
        assert updated_event.context.get("step1_completed") is True
        assert updated_event.context.get("model_uri") == "runs:/123/model"
    
    def test_event_type_progression(self):
        """Test typical event type progression through pipeline."""
        context = PipelineContext()
        
        # Pipeline progression
        events = [
            Event(type=EventType.INITIALIZE, payload="init", context=context),
            Event(type=EventType.DATA_READY, payload="data", context=context),
            Event(type=EventType.TRAINING_DONE, payload="model", context=context),
            Event(type=EventType.TUNING_DONE, payload="tuned", context=context),
            Event(type=EventType.EVALUATION_DONE, payload="evaluated", context=context),
            Event(type=EventType.PUSHER_DONE, payload="deployed", context=context),
        ]
        
        # Verify progression
        expected_types = [
            EventType.INITIALIZE,
            EventType.DATA_READY,
            EventType.TRAINING_DONE,
            EventType.TUNING_DONE,
            EventType.EVALUATION_DONE,
            EventType.PUSHER_DONE,
        ]
        
        actual_types = [event.type for event in events]
        assert actual_types == expected_types
    
    def test_context_persistence_across_events(self):
        """Test that context data persists across different events."""
        context = PipelineContext()
        
        # First event adds data to context
        event1 = Event(type=EventType.INITIALIZE, payload="data1", context=context)
        event1.context.set("train_data_uri", "path/to/train.csv")
        
        # Second event uses same context
        event2 = Event(type=EventType.DATA_READY, payload="data2", context=context)
        event2.context.set("model_uri", "runs:/123/model")
        
        # Third event should see all previous context data
        event3 = Event(type=EventType.TRAINING_DONE, payload="data3", context=context)
        
        assert event3.context.get("train_data_uri") == "path/to/train.csv"
        assert event3.context.get("model_uri") == "runs:/123/model"
        
        # All events share the same context object
        assert event1.context is event2.context
        assert event2.context is event3.context
