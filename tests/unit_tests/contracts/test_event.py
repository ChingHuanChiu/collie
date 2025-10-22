"""
Unit Tests for Event System

This module contains unit tests for the event system components including:
- PipelineContext: Context manager for pipeline data
- EventType: Enumeration of event types
- Event: Event class for pipeline communication
- EventHandler: Abstract base class for event handlers
"""

import pytest
from unittest.mock import Mock
from typing import Any

from collie.contracts.event import Event, EventType, PipelineContext, EventHandler
from collie.core.models import TransformerPayload


@pytest.mark.unit
class TestPipelineContext:
    """Unit tests for PipelineContext class."""
    
    def test_initialization_with_empty_data(self):
        """Test PipelineContext initialization with no data."""
        context = PipelineContext()
        assert context.data == {}
    
    def test_initialization_with_data(self):
        """Test PipelineContext initialization with initial data."""
        initial_data = {"key1": "value1", "key2": 42}
        context = PipelineContext(data=initial_data)
        assert context.data == initial_data
    
    def test_get_existing_key(self):
        """Test retrieving an existing key from context."""
        context = PipelineContext({"test_key": "test_value"})
        result = context.get("test_key")
        assert result == "test_value"
    
    def test_get_non_existing_key_returns_none(self):
        """Test getting a non-existing key returns None by default."""
        context = PipelineContext()
        result = context.get("non_existing_key")
        assert result is None
    
    def test_get_non_existing_key_with_default(self):
        """Test getting a non-existing key with custom default value."""
        context = PipelineContext()
        result = context.get("non_existing_key", "default_value")
        assert result == "default_value"
    
    def test_set_new_key(self):
        """Test setting a new key-value pair."""
        context = PipelineContext()
        context.set("new_key", "new_value")
        assert context.data["new_key"] == "new_value"
    
    def test_set_updates_existing_key(self):
        """Test updating an existing key value."""
        context = PipelineContext({"existing_key": "old_value"})
        context.set("existing_key", "new_value")
        assert context.data["existing_key"] == "new_value"
    
    def test_to_dict_returns_data(self):
        """Test converting context to dictionary."""
        initial_data = {"key1": "value1", "key2": 42}
        context = PipelineContext(data=initial_data)
        result = context.to_dict()
        assert result == initial_data
        assert result is context.data


@pytest.mark.unit
class TestEventType:
    """Unit tests for EventType enum."""
    
    def test_enum_has_all_expected_members(self):
        """Test that EventType contains all expected event type members."""
        expected_members = {
            'INITIALIZE', 'DATA_READY', 'TRAINING_DONE', 
            'TUNING_DONE', 'EVALUATION_DONE', 'PUSHER_DONE', 'ERROR'
        }
        actual_members = {member.name for member in EventType}
        assert actual_members == expected_members
    
    def test_enum_values_are_unique(self):
        """Test that EventType values are unique integers."""
        values = [member.value for member in EventType]
        assert len(values) == len(set(values))
        assert all(isinstance(value, int) for value in values)


@pytest.mark.unit
class TestEvent:
    """Unit tests for Event class."""
    
    def test_creation_with_minimal_parameters(self):
        """Test creating an Event with only required parameters."""
        payload = "test_payload"
        event = Event(payload=payload)
        
        assert event.type is None
        assert event.payload == payload
        assert isinstance(event.context, PipelineContext)
        assert event.context.data == {}
    
    def test_creation_with_all_parameters(self):
        """Test creating an Event with all parameters specified."""
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
    
    def test_creation_with_custom_context(self):
        """Test creating an Event with custom context data."""
        payload = "test_payload"
        custom_context = PipelineContext({"custom_key": "custom_value"})
        
        event = Event(payload=payload, context=custom_context)
        
        assert event.context == custom_context
        assert event.context.get("custom_key") == "custom_value"
    
    def test_allows_arbitrary_types_in_payload(self):
        """Test that Event accepts arbitrary types due to model configuration."""
        mock_payload = Mock()
        context = PipelineContext()
        
        event = Event(
            type=EventType.INITIALIZE,
            payload=mock_payload,
            context=context
        )
        
        assert event.payload == mock_payload
        assert event.context == context
    
    def test_event_attributes_match_constructor(self):
        """Test that Event attributes match constructor parameters."""
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
        
        assert event1.type == event2.type
        assert event1.payload == event2.payload
        assert event1.context.to_dict() == event2.context.to_dict()


class ConcreteEventHandler(EventHandler):
    """Concrete implementation of EventHandler for testing purposes."""
    
    def _handle(self, event: Event) -> Event:
        """Implementation that returns the event unchanged."""
        return event


@pytest.mark.unit
class TestEventHandler:
    """Unit tests for EventHandler abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that EventHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EventHandler()
    
    def test_can_instantiate_concrete_implementation(self):
        """Test that concrete implementation can be instantiated."""
        handler = ConcreteEventHandler()
        assert isinstance(handler, EventHandler)
    
    def test_handle_method_processes_event(self):
        """Test that concrete implementation's handle method works correctly."""
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
    
    def test_handles_complex_payload_types(self):
        """Test EventHandler with complex payload types like DataFrames."""
        handler = ConcreteEventHandler()
        
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


@pytest.mark.unit
class TestEventContextInteraction:
    """Unit tests for Event and Context interaction."""
    
    def test_context_updates_persist_in_event(self):
        """Test that context updates are reflected in the event."""
        context = PipelineContext()
        event = Event(
            type=EventType.INITIALIZE,
            payload="initial_data",
            context=context
        )
        
        context.set("step1_completed", True)
        context.set("model_uri", "runs:/123/model")
        
        assert event.context.get("step1_completed") is True
        assert event.context.get("model_uri") == "runs:/123/model"
    
    def test_multiple_events_share_same_context(self):
        """Test that multiple events can share the same context object."""
        context = PipelineContext()
        
        event1 = Event(type=EventType.INITIALIZE, payload="data1", context=context)
        event1.context.set("train_data_uri", "path/to/train.csv")
        
        event2 = Event(type=EventType.DATA_READY, payload="data2", context=context)
        event2.context.set("model_uri", "runs:/123/model")
        
        event3 = Event(type=EventType.TRAINING_DONE, payload="data3", context=context)
        
        assert event3.context.get("train_data_uri") == "path/to/train.csv"
        assert event3.context.get("model_uri") == "runs:/123/model"
        assert event1.context is event2.context is event3.context
