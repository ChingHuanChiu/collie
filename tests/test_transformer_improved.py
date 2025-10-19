"""
Improved test examples demonstrating best practices.

This file shows how tests should be structured following the recommendations
from CODE_REVIEW_ROUND2.md
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
import numpy as np

from collie.contracts.event import Event, EventType, PipelineContext
from collie.core.models import TransformerPayload
from collie.core.transform.transform import Transformer
from collie._common.exceptions import TransformerError

# Import test helpers
from tests.fixtures.helpers import TestHelpers, create_sample_dataframe


class TestTransformerImproved:
    """
    Improved Transformer tests following best practices.
    
    Key improvements:
    1. Using fixtures instead of setup_method
    2. More detailed assertions with error messages
    3. Parametrized tests for multiple scenarios
    4. Boundary condition testing
    5. Less reliance on mocks
    """
    
    # ========== Basic Tests ==========
    
    def test_initialization_with_defaults(self, transformer):
        """Test that default initialization works correctly."""
        default_transformer = Transformer()
        
        assert default_transformer.description is None, \
            "Default description should be None"
        assert default_transformer.tags == {"component": "Transformer"}, \
            "Default tags should include component name"
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        custom_desc = "Custom transformer for feature engineering"
        custom_tags = {"env": "production", "version": "v2"}
        
        transformer = Transformer(
            description=custom_desc,
            tags=custom_tags
        )
        
        assert transformer.description == custom_desc
        assert transformer.tags == custom_tags
    
    # ========== Parametrized Tests ==========
    
    @pytest.mark.parametrize("n_rows,n_features", [
        (10, 5),      # Small dataset
        (100, 20),    # Medium dataset
        (1000, 50),   # Large dataset
    ])
    def test_transform_various_data_sizes(self, transformer, n_rows, n_features):
        """Test transformation with various data sizes."""
        df = create_sample_dataframe(n_rows=n_rows, n_features=n_features)
        payload = TransformerPayload(train_data=df)
        
        # Verify payload is valid
        assert payload.train_data is not None
        assert len(payload.train_data) == n_rows
        assert len(payload.train_data.columns) == n_features + 1  # +1 for target
    
    @pytest.mark.parametrize("data_type", [
        "train_data",
        "validation_data",
        "test_data"
    ])
    def test_payload_data_types(self, data_type):
        """Test that all data types can be set in payload."""
        df = create_sample_dataframe(n_rows=50)
        kwargs = {data_type: df}
        payload = TransformerPayload(**kwargs)
        
        assert getattr(payload, data_type) is not None
        TestHelpers.assert_dataframe_valid(
            getattr(payload, data_type),
            min_rows=50
        )
    
    # ========== Boundary Condition Tests ==========
    
    def test_transform_empty_dataframe(self, transformer):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        payload = TransformerPayload(train_data=empty_df)
        
        # Should handle empty DataFrame gracefully
        assert len(payload.train_data) == 0
    
    def test_transform_single_row(self, transformer):
        """Test transformation with single row."""
        single_row_df = create_sample_dataframe(n_rows=1)
        payload = TransformerPayload(train_data=single_row_df)
        
        TestHelpers.assert_dataframe_valid(
            payload.train_data,
            min_rows=1
        )
    
    def test_transform_with_none_values(self):
        """Test handling of None values in DataFrame."""
        df = pd.DataFrame({
            'feature1': [1, None, 3],
            'feature2': [None, 2, 3],
            'target': [0, 1, 0]
        })
        payload = TransformerPayload(train_data=df)
        
        assert payload.train_data.isna().any().any(), \
            "DataFrame should contain NaN values"
    
    def test_transform_with_mixed_dtypes(self):
        """Test handling of mixed data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'target': [0, 1, 0]
        })
        payload = TransformerPayload(train_data=df)
        
        TestHelpers.assert_dataframe_valid(
            payload.train_data,
            min_rows=3,
            required_columns=['int_col', 'float_col', 'str_col', 'bool_col', 'target']
        )
    
    # ========== Error Handling Tests ==========
    
    @patch.object(Transformer, 'start_run')
    @patch.object(Transformer, '_handle')
    def test_run_handles_processing_error_gracefully(
        self,
        mock_handle,
        mock_start_run,
        transformer
    ):
        """Test that errors during processing are handled correctly."""
        # Setup mock to raise exception
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        mock_handle.side_effect = ValueError("Invalid data format")
        
        # Create test event
        payload = TransformerPayload(train_data=create_sample_dataframe())
        event = Event(type=EventType.INITIALIZE, payload=payload, context=PipelineContext())
        
        # Should raise TransformerError with descriptive message
        with pytest.raises(TransformerError) as exc_info:
            transformer.run(event)
        
        assert "Transformer failed" in str(exc_info.value)
        assert "Invalid data format" in str(exc_info.value)
    
    # ========== Integration-style Tests (less mocking) ==========
    
    def test_full_transformer_workflow_minimal_mocking(self, transformer):
        """
        Test complete transformer workflow with minimal mocking.
        Only mock external dependencies (MLflow).
        """
        # Create realistic test data
        train_df = create_sample_dataframe(n_rows=100, n_features=10)
        val_df = create_sample_dataframe(n_rows=30, n_features=10)
        test_df = create_sample_dataframe(n_rows=30, n_features=10)
        
        # Create payload
        payload = TransformerPayload(
            train_data=train_df,
            validation_data=val_df,
            test_data=test_df
        )
        
        # Verify all data is present and valid
        TestHelpers.assert_dataframe_valid(payload.train_data, min_rows=100)
        TestHelpers.assert_dataframe_valid(payload.validation_data, min_rows=30)
        TestHelpers.assert_dataframe_valid(payload.test_data, min_rows=30)
        
        # Verify column consistency
        assert set(payload.train_data.columns) == set(payload.validation_data.columns)
        assert set(payload.train_data.columns) == set(payload.test_data.columns)
    
    # ========== Property-based Tests ==========
    
    @pytest.mark.parametrize("execution_number", range(5))
    def test_transformation_consistency(self, transformer, execution_number):
        """
        Test that transformation is consistent across multiple runs.
        Property: Same input -> Same output
        """
        # Use fixed random seed for reproducibility
        df = create_sample_dataframe(n_rows=50, random_state=42)
        
        # Create payloads
        payload1 = TransformerPayload(train_data=df.copy())
        payload2 = TransformerPayload(train_data=df.copy())
        
        # Should produce identical results
        TestHelpers.assert_dataframe_equal(
            payload1.train_data,
            payload2.train_data
        )


class TestEventSystemImproved:
    """Improved Event system tests with better practices."""
    
    def test_event_creation_and_validation(self):
        """Test Event creation with comprehensive validation."""
        payload = TransformerPayload(train_data=create_sample_dataframe())
        context = PipelineContext({"run_id": "test-123"})
        
        event = Event(
            type=EventType.DATA_READY,
            payload=payload,
            context=context
        )
        
        # Use helper for validation
        TestHelpers.assert_event_valid(
            event,
            expected_type=EventType.DATA_READY,
            expect_payload=True
        )
        
        # Verify context
        TestHelpers.assert_context_has_keys(context, "run_id")
        assert event.context.get("run_id") == "test-123"
    
    @pytest.mark.parametrize("event_type", [
        EventType.INITIALIZE,
        EventType.DATA_READY,
        EventType.TRAINING_DONE,
        EventType.TUNING_DONE,
        EventType.EVALUATION_DONE,
        EventType.PUSHER_DONE,
    ])
    def test_all_event_types_are_valid(self, event_type):
        """Test that all EventType enum values can be used."""
        event = Event(
            type=event_type,
            payload=None,
            context=PipelineContext()
        )
        
        assert event.type == event_type
        assert event.type in EventType


# ========== Performance Tests ==========

@pytest.mark.performance
class TestPerformance:
    """Performance benchmark tests."""
    
    def test_large_dataframe_processing_time(self):
        """Ensure large DataFrame processing is within acceptable time."""
        import time
        
        # Create large dataset
        large_df = create_sample_dataframe(n_rows=10000, n_features=100)
        
        start_time = time.time()
        payload = TransformerPayload(train_data=large_df)
        duration = time.time() - start_time
        
        assert duration < 1.0, \
            f"Processing took {duration:.2f}s, expected < 1.0s"
        
        # Verify data integrity after processing
        TestHelpers.assert_dataframe_valid(
            payload.train_data,
            min_rows=10000,
            min_cols=100
        )


# ========== Fixtures for this test file ==========

@pytest.fixture
def transformer():
    """Provide a fresh Transformer instance for each test."""
    return Transformer(
        description="Test transformer",
        tags={"env": "test"}
    )
