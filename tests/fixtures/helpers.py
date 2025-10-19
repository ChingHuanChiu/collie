"""
Test helper utilities and assertion functions.
"""
from typing import Any
import pandas as pd
from collie.contracts.event import Event, EventType, PipelineContext


class TestHelpers:
    """Collection of helper methods for testing."""
    
    @staticmethod
    def assert_event_valid(
        event: Event,
        expected_type: EventType | None = None,
        expect_payload: bool = True
    ) -> None:
        """
        Validate Event object integrity.
        
        Args:
            event: Event object to validate
            expected_type: Expected EventType (optional)
            expect_payload: Whether payload should not be None
        """
        assert event is not None, "Event should not be None"
        
        if expected_type is not None:
            assert event.type == expected_type, \
                f"Expected event type {expected_type}, got {event.type}"
        
        if expect_payload:
            assert event.payload is not None, \
                "Event payload should not be None when expect_payload=True"
        
        assert isinstance(event.context, PipelineContext), \
            f"Event context should be PipelineContext, got {type(event.context)}"
    
    @staticmethod
    def assert_dataframe_valid(
        df: pd.DataFrame,
        min_rows: int | None = None,
        min_cols: int | None = None,
        required_columns: list[str] | None = None
    ) -> None:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum number of rows expected
            min_cols: Minimum number of columns expected
            required_columns: List of required column names
        """
        assert df is not None, "DataFrame should not be None"
        assert isinstance(df, pd.DataFrame), \
            f"Expected pandas DataFrame, got {type(df)}"
        
        if min_rows is not None:
            assert len(df) >= min_rows, \
                f"DataFrame should have at least {min_rows} rows, got {len(df)}"
        
        if min_cols is not None:
            assert len(df.columns) >= min_cols, \
                f"DataFrame should have at least {min_cols} columns, got {len(df.columns)}"
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            assert not missing_cols, \
                f"DataFrame missing required columns: {missing_cols}"
    
    @staticmethod
    def assert_dataframe_equal(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        check_like: bool = True,
        check_dtype: bool = False
    ) -> None:
        """
        Assert two DataFrames are equal.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            check_like: Allow column order differences
            check_dtype: Whether to check dtypes
        """
        pd.testing.assert_frame_equal(
            df1, df2,
            check_like=check_like,
            check_dtype=check_dtype
        )
    
    @staticmethod
    def assert_context_has_keys(context: PipelineContext, *keys: str) -> None:
        """
        Assert that context contains all specified keys.
        
        Args:
            context: PipelineContext to check
            keys: Keys that should exist in context
        """
        missing_keys = [k for k in keys if k not in context.data]
        assert not missing_keys, \
            f"Context missing keys: {missing_keys}"
    
    @staticmethod
    def assert_metrics_valid(metrics: list[dict[str, Any]]) -> None:
        """
        Validate metrics list structure.
        
        Args:
            metrics: List of metric dictionaries
        """
        assert isinstance(metrics, list), \
            f"Metrics should be a list, got {type(metrics)}"
        assert len(metrics) > 0, "Metrics list should not be empty"
        
        for i, metric in enumerate(metrics):
            assert isinstance(metric, dict), \
                f"Metric at index {i} should be dict, got {type(metric)}"
            assert len(metric) > 0, \
                f"Metric dict at index {i} should not be empty"


def create_sample_dataframe(
    n_rows: int = 100,
    n_features: int = 5,
    include_target: bool = True,
    random_state: int | None = 42
) -> pd.DataFrame:
    """
    Create a sample DataFrame for testing.
    
    Args:
        n_rows: Number of rows
        n_features: Number of feature columns
        include_target: Whether to include target column
        random_state: Random seed for reproducibility
    
    Returns:
        Sample DataFrame
    """
    import numpy as np
    
    if random_state is not None:
        np.random.seed(random_state)
    
    data = {
        f'feature_{i}': np.random.randn(n_rows)
        for i in range(n_features)
    }
    
    if include_target:
        data['target'] = np.random.randint(0, 2, n_rows)
    
    return pd.DataFrame(data)


def create_sample_model(model_type: str = "sklearn"):
    """
    Create a sample model for testing.
    
    Args:
        model_type: Type of model to create ('sklearn', 'pytorch', etc.)
    
    Returns:
        Sample model instance
    """
    if model_type == "sklearn":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10, random_state=42)
    
    elif model_type == "pytorch":
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        
        return SimpleModel()
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
