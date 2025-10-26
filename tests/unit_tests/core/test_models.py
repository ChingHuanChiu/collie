import pytest
import pandas as pd
from collie.core.models import (
    BasePayload,
    TransformerPayload,
    TrainerPayload,
    TunerPayload,
    TransformerArtifact,
    TrainerArtifact,
    TunerArtifact,
    EvaluatorArtifact,
)


class TestBasePayload:
    """Test suite for BasePayload class."""
    
    def test_base_payload_initialization(self):
        """Test BasePayload can be initialized."""
        payload = BasePayload()
        
        assert payload.extra_data == {}
    
    def test_base_payload_with_extra_data(self):
        """Test BasePayload initialization with extra_data."""
        extra = {"key": "value", "count": 42}
        payload = BasePayload(extra_data=extra)
        
        assert payload.extra_data == extra
    
    def test_set_extra_single_value(self):
        """Test set_extra method adds key-value pair."""
        payload = BasePayload()
        result = payload.set_extra("feature_names", ["age", "income"])
        
        assert payload.extra_data["feature_names"] == ["age", "income"]
        assert result is payload  # Returns self for chaining
    
    def test_set_extra_chaining(self):
        """Test set_extra supports method chaining."""
        payload = BasePayload()
        result = payload.set_extra("key1", "value1").set_extra("key2", "value2")
        
        assert payload.extra_data["key1"] == "value1"
        assert payload.extra_data["key2"] == "value2"
        assert result is payload
    
    def test_get_extra_existing_key(self):
        """Test get_extra retrieves existing value."""
        payload = BasePayload()
        payload.set_extra("test_key", "test_value")
        
        value = payload.get_extra("test_key")
        assert value == "test_value"
    
    def test_get_extra_missing_key_with_default(self):
        """Test get_extra returns default for missing key."""
        payload = BasePayload()
        
        value = payload.get_extra("missing_key", default="default_value")
        assert value == "default_value"
    
    def test_get_extra_missing_key_no_default(self):
        """Test get_extra returns None for missing key without default."""
        payload = BasePayload()
        
        value = payload.get_extra("missing_key")
        assert value is None
    
    def test_has_extra_existing_key(self):
        """Test has_extra returns True for existing key."""
        payload = BasePayload()
        payload.set_extra("existing", "value")
        
        assert payload.has_extra("existing") is True
    
    def test_has_extra_missing_key(self):
        """Test has_extra returns False for missing key."""
        payload = BasePayload()
        
        assert payload.has_extra("missing") is False
    
    def test_extra_data_with_complex_types(self):
        """Test extra_data can store complex types."""
        payload = BasePayload()
        
        payload.set_extra("list", [1, 2, 3])
        payload.set_extra("dict", {"nested": "value"})
        payload.set_extra("dataframe", pd.DataFrame({"col": [1, 2]}))
        
        assert payload.get_extra("list") == [1, 2, 3]
        assert payload.get_extra("dict") == {"nested": "value"}
        assert isinstance(payload.get_extra("dataframe"), pd.DataFrame)


class TestTransformerPayload:
    """Test suite for TransformerPayload."""
    
    def test_transformer_payload_initialization_empty(self):
        """Test TransformerPayload can be initialized without data."""
        payload = TransformerPayload()
        
        assert payload.train_data is None
        assert payload.validation_data is None
        assert payload.test_data is None
        assert payload.extra_data == {}
    
    def test_transformer_payload_with_train_data(self):
        """Test TransformerPayload with train_data."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        payload = TransformerPayload(train_data=df)
        
        assert payload.train_data is not None
        assert isinstance(payload.train_data, pd.DataFrame)
        assert len(payload.train_data) == 3
    
    def test_transformer_payload_with_all_data(self):
        """Test TransformerPayload with all data splits."""
        train_df = pd.DataFrame({"col": [1, 2, 3]})
        val_df = pd.DataFrame({"col": [4, 5]})
        test_df = pd.DataFrame({"col": [6]})
        
        payload = TransformerPayload(
            train_data=train_df,
            validation_data=val_df,
            test_data=test_df
        )
        
        assert len(payload.train_data) == 3
        assert len(payload.validation_data) == 2
        assert len(payload.test_data) == 1
    
    def test_transformer_payload_inherits_base_methods(self):
        """Test TransformerPayload inherits BasePayload methods."""
        payload = TransformerPayload()
        
        payload.set_extra("feature_names", ["col1", "col2"])
        assert payload.get_extra("feature_names") == ["col1", "col2"]
        assert payload.has_extra("feature_names") is True
    
    def test_transformer_payload_model_dump(self):
        """Test TransformerPayload can be dumped to dict."""
        df = pd.DataFrame({"col": [1, 2]})
        payload = TransformerPayload(train_data=df)
        
        dumped = payload.model_dump()
        assert "train_data" in dumped
        assert "validation_data" in dumped
        assert "test_data" in dumped
        assert "extra_data" in dumped


class TestTrainerPayload:
    """Test suite for TrainerPayload."""
    
    def test_trainer_payload_initialization_empty(self):
        """Test TrainerPayload can be initialized without model."""
        payload = TrainerPayload()
        
        assert payload.model is None
        assert payload.extra_data == {}
    
    def test_trainer_payload_with_model(self):
        """Test TrainerPayload with model object."""
        mock_model = {"type": "sklearn", "params": {}}
        payload = TrainerPayload(model=mock_model)
        
        assert payload.model is not None
        assert payload.model["type"] == "sklearn"
    
    def test_trainer_payload_inherits_base_methods(self):
        """Test TrainerPayload inherits BasePayload methods."""
        payload = TrainerPayload()
        
        payload.set_extra("model_params", {"n_estimators": 100})
        assert payload.get_extra("model_params") == {"n_estimators": 100}
    
    def test_trainer_payload_arbitrary_types(self):
        """Test TrainerPayload accepts arbitrary types for model."""
        class CustomModel:
            def __init__(self):
                self.trained = False
        
        model = CustomModel()
        payload = TrainerPayload(model=model)
        
        assert isinstance(payload.model, CustomModel)
        assert payload.model.trained is False


class TestTunerPayload:
    """Test suite for TunerPayload."""
    
    def test_tuner_payload_initialization(self):
        """Test TunerPayload can be initialized."""
        payload = TunerPayload(hyperparameters={})
        
        assert payload.extra_data == {}
        assert payload.hyperparameters == {}
    
    def test_tuner_payload_with_hyperparameters(self):
        """Test TunerPayload with hyperparameters."""
        hyperparams = {
            "learning_rate": 0.01,
            "max_depth": 5
        }
        payload = TunerPayload(hyperparameters=hyperparams)
        
        assert payload.hyperparameters["learning_rate"] == 0.01
        assert payload.hyperparameters["max_depth"] == 5


class TestTransformerArtifact:
    """Test suite for TransformerArtifact."""
    
    def test_transformer_artifact_defaults(self):
        """Test TransformerArtifact default values."""
        artifact = TransformerArtifact()
        
        assert artifact.train_data == "train.csv"
        assert artifact.validation_data == "validation.csv"
        assert artifact.test_data == "test.csv"
    
    def test_transformer_artifact_custom_values(self):
        """Test TransformerArtifact with custom filenames."""
        artifact = TransformerArtifact(
            train_data="custom_train.parquet",
            validation_data="custom_val.parquet",
            test_data="custom_test.parquet"
        )
        
        assert artifact.train_data == "custom_train.parquet"
        assert artifact.validation_data == "custom_val.parquet"
        assert artifact.test_data == "custom_test.parquet"
    
    def test_transformer_artifact_model_dump(self):
        """Test TransformerArtifact can be dumped to dict."""
        artifact = TransformerArtifact()
        dumped = artifact.model_dump()
        
        assert dumped["train_data"] == "train.csv"
        assert dumped["validation_data"] == "validation.csv"
        assert dumped["test_data"] == "test.csv"


class TestTrainerArtifact:
    """Test suite for TrainerArtifact."""
    
    def test_trainer_artifact_default(self):
        """Test TrainerArtifact default value."""
        artifact = TrainerArtifact()
        
        assert artifact.model == "model"
    
    def test_trainer_artifact_custom_value(self):
        """Test TrainerArtifact with custom model path."""
        artifact = TrainerArtifact(model="models/best_model.pkl")
        
        assert artifact.model == "models/best_model.pkl"


class TestTunerArtifact:
    """Test suite for TunerArtifact."""
    
    def test_tuner_artifact_default(self):
        """Test TunerArtifact default value."""
        artifact = TunerArtifact()
        
        assert artifact.hyperparameters == "hyperparameters.json"
    
    def test_tuner_artifact_custom_value(self):
        """Test TunerArtifact with custom filename."""
        artifact = TunerArtifact(hyperparameters="best_params.yaml")
        
        assert artifact.hyperparameters == "best_params.yaml"


class TestEvaluatorArtifact:
    """Test suite for EvaluatorArtifact."""
    
    def test_evaluator_artifact_default(self):
        """Test EvaluatorArtifact default value."""
        artifact = EvaluatorArtifact()
        
        assert artifact.report == "report.json"
    
    def test_evaluator_artifact_custom_value(self):
        """Test EvaluatorArtifact with custom filename."""
        artifact = EvaluatorArtifact(report="evaluation_report.html")
        
        assert artifact.report == "evaluation_report.html"


class TestPayloadInteraction:
    """Test interaction between different payload types."""
    
    def test_transformer_to_trainer_workflow(self):
        """Test typical workflow from transformer to trainer."""
        # Transformer creates data
        transformer_payload = TransformerPayload(
            train_data=pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
        )
        transformer_payload.set_extra("n_features", 1)
        
        # Trainer would use this data
        assert transformer_payload.has_extra("n_features")
        assert len(transformer_payload.train_data) == 3
    
    def test_payload_serialization(self):
        """Test that payloads can be serialized and deserialized."""
        original = TransformerPayload(
            train_data=pd.DataFrame({"col": [1, 2, 3]})
        )
        original.set_extra("version", "1.0")
        
        # Dump to dict
        dumped = original.model_dump()
        
        # Should contain all fields
        assert "train_data" in dumped
        assert "extra_data" in dumped
        assert dumped["extra_data"]["version"] == "1.0"
    
    def test_multiple_payloads_independence(self):
        """Test that multiple payload instances are independent."""
        payload1 = TransformerPayload()
        payload2 = TransformerPayload()
        
        payload1.set_extra("key1", "value1")
        payload2.set_extra("key2", "value2")
        
        assert payload1.has_extra("key1")
        assert not payload1.has_extra("key2")
        assert payload2.has_extra("key2")
        assert not payload2.has_extra("key1")
