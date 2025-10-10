class MLflowConfigurationError(Exception):
    """Raised when MLflow configuration is invalid."""
    pass


class MLflowOperationError(Exception):
    """Raised when MLflow operations fail."""
    pass


class OrchestratorError(Exception):
    """Raised for errors in the orchestrator process."""
    pass


class TransformerError(Exception):
    """Raised when data transformation fails."""
    pass


class TrainerError(Exception):
    """Raised when model training fails."""
    pass


class TunerError(Exception):
    """Raised when hyperparameter tuning fails."""
    pass


class EvaluatorError(Exception):
    """Raised when model evaluation fails."""
    pass


class PusherError(Exception):
    """Raised when model pushing/deployment fails."""
    pass