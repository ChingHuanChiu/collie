
class CollieBaseException(Exception):
    """Base exception for all Collie framework errors."""
    
    def __init__(self, message: str, component: str = None, details: dict = None):
        self.message = message
        self.component = component or self.__class__.__name__.replace('Error', '')
        self.details = details or {}
        
        detailed_message = f"[{self.component}] {message}"
        if self.details:
            detailed_message += f" Details: {self.details}"
        
        super().__init__(detailed_message)


class MLflowConfigurationError(CollieBaseException):
    """Raised when MLflow configuration is invalid."""
    
    def __init__(self, message: str, config_param: str = None, **kwargs):
        details = kwargs.get('details', {})
        if config_param:
            details['config_parameter'] = config_param
        super().__init__(message, component="MLflow Config", details=details)


class MLflowOperationError(CollieBaseException):
    """Raised when MLflow operations fail."""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        super().__init__(message, component="MLflow Operation", details=details)


class OrchestratorError(CollieBaseException):
    """Raised for errors in the orchestrator process."""
    
    def __init__(self, message: str, pipeline_stage: str = None, **kwargs):
        details = kwargs.get('details', {})
        if pipeline_stage:
            details['pipeline_stage'] = pipeline_stage
        super().__init__(message, component="Orchestrator", details=details)


class TransformerError(CollieBaseException):
    """Raised when data transformation fails."""
    
    def __init__(self, message: str, data_type: str = None, **kwargs):
        details = kwargs.get('details', {})
        if data_type:
            details['data_type'] = data_type
        super().__init__(message, component="Transformer", details=details)


class TrainerError(CollieBaseException):
    """Raised when model training fails."""
    
    def __init__(self, message: str, model_type: str = None, **kwargs):
        details = kwargs.get('details', {})
        if model_type:
            details['model_type'] = model_type
        super().__init__(message, component="Trainer", details=details)


class TunerError(CollieBaseException):
    """Raised when hyperparameter tuning fails."""
    
    def __init__(self, message: str, tuning_method: str = None, **kwargs):
        details = kwargs.get('details', {})
        if tuning_method:
            details['tuning_method'] = tuning_method
        super().__init__(message, component="Tuner", details=details)


class EvaluatorError(CollieBaseException):
    """Raised when model evaluation fails."""
    
    def __init__(self, message: str, metric: str = None, **kwargs):
        details = kwargs.get('details', {})
        if metric:
            details['metric'] = metric
        super().__init__(message, component="Evaluator", details=details)


class PusherError(CollieBaseException):
    """Raised when model pushing/deployment fails."""
    
    def __init__(self, message: str, deployment_target: str = None, **kwargs):
        details = kwargs.get('details', {})
        if deployment_target:
            details['deployment_target'] = deployment_target
        super().__init__(message, component="Pusher", details=details)


class ModelFlavorError(CollieBaseException):
    """Raised when model flavor operations fail."""
    
    def __init__(self, message: str, flavor: str = None, **kwargs):
        details = kwargs.get('details', {})
        if flavor:
            details['flavor'] = flavor
        super().__init__(message, component="Model Flavor", details=details)