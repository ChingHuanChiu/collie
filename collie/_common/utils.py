import logging


def get_logger() -> logging.Logger:
    """
    Return a logger that logs messages with severity level info or higher.
    
    Returns:
        A logger that logs messages with severity level info or higher.
    """
    logging.basicConfig(level=logging.INFO)  
    logger = logging.getLogger(__name__)
    return logger
