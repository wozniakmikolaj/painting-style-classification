import logging.config
import json
import os

logging_config_path = os.path.join(os.getcwd(), 'configs', 'logging_config.json')

with open(logging_config_path, 'r') as f:
    config = json.load(f)
    logging.config.dictConfig(config)
    logging.captureWarnings(True)


def get_logger(name: str):
    """Logs a message
    Args:
    name(str): name of logger
    """
    logger = logging.getLogger(name)
    return logger
