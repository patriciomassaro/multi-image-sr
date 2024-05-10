import logging
import os
def initialize_logger(log_name: str) -> logging.Logger:
    """
    Initializes the logger
    :param log_name:
    :return:
    """
    # check if the logs directory exists
    print(log_name)
    if not os.path.exists(os.path.dirname(log_name)):
        os.makedirs(os.path.dirname(log_name))
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(log_name)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger