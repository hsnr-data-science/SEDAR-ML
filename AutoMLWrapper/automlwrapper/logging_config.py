import logging

def setup_logging():
    logger = logging.getLogger('automlwrapper')
    logger.setLevel(logging.DEBUG)  # Set the root level of logging. Adjust as necessary.

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)  # Adjust the level of the console handler

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.propagate = False
