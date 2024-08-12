import logging
import sys

# Global flag to control verbosity of the logging
verbose = False

def get_logger(name='BIPeC_detector', pattern='%(asctime)s %(levelname)s %(name)s: %(message)s',
               date_format='%H:%M:%S', handler=None):
    """
    Configures and retrieves a logger instance with specified settings.

    Args:
        name (str): The name of the logger instance.
        pattern (str): The format pattern for log messages.
        date_format (str): The date format to be used in the pattern.
        handler (logging.Handler, optional): The logging handler. If None, defaults to stdout.

    Returns:
        logging.Logger: The configured logger.
    """
    # Retrieve the logger instance
    logger = logging.getLogger(name)
    # Set log level based on the global `verbose` flag
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Configure handler and formatter only if the logger does not already have handlers
    if not logger.handlers:
        # If no handler is provided, use a default stream handler that outputs to stdout
        if handler is None:
            handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(pattern, date_format)
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.addHandler(handler)

        # Prevent the logger from propagating messages to the root logger
        logger.propagate = False

    return logger

# Example usage
if __name__ == "__main__":
    # Set verbose to True if detailed logs are needed
    verbose = True
    log = get_logger()
    log.debug("This is a debug message.")
    log.info("This is an informational message.")
