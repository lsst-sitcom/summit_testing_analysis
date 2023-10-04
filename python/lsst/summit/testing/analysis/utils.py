import logging


def create_logger(name: str) -> logging.Logger:
    """
    Create a logger object with the specified name and returns it.

    Parameters
    ----------
    name : str
        The name of the logger object.

    Returns
    -------
    logger : logging.Logger
        The logger object with the specified name.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter.datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler()
    handler.setLevel(logging.ERROR)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    return logger
