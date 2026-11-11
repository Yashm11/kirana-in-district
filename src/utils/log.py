"""One-line logging setup shared by every stage."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints time, level and message to the terminal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
