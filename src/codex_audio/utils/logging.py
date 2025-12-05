import logging


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} {levelname} {name} {message}",
        style="{",
    )
    return logging.getLogger(name)
