import logging
import logging.config
import os
from multiprocessing import current_process


log_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "console_formatter": {
            "format": "%(levelname)s - %(message)s"
        },
        "file_formatter": {
            "format": "[%(asctime)s] %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console_handler": {
            "level": "INFO",
            "formatter": "console_formatter",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "level": "DEBUG",
            "formatter": "file_formatter",
            "class": "logging.FileHandler",
            "filename": "",
            "mode": "a",
            "encoding": "utf-8"
        }
    },
    "loggers": {
        "main": {
            "handlers": ["console_handler", "file_handler"],
            "level": "DEBUG",
            "propagate": False
        },
        "supercell_worker": {
            "handlers": ["file_handler"],
            "level": "DEBUG",
            "propagate": False
        },
        "collect_worker": {
            "handlers": ["file_handler"],
            "level": "DEBUG",
            "propagate": False
        }
        }
    }


def configure_logging(result_path: str, log_filename: str = "main") -> None:
    """
    Configure logging for the main process and the supercell and collect workers.
    :param result_path: Logs will be saved in result_path/logs.
    :param log_filename: Log filename.
    :return: None.
    """
    os.makedirs(os.path.join(result_path, "logs"), exist_ok=True)
    log_config["handlers"]["file_handler"]["filename"] = os.path.join(result_path, "logs", f"{log_filename}.log")
    logging.config.dictConfig(log_config)


def get_main_logger(result_path: str) -> logging.Logger:
    """
    Get the main logger.
    :param result_path: Logs will be saved in result_path/logs.
    :return: Logger.
    """
    configure_logging(result_path)
    return logging.getLogger("main")


def get_supercell_worker_logger(result_path: str) -> logging.Logger:
    """
    Get the supercell worker logger.
    :param result_path: Logs will be saved in result_path/logs.
    :return: Logger.
    """
    configure_logging(result_path, f"supercell_{current_process().name}")
    return logging.getLogger("supercell_worker")


def get_collect_worker_logger(result_path: str) -> logging.Logger:
    """
    Get the collect worker logger.
    :param result_path: Logs will be saved in result_path/logs.
    :return: Logger.
    """
    configure_logging(result_path, f"collect_{current_process().name}")
    return logging.getLogger("collect_worker")
