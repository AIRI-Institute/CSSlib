"""Module for project logging setup. Contains functions returning specific loggers: main logger, supercell_worker logger, collect_worker logger."""

__all__ = [
    'get_main_logger',
    'get_supercell_worker_logger',
    'get_collect_worker_logger'
]

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
        "css": {
            "handlers": ["console_handler", "file_handler"],
            "level": "DEBUG",
            "propagate": False
        },
        "tools": {
            "handlers": ["console_handler", "file_handler"],
            "level": "DEBUG",
            "propagate": False
        },
        "config": {
            "handlers": ["console_handler"],
            "level": "DEBUG",
            "propagate": False
        },
        "exceptions": {
            "handlers": ["file_handler"],
            "level": "WARNING",
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
        },
    }
}


def configure_logging(result_path: str, log_filename: str = "main") -> None:
    """
        Configure logging for the main process and the supercell and collect workers.
        
        Args:
            result_path (str): the folder in which logs folder will be created. 
            log_filename (str): a log filename.
        
        Return:
            None.
    """
    os.makedirs(os.path.join(result_path, "logs"), exist_ok=True)
    log_config["handlers"]["file_handler"]["filename"] = os.path.join(result_path, "logs", f"{log_filename}.log")
    logging.config.dictConfig(log_config)


def get_css_logger(result_path: str) -> logging.Logger:
    """
        Configures the main logger and returns it.
    
        Args:
            result_path (str): the folder in which logs folder will be created.
    
        Return: 
            Logger.
    """
    configure_logging(result_path)
    return logging.getLogger("css")


def get_supercell_worker_logger(result_path: str) -> logging.Logger:
    """
        Configures the supercell worker logger and returns it.
        
        Args:
            result_path (str): the folder in which logs folder will be created.
    
        Return: 
            Logger.
    """
    configure_logging(result_path, f"supercell_{current_process().name}")
    return logging.getLogger("supercell_worker")


def get_collect_worker_logger(result_path: str) -> logging.Logger:
    """
        Configures the collect worker logger and returns it.
        
        Args:
            result_path (str): the folder in which logs folder will be created.
    
        Return: 
            Logger.
    """
    configure_logging(result_path, f"collect_{current_process().name}")
    return logging.getLogger("collect_worker")
