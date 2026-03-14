"""
    Module for project logging setup. Contains functions returning specific loggers: 
    - css
    - tools
    - config
    - exceptions
    - supercell_worker 
    - collect_worker
"""

__all__ = [
    'get_css_logger',
    'get_tools_logger',
    'get_config_logger',
    'get_exceptions_logger',
    'get_supercell_worker_logger',
    'get_collect_worker_logger'
]

import logging
import logging.config
import os
from multiprocessing import current_process


_log_config = {
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
            "filename": os.path.join(os.path.abspath(__file__).rstrip('logging_.py'), 'csslib.log'),
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
            "handlers": ["console_handler", "file_handler"],
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


def _configure_logging(result_path: str, log_filename: str = "main") -> None:
    """
        Configure logging for the csslib modules.
        
        Args:
            result_path (str): the folder in which logs folder will be created. 
            log_filename (str): a log filename.
    """
    os.makedirs(os.path.join(result_path, "logs"), exist_ok=True)
    _log_config["handlers"]["file_handler"]["filename"] = os.path.join(result_path, "logs", f"{log_filename}.log")
    logging.config.dictConfig(_log_config)


def get_css_logger(result_path: str) -> logging.Logger:
    """
        Configures the css logger and returns it.
    
        Args:
            result_path (str): the folder in which logs folder will be created.
    
        Return: 
            Logger.
    """
    _configure_logging(result_path)
    return logging.getLogger("css")


def get_tools_logger() -> logging.Logger:
    """
        Configures the tools logger and returns it.
    
        Args:
            result_path (str): the folder in which logs folder will be created.
    
        Return: 
            Logger.
    """
    logging.config.dictConfig(_log_config)
    return logging.getLogger("tools")


def get_config_logger() -> logging.Logger:
    """
        Configures the config logger and returns it.
    
        Args:
            result_path (str): the folder in which logs folder will be created.
    
        Return: 
            Logger.
    """
    logging.config.dictConfig(_log_config)
    return logging.getLogger("config")


def get_exceptions_logger() -> logging.Logger:
    """
        Configures the exceptions logger and returns it.
    
        Args:
            result_path (str): the folder in which logs folder will be created.
    
        Return: 
            Logger.
    """
    logging.config.dictConfig(_log_config)
    return logging.getLogger("exceptions")


def get_supercell_worker_logger(result_path: str) -> logging.Logger:
    """
        Configures the supercell worker logger and returns it.
        
        Args:
            result_path (str): the folder in which logs folder will be created.
    
        Return: 
            Logger.
    """
    _configure_logging(result_path, f"supercell_{current_process().name}")
    return logging.getLogger("supercell_worker")


def get_collect_worker_logger(result_path: str) -> logging.Logger:
    """
        Configures the collect worker logger and returns it.
        
        Args:
            result_path (str): the folder in which logs folder will be created.
    
        Return: 
            Logger.
    """
    _configure_logging(result_path, f"collect_{current_process().name}")
    return logging.getLogger("collect_worker")
