"""Module with csslib exceptions classes."""

__all__ = []

from pydantic import ValidationError
from csslib.logging_ import get_exceptions_logger

logger = get_exceptions_logger()


class CSSlibException(Exception):
    """
        Abstract class for CSSlib exceptions. 
    """
    def __init__(self, message):
        """
            Initialization method of the CSSlibException abstract class. Stores the information about an exception in the .log file.

            Args:
                message (str): Text description of the error.
        """
        self.message = message
        super().__init__(self.message)
        logger.critical('', exc_info=self, stack_info=True)


class ConfigurationError(CSSlibException):
    """
        Raised when there is an issue with the configuration file.
    """
    def __init__(self, message="Configuration issue"):
        """
            Initialization method of the ConfigurationError class.

            Args:
                message (str, optional): Text description of the error. Defaults to "Configuration issue".
        """
        super().__init__(message)


class ConfigurationNotFoundError(CSSlibException):
    """
        Raised when configuration file is not found.
    """
    def __init__(self, message="Configuration file is not found."):
        """
            Initialization method of the ConfigurationNotFoundError class.

            Args:
                message (str, optional): Text description of the error. Defaults to "Configuration file is not found.".
        """
        super().__init__(message)


class ResultsFolderExistError(CSSlibException):
    """
        Raised when the results folder exists and rewrite_results flag in CSS class initialisation method is False.
    """
    def __init__(self, message=None):
        """
            Initialization method of the ResultsFolderExistError class. 

            Args:
                message (str, optional): Text description of the error. Defaults to None.
        """
        super().__init__(message if message is not None else \
                'Results folder is exists. Set `rewrite_results` flag as True to rewrite results.')


class DataLoaderError(CSSlibException):
    """
        Raised when errors in the DataLoader class occurs.
    """
    def __init__(self, message='Unexpected error in the DataLoader class is occured.'):
        """
            Initialization method of the DataLoaderError class. 

            Args:
                message (str, optional): Text description of the error.
        """
        super().__init__(message)


class StructureNotFoundError(CSSlibException):
    """
        Raised when structure .cif file is not found.
    """
    def __init__(self, message="Structure is not found."):
        """
            Initialization method of the StructureNotFoundError class.

            Args:
                message (str, optional): Text description of the error. Defaults to "Structure is not found.".
        """
        super().__init__(message)


class CalculatorError(CSSlibException):
    """
        Raised when errors in the Calculator class occurs.
    """
    def __init__(self, message="Unexpected error in the Calculator class is occured."):
        """
            Initialization method of the CalculatorError class.

            Args:
                message (str, optional): Text description of the error. Defaults to "Unexpected error in the Calculator class is occured.".
        """
        super().__init__(message)


class RemoteConnectionError(CSSlibException):
    """
        Raised when errors in the RemoteConnection class occurs.
    """
    def __init__(self, message="Unexpected error in the RemoteConnection class is occured."):
        """
            Initialization method of the RemoteConnectionError class.

            Args:
                message (str, optional): Text description of the error. Defaults to "Unexpected error in the Worker class is occured.".
        """
        super().__init__(message)


class WorkerError(CSSlibException):
    """
        Raised when errors in the Worker class occurs.
    """
    def __init__(self, message="Unexpected error in the Worker class is occured."):
        """
            Initialization method of the WorkerError class.

            Args:
                message (str, optional): Text description of the error. Defaults to "Unexpected error in the Worker class is occured.".
        """
        super().__init__(message)


class RemoteWorkerError(CSSlibException):
    """
        Raised when errors in the RemoteWorker class occurs.
    """
    def __init__(self, message="Unexpected error in the RemoteWorker class is occured."):
        """
            Initialization method of the RemoteWorkerError class.

            Args:
                message (str, optional): Text description of the error. Defaults to "Unexpected error in the RemoteWorker class is occured.".
        """
        super().__init__(message)


class VisualizationError(CSSlibException):
    """
        Raised when errors in the Visualization module occurs.
    """
    def __init__(self, message='Unexpected error in the Visualization class is occured.'):
        """
            Initialization method of the VisualizationError class. 

            Args:
                message (str, optional): Text description of the error.
        """
        super().__init__(message)


def catch_config_errors(e: ValidationError) -> str:
    """
        Function for processing of the pydantic scheme exceptions. For internal library use! 

        Args:
            e (ValidationError): pydantic validation error object

        Return:
            str: the message with full description of the config fields mistakes. 
    """
    message = '\nThe following configuration errors were detected:\n'
    for el in e.errors():
        loc = el['loc']
        match el['type']:
            case 'string_type':
                message += f'  - Incorrect parameter type at position - {str(loc)}. String is required;\n'
            case 'int_parsing':
                message += f'  - Incorrect parameter type at position - {str(loc)}. Integer is required;\n'
            case 'missing':
                message += f'  - Missing parameter `{loc[-1]}` at position - {str(loc[:-1])};\n' if len(loc) > 1 \
                        else f'  - Missing parameter `{loc[-1]}` at top level of json;\n' 
            case 'extra_forbidden':
                message += f'  - Unknown parameter `{loc[-1]}` at position - {str(loc[:-1])};\n' if len(loc) > 1 \
                        else f'  - Unknown parameter `{loc[-1]}` at top level of json;\n'
            case _:
                message += f'  - Undocumented exception occured at position - {str(loc)}. Exception type - {el["type"]};\n'
    message += 'Check `get_available_config_fields` function of `csslib.config`.\n'
    return message
