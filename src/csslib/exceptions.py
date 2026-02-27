from pydantic import ValidationError


class CSSlibException(Exception):
    '''
        Abstract class for CSSlib exceptions. 
    '''
    pass    


class ConfigurationError(CSSlibException):
    """
        Raised when there is an issue with the configuration file.
    """
    def __init__(self, message="Configuration issue"):
        self.message = message
        super().__init__(self.message)


class ConfigurationNotFoundError(CSSlibException):
    """
        Raised when configuration file is not found.
    """
    def __init__(self, message="Configuration file is not found."):
        self.message = message
        super().__init__(self.message)


class ResultsFolderExistError(CSSlibException):
    """
        Raised when results folder is exists.
    """
    def __init__(self, message=None):
        self.message = message if message is not None else \
                "Results folder is exists. Set `rewrite_results` flag as True to rewrite results."
        super().__init__(self.message)


class StructureNotFoundError(CSSlibException):
    """
        Raised when structure .cif file is not found.
    """
    def __init__(self, message="Structure is not found."):
        self.message = message
        super().__init__(self.message)


def catch_config_errors(e: ValidationError):
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
