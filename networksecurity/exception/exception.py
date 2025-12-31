import sys
from networksecurity.logging import logger


class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):
        # Call parent Exception class
        super().__init__(error_message)

        self.error_message = error_message

        # Extract traceback details safely
        exc_type, exc_value, exc_tb = error_details.exc_info()

        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = "Unknown"
            self.file_name = "Unknown"

    def __str__(self):
        return (
            f"Error occured in python script name [{self.file_name}] "
            f"line number [{self.lineno}] "
            f"error message [{self.error_message}]"
        )


if __name__ == '__main__':
    try:
        logger.info("Entering try block")
        a = 1 / 0
    except Exception as e:
        raise NetworkSecurityException(e, sys)
