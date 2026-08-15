import sys

from src.logger import get_logger


logger = get_logger(__name__)


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Create a detailed error message containing
    the file name and line number where the error occurred.
    """

    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error occurred: {str(error)}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in Python script "
        f"[{file_name}] at line [{line_number}]: "
        f"{str(error)}"
    )

    return error_message


class CustomException(Exception):
    """
    Custom exception used throughout the project.
    """

    def __init__(
        self,
        error_message: Exception,
        error_detail: sys
    ) -> None:

        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message


if __name__ == "__main__":

    try:
        result = 1 / 0

    except Exception as error:

        logger.exception(
            "Test exception occurred."
        )

        raise CustomException(
            error,
            sys
        ) from error