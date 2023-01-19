from datetime import datetime


def get_current_datetime(fmt="%Y%m%d"):
    """Get the current date and/ or time as a string.

    Args:
        fmt (str, optional): Format to return the date/ time in. Defaults to "%Y%m%d".
    Returns:
        str: The current date/ time as a string.
    """
    return datetime.now().strftime(fmt)
