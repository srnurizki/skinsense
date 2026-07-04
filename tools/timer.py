# Import Libraries
import time
import logging
import functools

# Instantiate Logger
logger = logging.getLogger(__name__)

# Timer Wrapper
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f'{func.__name__} completed in {end - start:.3f}s')
        return result
    return wrapper