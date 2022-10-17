'''
Module for performance measures of search process.
'''
from typing import Callable, Any
import time
import os
import psutil


# performance measurement
def performance(func: Callable) -> Callable:
    '''
    Decorator that fixes time of the work of a function.
    '''
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        work_time = round((time.time() - start_time), 3)
        print(f'Search worked {work_time} seconds')
        return result
    return wrapper


# profile
# by this stackoverflow discussion
# https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
def process_memory() -> float:
    '''
    Get consumed memory.
    '''
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def profile(func: Callable) -> Callable:
    '''
    Decorator that fixes memory usage of the function.
    '''
    def wrapper(*args, **kwargs) -> Any:
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        final_time = mem_after - mem_before
        print(f'Consumed memory: {final_time}')
        return result
    return wrapper