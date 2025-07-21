from datetime import datetime,timezone,timedelta
import pytz
import os
import pandas as pd
from typing import Sequence


def get_inv_dict(dct: dict) -> dict:
    return {dct[k]:k for k in dct.keys()}


def is_daylight_saving(dt: datetime, tz='CET') -> bool:
    """ Checks if a given date is in daylight saving time.
    Returns True if the date is in daylight saving time, False otherwise.
    """
    tz = pytz.timezone(tz)
    date = tz.localize(dt)
    return date.dst() != timedelta(0)


def pandas_io_handler(filename: str, how: str = 'read', obj: pd.DataFrame = None, evaluate: bool = True, **kwargs) -> pd.DataFrame:
    file_format = os.path.splitext(filename)[1][1:]
    if file_format=='pkl':
        file_format='pickle'
    if how=='read':
        eval_str = 'pd.read_{}("{}", **kwargs)'.format(file_format, filename)
        if evaluate:
            return eval(eval_str)
        return eval_str
    if how=='save':
        if obj is None:
            return False
        eval_str = 'obj.to_{}'.format(file_format)
        if evaluate:
            return eval(eval_str+'(filename, **kwargs)')
        return eval_str
    
def get_columns_from_parquet_file(parquet_filepath):
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(parquet_filepath)
    cols = parquet_file.schema_arrow.names
    return cols
    
def _flatten_list_gen(L: Sequence) -> None:
    for item in L:
        if isinstance(item, str):
            yield item
        else:
            try:
                yield from flatten(item)
            except TypeError:
                yield item
                

def flatten(L: Sequence) -> list:
    return list(_flatten_list_gen(L))
                
                
from collections.abc import MutableMapping
def _flatten_dict_gen(d: dict, parent_key, sep=False) -> None:
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key and sep else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.') -> dict:
    return dict(_flatten_dict_gen(d, parent_key, sep))
    
    
def filter_list(list_to_filter: list, list_to_look: list, return_indexes: bool =False) -> list:
    """
    Returns the filtered list(or indexes, if return_indexes) by looking for the list_to_look elements in sequential order into list_to_filter... once found, list_to_filter = list_to_filter[found_index:]
    EG: list_to_filter = [1,2,5,4,7], list_to_look=[1,2,7,4]
    filtered_list = [1,2,7]... because it will look for any element in sequential order, since there's no 4 after 7 in list_to_look, 4 is not found in the sublist after 7...
    """
    last_i = -1
    final_list = []
    for i,el in enumerate(list_to_look):    
        try:
            curr_i = list_to_filter.index(el)
            last_i+=curr_i+1
            final_list.append(last_i if return_indexes else list_to_filter[curr_i])
            list_to_filter = list_to_filter[curr_i+1:]
        except ValueError:
            continue
    return final_list
    
def map_to_count(l: list, count_unique: bool=True) -> dict[str, int] or int:
    if count_unique:
        return dict(sorted(list(zip(*np.unique(l, return_counts=True))), key=lambda e: e[1], reverse=True))
    return len(l)
    
def get_package_root():
    from pathlib import Path
    
    current_file = __file__
    path = Path(current_file).resolve()
    print(path)
    while not (path / '.git').exists() and path != path.parent:
        path = path.parent
    return path
    
"""
def get_package_root():
    from pathlib import Path
    try:
        path = Path(__file__).resolve()
        print(path)
    except NameError:
        # __file__ is not defined in Jupyter
        path = Path(os.getcwd()).resolve()

    while (path / '__init__.py').exists():
        path = path.parent
    return path
"""