import os
import sys


def get_home_path():
    """
    Returns the local users home path
    """
    return os.path.expanduser('~')


def get_local_models_root():
    """
    returns the path to the default location for local atlas database
    """
    return os.path.join(get_home_path(), '.ts2d', 'models')

def get_package_root():

    """
    returns the path to the package's root folder
    """
    import ts2d
    return os.path.dirname(ts2d.__file__)

def get_package_data_root():
    """
    returns the path to the data folder within the installed ts2d package
    """
    return os.path.join(get_package_root(), 'data')
