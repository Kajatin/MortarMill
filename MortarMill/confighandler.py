import logging
import configparser


class Singleton(type):
    """ Metaclass used to create a singleton class instance """

    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigHandler(metaclass=Singleton):
    """description of class"""

    def __init__(self, filename):
        self.filename = filename
        
        # parse configuration file
        self.config = configparser.ConfigParser()
        self.config.read(self.filename)


    def getLogLevel(self):
        levels = {
            'debug':logging.DEBUG,
            'info':logging.INFO,
            'warning':logging.WARNING,
            'error':logging.ERROR,
            'critical':logging.CRITICAL
        }

        return levels.get(self.config['DEFAULT'].get('log_level'), logging.DEBUG)