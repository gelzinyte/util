import os
import yaml
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Config(dict):
    # https://martin-thoma.com/configuration-files-in-python/
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, filename):
        with open(str(filename)) as json_file:
            obj = json.loads(json_file.read())

        return cls(obj)

    @classmethod
    def from_yaml(cls, filename):
        with open(str(filename)) as yaml_file:
            cfg = yaml.safe_load(yaml_file)

        return cls(cfg)

    @classmethod
    def load(cls):

        if os.environ.get('UTIL_CONFIG') and Path(os.environ.get('UTIL_CONFIG')).is_file():
            file = Path(os.environ.get('UTIL_CONFIG'))
        # elif (Path.home() / '.util').is_file():
        #     file = Path.home() / '.util'
        else:
            return cls()

        logger.info('Using config file: {}'.format(file))

        config = cls.from_yaml(file)

        return config

    def save(self):
        if os.environ.get('UTIL_CONFIG'):
            file = Path(os.environ.get('UTIL_CONFIG'))
        else:
            file = 'config.txt'

        logger.info('The saved config\'s file: {}'.format(file))

        with open(str(file), 'w') as file:
            yaml.dump(self, file)

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, dict.__repr__(self))
