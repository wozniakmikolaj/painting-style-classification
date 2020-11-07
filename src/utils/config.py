"""Config class"""

import json


class Config:
    """Config class which contains data, train and models hyperparameters"""

    def __init__(self, paths, data, train, model):
        self.paths = paths
        self.data = data
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.paths, params.data, params.train, params.model)


class HelperObject(object):
    """Helper class to convert json into Python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)
