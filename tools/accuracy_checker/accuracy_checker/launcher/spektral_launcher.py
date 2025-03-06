import os

from .launcher import Launcher

import importlib

import numpy as np

from collections import OrderedDict
from ..utils import string_to_tuple

try:
    import spektral  # pylint: disable=C0415
except ImportError as import_error:
    raise ValueError(
        "Spektral isn't installed. Please, install it before using. \n{}".format(
            import_error.msg))
try:
    import tensorflow as tf  # pylint: disable=C0415
except ImportError as import_error:
    raise ValueError(
        "Tensorflow isn't installed. Please, install it before using. \n{}".format(
            import_error.msg))


class SpektralLauncher(Launcher):
    __provider__ = "spektral"

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)

        self.validate_config(config_entry)

        self._batch = self.get_value_from_config("batch")

        input_shapes = []

        for input_config in self.config['inputs']:
            input_shape = input_config['shape']
            input_shape = string_to_tuple(input_shape, casting_type=int)
            input_shapes.append((input_config['name'], (self._batch, *input_shape)))
        
        self._inputs = OrderedDict(input_shapes)

        self.output_names = self.get_value_from_config('output_names') or ['output']

        self._model = self._load_model(config_entry["model"])

    def _load_model(self, model_path):
        if str(model_path).endswith('keras') == True:
            model = tf.keras.saving.load_model(model_path, compile=True)

            return model

        else:
            print("Only .keras model save file type is supported")
            raise ValueError('Only .keras model save file type is supported')

    @property
    def batch(self):
        return self._batch

    def predict(self, inputs, metadata=None, **kwargs):
        class DummyDataset(spektral.data.Dataset):
            def __init__(self, _graph, **kwargs):
                self.graph = _graph

                super().__init__(**kwargs)

            def download(self):
                pass

            def read(self):
                output = []
                output.append(self.graph)
                return output
        
        result = []

        for batch_input in inputs:
            input_loader = spektral.data.SingleLoader(DummyDataset(batch_input['input'][0]))

            slice_input, _ = input_loader.__next__()

            result.append({'output': tf.math.argmax(self._model(slice_input, training=False), 1)})

        return result

    def release(self):
        del self._model
        del self._batch

    @property
    def output_blob(self):
        return next(iter(self.output_names))

    @property
    def inputs(self):
        return self._inputs