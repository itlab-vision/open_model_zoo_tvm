import os
from collections import OrderedDict

from ..config import NumberField
from .launcher import Launcher


class TVMLauncher(Launcher):
    __provider__ = 'tvm'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'batch': NumberField(value_type=int, min_value=1, optional=True, description="Batch size.", default=1),
            'dev_id': NumberField(value_type=int, min_value=0, optional=True, description="Device ID.", default=0),
        })
        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        try:
            import tvm  # pylint: disable=C0415
            import tvm.rpc  # pylint: disable=C0415
            import tvm.contrib.graph_executor  # pylint: disable=C0415
            self._tvm = tvm
            self._tvm_rpc = tvm.rpc
            self._tvm_runtime = tvm.contrib.graph_executor
        except ImportError as import_error:
            raise ValueError("TVM isn't installed. Please, install it before using. \n{}".format(import_error.msg))
        self.validate_config(config_entry)

        self._remote_session = self._tvm_rpc.LocalSession()
        self._device = self._remote_session.cpu(self.get_value_from_config('dev_id'))
        self._module = self._load_module(config_entry['model'])

        self._batch = self.get_value_from_config('batch')
        self._generate_inputs()
        self._outputs_names = list(range(self._module.get_num_outputs()))

    def _generate_inputs(self):
        config_inputs = self.config.get('inputs')
        input_shapes = OrderedDict()
        for input_description in config_inputs:
            input_shapes[input_description['name']] = input_description.get('shape', (self.batch,) + (-1,) * 3)
        self._inputs = input_shapes

    def _load_module(self, model_path):
        model_name = os.path.split(model_path)[-1]
        self._remote_session.upload(model_path)
        lib = self._remote_session.load_module(model_name)
        return self._tvm_runtime.GraphModule(lib["default"](self._device))

    @property
    def inputs(self):
        return self._inputs

    @property
    def batch(self):
        return self._batch

    @property
    def output_blob(self):
        return next(iter(self._outputs_names))

    def fit_to_input(self, data, layer_name, layout, precision):
        data = super().fit_to_input(data, layer_name, layout, precision)
        return self._tvm.nd.array(data, self._device)

    def predict(self, inputs, metadata=None, **kwargs):
        results = []
        for batch_input in inputs:
            for input_name, input_data in batch_input.items():
                self._module.set_input(input_name, input_data)
            self._module.run()
            results.append({output_name: self._module.get_output(output_name).asnumpy()
                            for output_name in self._outputs_names})
        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError('TVM Launcher does not support async mode yet')

    def release(self):
        del self._module
        del self._device
        del self._remote_session
