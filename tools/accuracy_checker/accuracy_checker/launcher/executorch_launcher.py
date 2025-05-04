from .launcher import Launcher, ListInputsField
from ..config import PathField, StringField
from pathlib import Path
from collections import OrderedDict

class ExecuTorchLauncher(Launcher):
    __provider__ = 'executorch'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'inputs': ListInputsField(optional=True),
            'model': PathField(is_directory=False, description="Path to model."),
            'device': StringField(choices=('cpu'), optional=True, description="Device: cpu"),
        })
        return parameters

    def __init__(self, config_entry, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        try:
            from torch import from_numpy
            from executorch.runtime import Verification, Runtime, Program, Method # pylint: disable=C0415
        except ImportError as import_error:
            raise ValueError(
                "ExecuTorch isn't installed. Please, install it before using. \n{}".format(import_error.msg)
            )
        self.from_num = from_numpy
        et_runtime = Runtime.get()
        program = et_runtime.load_program(
            Path(str(self.config['model'])),
            verification=Verification.Minimal,
        )
        self._module = program.load_method("forward")
        self._outputs_names = self.config.get("output_names")

        config_inputs = self.config.get("inputs")
        input_shapes = OrderedDict()
        for input_description in config_inputs:
            input_shapes[input_description["name"]] = input_description.get(
                "shape", (self.batch,) + (-1,) * 3
            )
        self._inputs = input_shapes

    @property
    def inputs(self):
        return self._inputs


    @property
    def batch(self):
        return 1

    @property
    def output_blob(self):
        res = 0
        return next(iter(self._outputs_names))


    def predict(self, inputs, metadata=None, **kwargs):
        results = []
        input_name = self.config.get("inputs")[0]["name"]
        for dataset_input in inputs:
            #print(dataset_input['data'].shape)
            pytensor = self.from_num(dataset_input[input_name])
            pytensor = pytensor.contiguous()
            output = self._module.execute((pytensor,))
            #print(self._outputs_names)
            results.append(
                {
                   self._outputs_names[0] : output[0].numpy()
                }
            )
        return results

    def release(self):
        del self._module
