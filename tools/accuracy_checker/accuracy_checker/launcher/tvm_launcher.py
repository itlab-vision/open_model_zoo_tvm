import os
from collections import OrderedDict

from ..config import NumberField, StringField, BoolField
from .launcher import Launcher


class TVMLauncher(Launcher):
    __provider__ = "tvm"

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                "batch": NumberField(
                    value_type=int,
                    min_value=1,
                    optional=True,
                    description="Batch size.",
                    default=1,
                ),
                "dev_id": NumberField(
                    value_type=int,
                    min_value=0,
                    optional=True,
                    description="Device ID.",
                    default=0,
                ),
                "device": StringField(
                    choices=["cpu", "gpu", "hex"],
                    optional=True,
                    description="Choose between cpu/gpu/hexagon run.",
                    default="cpu",
                ),
                "session": StringField(
                    choices=["local", "remote"],
                    optional=True,
                    description="Choose between local/remote run.",
                    default="local",
                ),
                "vm": BoolField(description="Run with Virtual Machine.", default=False),
            }
        )
        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        try:
            import tvm  # pylint: disable=C0415
            import tvm.rpc  # pylint: disable=C0415
            import tvm.contrib.graph_executor  # pylint: disable=C0415
            import tvm.runtime.vm

            self._tvm = tvm
            self._tvm_rpc = tvm.rpc
            self._tvm_runtime = tvm.contrib.graph_executor
        except ImportError as import_error:
            raise ValueError(
                "TVM isn't installed. Please, install it before using. \n{}".format(
                    import_error.msg
                )
            )
        self.validate_config(config_entry)

        self._device = self.get_value_from_config("device")
        self._session = self.get_value_from_config("session")
        self._vm_executor = self.get_value_from_config("vm")
        self._batch = self.get_value_from_config("batch")

        self._connect_tracker()

        self._get_device()

        self._generate_inputs()

        self._module = self._load_module(config_entry["model"])

        self._generate_outputs()

    def _connect_tracker(self):
        if self._session == "remote":
            print("Using RemoteSession.")
            tracker_host = os.environ["TVM_TRACKER_HOST"]
            tracker_port = int(os.environ["TVM_TRACKER_PORT"])
            key = "android"
            self._tracker = self._tvm_rpc.connect_tracker(tracker_host, tracker_port)
            if self._device == "hex":
                self._remote = self._tracker.request(
                    key,
                    priority=0,
                    session_timeout=200,
                    session_constructor_args=[
                        "tvm.contrib.hexagon.create_hexagon_session",
                        "hexagon-rpc",
                        256 * 1024,
                        os.environ.get("HEXAGON_SIM_ARGS", ""),
                        256 * 1024 * 1024,
                    ],
                )
                func = self._remote.get_function("device_api.hexagon.acquire_resources")
                func()
            else:
                self._remote = self._tracker.request(
                    key, priority=0, session_timeout=200
                )
        else:
            print("Using LocalSession.")
            self._remote = self._tvm_rpc.LocalSession()

    def _get_device(self):
        if self._device == "cpu":
            self._device = self._remote.cpu(self.get_value_from_config("dev_id"))
        if self._device == "gpu":
            self._device = self._remote.cl(self.get_value_from_config("dev_id"))
        if self._device == "hex":
            self._device = self._remote.hexagon(self.get_value_from_config("dev_id"))

    def _generate_inputs(self):
        config_inputs = self.config.get("inputs")
        input_shapes = OrderedDict()
        for input_description in config_inputs:
            input_shapes[input_description["name"]] = input_description.get(
                "shape", (self.batch,) + (-1,) * 3
            )
        self._inputs = input_shapes

    def _load_module(self, model_path):
        model_name = os.path.split(model_path)[-1]
        print(
            "Uploading lib on path {} to {} device.".format(model_path, self._session)
        )
        self._remote.upload(model_path)
        print("Lib has been uploaded to target device {}.".format(self._device))

        if self.get_value_from_config("device") == "hex":
            hex_load_module = self._remote.get_function("tvm.hexagon.load_module")
            lib = hex_load_module(model_name)
            with open(self.config.get("json"), "r") as file:
                graph_json = file.read()
            return self._tvm_runtime.create(graph_json, lib, self._device)
        else:
            lib = self._remote.load_module(model_name)

        if self.get_value_from_config("vm"):
            print("Runtime module: VirtualMachine.")
            return self._tvm.runtime.vm.VirtualMachine(lib, self._device)
        else:
            print("Runtime module: GraphExecutor.")
            return self._tvm_runtime.GraphModule(lib["default"](self._device))

    def _generate_outputs(self):
        if self._vm_executor:
            self._outputs_names = self.config.get("outputs")
        else:
            self._outputs_names = list(range(self._module.get_num_outputs()))

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
        input_name = self.config.get("inputs")[0]["name"]
        for batch_input in inputs:
            if self._vm_executor:
                self._module.set_input("main", batch_input[input_name])
                self._module.invoke_stateful("main")
                output = self._module.get_outputs()
                results.append(
                    {
                        output_name: output_value.asnumpy()
                        for output_name, output_value in zip(
                            self._outputs_names, output
                        )
                    }
                )
            else:
                for input_name, input_data in batch_input.items():
                    self._module.set_input(input_name, input_data)
                self._module.run()
                if self.config.get("adapter") == "bert_question_answering":
                    results.append(
                        {
                            str(output_name): self._module.get_output(output_name).asnumpy()
                            for output_name in self._outputs_names
                        }
                    )
                else:
                    results.append(
                        {
                            output_name: self._module.get_output(output_name).asnumpy()
                            for output_name in self._outputs_names
                        }
                    )
        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError("TVM Launcher does not support async mode yet")

    def release(self):
        del self._module
        del self._device
        del self._remote
        del self._session
        del self._vm_executor
        del self._batch
