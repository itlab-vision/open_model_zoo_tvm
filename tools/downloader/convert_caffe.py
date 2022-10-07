import coremltools
import onnxmltools

# Converter Caffe -> CoreML -> ONNX
# Tested with Python 3.7.13. Could not work with higher versions (Python 3.9.13 as an example)

# Update your input name and path for your caffe model
proto_file = '/Users/admin/workspace/open_model_zoo/tools/downloader/public/mobilenet-v2/mobilenet-v2.prototxt' 
input_caffe_path = '/Users/admin/workspace/open_model_zoo/tools/downloader/public/mobilenet-v2/mobilenet-v2.caffemodel'

# Update the output name and path for intermediate coreml model, or leave as is
model_name = "mobilenet-v2"
output_model_path = f"/Users/admin/workspace/scripts/models/OMZ_compiled/{model_name}"
output_coreml_model = f'{output_model_path}/{model_name}.mlmodel'

# Change this path to the output name and path for the onnx model
output_onnx_model = f'{output_model_path}/{model_name}.onnx'



# Convert Caffe model to CoreML 
coreml_model = coremltools.converters.caffe.convert((input_caffe_path, proto_file))

# Save CoreML model
coreml_model.save(output_coreml_model)

# Load a Core ML model
coreml_model = coremltools.utils.load_spec(output_coreml_model)

# Convert the Core ML model into ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model)

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, output_onnx_model)