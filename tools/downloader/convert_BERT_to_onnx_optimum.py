from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering

'''
This script should be finalized in order to correctly initialize weights and biases in all cases
'''

# Load the model from the hub and export it to the ONNX format
model = ORTModelForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking", from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking")

# Save the converted model
model.save_pretrained("/Users/admin/workspace/auxillary/tvm-samples/engagements/amd_mlperf_july15/bert/data/bert-large-cased-whole-word-masking-finetuned-squad/optimum_model")
tokenizer.save_pretrained("/Users/admin/workspace/auxillary/tvm-samples/engagements/amd_mlperf_july15/bert/data/bert-large-cased-whole-word-masking-finetuned-squad/optimum_model/tokenizer")