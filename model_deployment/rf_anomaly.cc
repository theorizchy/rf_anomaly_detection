#include "model.h"
#include <cstdio>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: rf_anomaly <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }


int main(int argc, char* argv[]) {
  // Load model from constant byte array
  std::unique_ptr<tflite::FlatBufferModel> model_ptr =
      tflite::FlatBufferModel::BuildFromBuffer(model, sizeof(model));
  TFLITE_MINIMAL_CHECK(model_ptr != nullptr);

  // Build interpreter with the loaded model
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  TFLITE_MINIMAL_CHECK(tflite::InterpreterBuilder(*model_ptr, resolver)(&interpreter) == kTfLiteOk);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // // Get indices of input and output tensors
  // int input_tensor_index = interpreter->inputs()[0];
  // int output_tensor_index = interpreter->outputs()[0];
  // // Fill input buffers (replace this with your actual input data)
  // float* input_data = interpreter->typed_input_tensor<float>(input_tensor_index);
  // // Your code to fill input_data with real-time data goes here
  // // Run inference
  // TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  // // Read output buffers
  // float* output_data = interpreter->typed_output_tensor<float>(output_tensor_index);
  // // Output data interpretation logic goes here
  // // Example: Printing output
  // printf("Output: %f\n", *output_data);


  // Read input data from text file
  std::ifstream input_file("train_data/X_test_scaled.txt");
  if (!input_file.is_open()) {
    std::cerr << "Error opening input file." << std::endl;
    return 1;
  }

  std::vector<float> input_data;
  float value;
  while (input_file >> value) {
    input_data.push_back(value);
  }
  input_file.close();

  // Set input tensor
  float* input_tensor_ptr = interpreter->typed_input_tensor<float>(0);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_tensor_ptr[i] = input_data[i];
  }

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // Get output tensor
  float* output_data = interpreter->typed_output_tensor<float>(0);

  // Convert output to binary predictions
  int prediction = (*output_data > 0.5) ? 1 : 0;

  // Output the prediction
  std::cout << "Prediction: " << prediction << std::endl;

  return 0;
}