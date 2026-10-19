// Real MindSpore Lite CPU execution of the same binary tensors used by ONNX/Core ML.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "include/api/context.h"
#include "include/api/model.h"

int main(int argc, char **argv) {
  if (argc != 4) return 1;
  const std::string root = argv[1];
  const int count = std::stoi(argv[3]);
  auto context = std::make_shared<mindspore::Context>();
  context->SetThreadNum(3);
  auto cpu = std::make_shared<mindspore::CPUDeviceInfo>();
  cpu->SetEnableFP16(false);
  context->MutableDeviceInfo().push_back(cpu);
  mindspore::Model model;
  if (model.Build(argv[2], mindspore::kMindIR, context) != mindspore::kSuccess) return 2;
  for (int row = 0; row < count; ++row) {
    auto inputs = model.GetInputs();
    for (auto &input : inputs) {
      if (input.Name().find('/') != std::string::npos) return 3;
      std::ifstream stream(root + "/input-" + std::to_string(row) + "-" + input.Name() + ".bin", std::ios::binary);
      if (!stream || input.MutableData() == nullptr) return 4;
      stream.read(static_cast<char *>(input.MutableData()), input.DataSize());
      if (static_cast<size_t>(stream.gcount()) != input.DataSize() || stream.peek() != EOF) return 5;
    }
    std::vector<mindspore::MSTensor> outputs;
    if (model.Predict(inputs, &outputs) != mindspore::kSuccess) return 6;
    for (const auto &output : outputs) {
      if (output.Name().find('/') != std::string::npos || output.DataType() != mindspore::DataType::kNumberTypeFloat32) return 7;
      std::ofstream stream(root + "/mindspore-" + std::to_string(row) + "-" + output.Name() + ".bin", std::ios::binary);
      stream.write(static_cast<const char *>(output.Data().get()), output.DataSize());
      if (!stream.good()) return 8;
      if (row == 0) {
        std::cout << output.Name();
        for (auto dimension : output.Shape()) std::cout << " " << dimension;
        std::cout << "\n";
      }
    }
  }
  std::cout << "MindSpore actual examples: " << count << "\n";
  return 0;
}
