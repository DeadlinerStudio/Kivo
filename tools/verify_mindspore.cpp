// Build with official MindSpore Lite runtime/include and -lmindspore-lite.
// Run on ARM64 Linux; output tensors are passed to the same KMP release acceptance test.
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "include/api/model.h"
#include "include/api/context.h"

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) return 1;
  const std::string root = argv[1];
  const int examples = std::stoi(argv[2]);
  auto context = std::make_shared<mindspore::Context>();
  context->SetThreadNum(2);
  auto cpu = std::make_shared<mindspore::CPUDeviceInfo>();
  cpu->SetEnableFP16(false);
  context->MutableDeviceInfo().push_back(cpu);
  mindspore::Model model;
  if (model.Build(root + "/" + (argc > 3 ? std::string(argv[3]) : "quick-add-zh-v1") + ".ms", mindspore::kMindIR, context) != mindspore::kSuccess) return 2;
  auto inputs = model.GetInputs();
  if (inputs.size() != 2) return 3;
  for (int index = 0; index < examples; ++index) {
    for (auto &input : inputs) {
      const std::string suffix = input.Name() == "input_ids" ? "ids" : input.Name() == "attention_mask" ? "mask" : "";
      if (suffix.empty() || input.Shape() != std::vector<int64_t>{1,128} || input.DataType() != mindspore::DataType::kNumberTypeInt32) return 4;
      std::ifstream file(root + "/checks/" + std::to_string(index) + "-" + suffix + ".bin", std::ios::binary);
      if (!file.read(static_cast<char*>(input.MutableData()), input.DataSize())) return 5;
    }
    std::vector<mindspore::MSTensor> outputs;
    if (model.Predict(inputs, &outputs) != mindspore::kSuccess || outputs.size() != 2) return 6;
    for (auto &output : outputs) {
      const auto shape = output.Name() == "intent_logits" ? std::vector<int64_t>{1,3} :
                         output.Name() == "slot_logits" ? std::vector<int64_t>{1,128,9} : std::vector<int64_t>{};
      if (shape.empty() || output.Shape() != shape || output.DataType() != mindspore::DataType::kNumberTypeFloat32) return 7;
      std::ofstream file(root + "/checks/" + std::to_string(index) + "-ms-" + output.Name() + ".bin", std::ios::binary);
      file.write(static_cast<const char*>(output.Data().get()), output.DataSize());
      if (!file) return 8;
      std::cout << index << " " << output.Name() << " float32 ";
      for (auto dimension : shape) std::cout << dimension << " ";
      std::cout << "\n";
    }
  }
}
