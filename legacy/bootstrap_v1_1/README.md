# Reproduce quick-add-zh-v1.1.0

Use Python 3.12 on macOS for training, export and actual Core ML validation:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train_model.py --output bootstrap --epochs 14
.venv/bin/python train_model.py --output refinement --resume bootstrap/weights.pt \
 --epochs 6 --learning-rate 0.0003 --slot-loss-weight 2
.venv/bin/python train_model.py --output artifacts --resume refinement/weights.pt \
 --epochs 3 --learning-rate 0.0002 --slot-loss-weight 2 --repeat-loss-weight 3
```

Use the official MindSpore Lite 2.7.0 Linux x64 converter (its converter/lib must be on LD_LIBRARY_PATH):

```sh
converter_lite --fmk=ONNX --modelFile=artifacts/quick-add-zh-v1.onnx --outputFile=artifacts/quick-add-zh-v1
```

Produce the production manifest and actual inference fixtures:

```sh
.venv/bin/python create_model_manifest.py --model-id quick-add-zh-v1.1.0 \
 --vocabulary artifacts/vocabulary.json \
 --base-url https://github.com/DeadlinerStudio/DeadlinerCore/releases/download/quick-add-zh-v1.1.0 \
 --android artifacts/quick-add-zh-v1.onnx --ios artifacts/quick-add-zh-v1.mlmodel \
 --ohos artifacts/quick-add-zh-v1.ms --output artifacts/quick-add-model.json
.venv/bin/python verify_exports.py artifacts
```

`verify_mindspore.cpp` runs actual ARM64 Linux MindSpore Lite outputs from the generated binary inputs. Compile against the matching official runtime:

```sh
g++ -std=c++17 verify_mindspore.cpp -I/path/to/lite/runtime \
 -L/path/to/lite/runtime/lib -L/path/to/lite/runtime/third_party/glog \
 -lmindspore-lite -l:libmindspore_glog.so.0 -o verify-ms
LD_LIBRARY_PATH=/path/to/lite/runtime/lib:/path/to/lite/runtime/third_party/glog ./verify-ms artifacts 19
.venv/bin/python verify_mindspore_outputs.py artifacts
```

KMP opt-in tests use `QUICK_ADD_RELEASE_CHECK_DIR` pointing to these artifacts and verify decoding and actual HTTPS downloads. Training and template test results are bootstrap regression evidence, not an independent real-world benchmark. Torch/OS differences can change floating-point weights; use the published SHA-256 manifest for exact release downloads.
