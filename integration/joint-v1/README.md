# Kivo Tiny v1 reference

`reference.json` 包含 14 条已曝光开发回归的输入、完整提案、1530 维正确性特征和原始 quality logit，以及 18 条 tokenizer 输入/offset/code point→UTF-16 映射参考。它们用于移植核对，不是独立准确率证据。原生客户端尚未执行本文件。

生成命令（需要本地冻结权重、预览包与 Core 运行时）：

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ../DeadlinerCore/build/quickadd-env/bin/python tools/prepare_kivo_fixtures.py \
  --prototype reports/zh-joint-v3/type42-base/prototype.json \
  --bundle artifacts/kivo-tiny-v1-preview \
  --core-runtime artifacts/zh-joint-v3-runtime \
  --output integration/joint-v1/reference.json
```

生成器拒绝覆盖已有文件；重新诊断应输出到新位置后比较。固定长度 96；特殊 token/padding 的 offset 为 `[0,0]`。`codePointToUtf16` 为每个 code point 边界到 UTF-16 边界的映射，长度为 code point 数+1。不要用 Swift 字素数解释这个数组。

对照顺序：tokenizer IDs/mask/offset 精确相同 → 片段位置与业务提案/状态/原因精确相同 → 特征与 logit 数值容差核对 → 最终置信度绝对差 ≤1e-5。参考浮点特征来自 PyTorch，ONNX 原始输出已观察最大差约 1.34e-4；不要求浮点逐位相同。source 从参考 `JOINT_PROTOTYPE` 映射至客户端 `LOCAL`。

完整协议见[接入说明](../../docs/kivo-integration-v1.md)。
