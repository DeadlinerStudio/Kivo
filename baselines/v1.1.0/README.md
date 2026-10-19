# 已发布基线

权重来源：https://github.com/DeadlinerStudio/DeadlinerCore/releases/tag/quick-add-zh-v1.1.0

quick-add-model.json 保留三个不可变下载 URL、SHA-256、字节长度和生产阈值；不是新候选模型。
本机 artifacts/baselines/v1.1.0 已拷贝对应权重和 vocabulary.json，Git 不跟踪二进制。
另一台机器从 Release 下载三个模型及 vocabulary.json，并把此清单一起放入上述目录。benchmark 会验证 ONNX SHA。
