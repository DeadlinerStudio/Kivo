# 公开语料与预训练对照（2026-10-10）

本轮不依赖用户输入记录。MASSIVE-1.1 已有中英文公开语料用于辅助意图/槽位监督；原始标签不直接伪造成 Deadliner 完整提案。只使用官方 train；官方 dev 保留但不参与辅助训练，官方 test 未导入。许可、署名见 data/sources/massive。

## 数据和隔离

- data/zh-g3 / en-g3 分别包含13,351 / 19,935条训练样例；原900条开发集保持不变。
- 新评估集1,200 / 720条，使用各30个不同于新增训练标题的词汇家族；含1,080 / 600条明确可接受请求及120 / 120条应暂缓请求。
- 新评估集是助手编写的模板组合，用于标题词汇泛化检查。标题家族隔离不等于所有语序类型隔离，也不是真实用户测试、人工金标或正式独立盲测。重复家族的1200条不能当1200种独立语言表达。
- 训练增强只作用于TIME中的数字空格、全角空格和制表符，更新Unicode槽位偏移，不改标题内容。保留augmentationParent及来源。NFKC去重会合并全角/普通空格等价变体，不能把生成次数当作独立表达数量。
- data/zh-g3-regression / en-g3-regression单独保存已知故障及原挑战，包括截图原文，不把已知回归当独立评估。
- train-only词表；公开辅助CNN词表可使用官方train。预训练对照可使用其已有分词词表，禁止从dev/test扩充词汇。

## 对照实现

1. v2：已发布权重，使用固定阈值和生产KMP。
2. public CNN：原字符CNN，先以MASSIVE原始60意图/槽位辅助训练3 epoch，再用增强业务数据训练20 epoch。与v2相比同时改变数据和预训练，不能把收益单独归因于公共语料。
3. pretrained teacher：中文uer/roberta-tiny-word-chinese-cluecorpussmall、英文prajjwal1/bert-tiny，微调4 epoch。权重修订SHA、来源模型卡和训练数据SHA写入training-report。仅离线PyTorch对照，没有端侧导出、Tokenizer集成或三格式/真机验证。

全部候选保留v2固定阈值 intent=.85、margin=.25、slot=.9；未根据新评估集降低阈值。完整提案比较调用当前生产KMP解码器；教师WordPiece/SentencePiece槽位经明确字符对齐适配，不把参考答案作为推理输入。空白桥接、BIO展开单独测试。适配器按80条分批处理，避免大JSON峰值内存导致进程退出。

结果和最终结论由 docs/evidence/g3-generalization.json 及本页末尾记录。桌面延迟不能替代手机P95；目前没有修改三个APP下载清单或GitHub Releases。

## 复现入口

在训练工程根目录，安装 `.[train,teacher]`；可复用Core/build/quickadd-env Python环境。

```sh
PYTHONPATH=src python -m quickadd_training.generalization_data
PYTHONPATH=src python -m quickadd_training.validate data/zh-g3
PYTHONPATH=src python -m quickadd_training.validate data/en-g3
PYTHONPATH=src python -m quickadd_training.prepare_adapter --core ../DeadlinerCore --output reports/adapter.json
PYTHONPATH=src python -m quickadd_training.train --locale zh --data-dir data/zh-g3 --output artifacts/zh-g3-public --stem quick-add-zh-g3 --epochs 20 --allow-synthetic --public-data data/sources/massive/zh-CN-train-dev.jsonl --public-epochs 3
PYTHONPATH=src python -m quickadd_training.pretrained_teacher --locale zh --data-dir data/zh-g3 --output artifacts/zh-g3-teacher --epochs 4 --core ../DeadlinerCore --extra-evaluation data/zh-g3-regression/dev.jsonl
```

英文同样执行，locale=en、源文件=en-US-train-dev.jsonl。CNN用manifest生成本地清单后按benchmark --split test、--adapter-metadata reports/adapter.json评估。教师可用--evaluate-only复用已训练权重，修订号从已存报告读取。冻结集SHA在每个data/*-g3/provenance.json；工具不自动发布候选。

## 本轮实际结果

| 候选 | 中文正确 / 可接受 | 英文正确 / 可接受 | 中文 / 英文错误接受 |
|---|---:|---:|---:|
|规则|1080/1080 (100.0%)|0/600 (0.0%)|60 / 450|
|v2|31/1080 (2.9%)|72/600 (12.0%)|0 / 0|
|公开辅助CNN|234/1080 (21.7%)|195/600 (32.5%)|0 / 0|
|预训练离线对照|658/1080 (60.9%)|551/600 (91.8%)|0 / 0|

截图原文及四种中文变体：公开辅助CNN全部正确；中文预训练对照4/5正确，汉字八点变体仍拒绝。已知回归不当作泛化结果。教师离线检查点约53.14MB中文、17.57MB英文，不是三格式导出大小或APK增量。中文上游模型卡没有明确权重许可字段，任何公开分发前需要核实；本轮仅本地对照。

结论：保持CNN候选为实验产物；预训练路线优先进入下一轮端侧结构、量化/蒸馏和转换评估。中文61%仍不足以完成产品验收，不宣称可以替换线上。28项Python测试通过；数据、规格与文件大小检查通过。本轮不修改APP和GitHub模型发布。
