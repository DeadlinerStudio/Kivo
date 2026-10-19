# Kivo

Deadliner 快速添加端侧模型的训练、评测与导出工程。Kivo 负责从自然语言中提取任务与习惯字段，日期、执行日与目标计次的业务归一化由 DeadlinerCore 负责。

仓库：<https://github.com/DeadlinerStudio/Kivo>。本地目录与 Python 包名 `quickadd_training` 保留，已有脚本仍可使用。

**当前版本是实验原型，尚未接入客户端。** 主合成留出正确接受覆盖 TASK 97.56%、HABIT 92.44%，但旧表达回放仅 63.54%，尚未证明跨表达 70% 目标。详见[联合原型报告](docs/task-habit-joint-prototype-v3.md)与[客户端接入说明](docs/kivo-integration-v1.md)。权重与完整运行报告不进入 Git；本地可生成带 SHA 校验的接入预览包。

## 现状与边界

- `legacy/bootstrap_v1_1` 保留旧合成训练、19 条回归及三端转换工具；它不是独立准确率证据。
- `data/dev.jsonl` 为 35 条助手编写的开发挑战，包括截图故障、标题中的时间词、否定、条件、多任务、非法/未完成时间。期望需人工复核。
- 根目录 `data/train.jsonl` 与全部 `test.jsonl` 为空；正式人工审核训练和最终评估仍无可用数据。分语言目录已增加合成实验训练集。不能把 dev 再拆一部分声称盲测。
- 新训练入口只从 train 构建词表，不读取 test 或旧 acceptance。
- `benchmark` 使用实际 ONNX logits，然后调用 **DeadlinerCore 生产 QuickAddParser 和 QuickAddCharacterCodec**，按全部业务字段判定；hybrid 复现共享 QuickAddLocalRecognizer 的非空模型优先/空结果回退规则。
- Android/iOS/Harmony UI 对非 ready 结果的采用策略可能不同；本报告对应共享 KMP 算法，不是三个 UI 的端到端结果。发布需各端真机验证。

## 开始

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[train,teacher,coreml]'
.venv/bin/python -m quickadd_training.validate data
.venv/bin/python -m unittest discover -s tests -v
```

数据格式验证可直接使用 `PYTHONPATH=src python3 -m quickadd_training.validate data`；完整测试集还需要 `train,teacher` 依赖。

## 测当前规则与模型

```sh
export JAVA_HOME=/Library/Java/JavaVirtualMachines/zulu-21.jdk/Contents/Home
.venv/bin/python -m quickadd_training.benchmark \
 --core ../DeadlinerCore --data-dir data --split dev --streaming \
 --model-dir artifacts/baselines/v1.1.0 --output reports/v1.1-dev
```

本机已在 `artifacts/baselines/v1.1.0` 保存已发布的三格式 v1.1 权重（Git 忽略）。清单版本受控在 `baselines/v1.1.0`，另一台机器可按其中链接下载并校验。

模型文件应包含 ONNX 和对应 `quick-add-model.json`；模型 SHA 必须匹配清单。无需联网。去掉 model-dir 可运行规则基线。
首次运行需要 Core 的 macOS/Kotlin Native 构建依赖，工具使用 offline；日志保存在输出目录。适配器源文件位于 Core 的 `shared/src/macosArm64Test/.../TrainingBenchmarkAdapterTest.kt`。它不进入发布包。

产物：`report.json`（三种策略的正确覆盖率、已接受结果的错误率、负例误接受、未接受比例、来源、P95）、`predictions.json`、`kmp-input.json`（真实 logits）、`kmp-build.log`。逐字符轨迹记录提案变化和提前接受，仅描述稳定性，不将所有前缀误认作负例。

所有计时为桌面 CPU/工具链代理值，单次样本、预热后，不能表示手机 P95；不包含 UI debounce。dev 数据及现有权重均公开可见，不能作独立真实准确率结论。

## 正式训练

先按 [数据规范](data/README.md) 补充独立人工审核语料及字符 spans，按表达家族划分 train/dev/test；最终测试冻结并记录 SHA。训练前执行 `validate` 检查所有 split 的文本/家族泄漏。

```sh
.venv/bin/python -m quickadd_training.train --data-dir data \
 --output artifacts/candidate --epochs 20 --seed 42 --coreml
```

输出 weights、ONNX、可选 Core ML、仅训练词表、训练/开发文件 SHA 与诊断指标。当前结构为小型字符 CNN 实验基线，后续结构比较共用同一数据与评估；不保证这个结构最终胜出。训练不自动发布；阈值在 dev 上校准且写入 manifest，禁止用最终测试调参。

MindSpore `.ms` 使用官方转换工具；新 manifest 入口为 `python -m quickadd_training.manifest --help`，支持 locale。旧 `verify_exports.py` 的 19 例检查仅适用于旧回归。新候选使用任意语料的一致性入口：

```sh
.venv/bin/python -m quickadd_training.parity --directory artifacts/candidate \
 --data data/dev.jsonl --coreml
# 用 tools/verify_mindspore.cpp 的原生执行器读取 checks；参数为目录、语料行数、模型 stem。
# 运行后再次检查，必须同时指定两种后端，才可能 threeFormatParityPassed=true。
.venv/bin/python -m quickadd_training.parity --directory artifacts/candidate \
 --data data/dev.jsonl --coreml --mindspore
```

工具校验张量形状、有限值、绝对误差、有效字符 argmax 一致性，并检查 MindSpore 输入未过期。保留模型文件和语料 SHA；只测 ONNX/Core ML 时不会声称三格式通过。最终还需用同一冻结语料运行 KMP benchmark 检查完整提案。

## 发布门禁

初始工程默认要求：至少 200 条独立审核最终测试（含至少 100 个可接受目标），hybrid 已接受结果错误率不高于规则且 <=1%，正确覆盖率至少提升 5 个百分点；三端各 >=100 个真机样本、P95<=50ms、端到端通过；三格式一致性与流式轨迹人工审核。门槛为项目初始工程目标，不是现有模型已达成的事实。

```sh
.venv/bin/python -m quickadd_training.gate reports/candidate-test/report.json release-evidence.json
```

证据格式见 `release-evidence.example.json`。默认证据为空且会拒绝发布；人工审核字段必须由实际审核提供。门禁是本工程发布流程要求，不会拦截 GitHub 手动上传，需要发布者遵守。没有发布命令，也不会修改现有 APP 模型地址。

## 已完成验证

见 [v1.1 开发基线](docs/baseline-v1.1-development.md)：16 项指标/数据/门禁测试、真实 KMP 三策略比较、34 条输入的逐字符轨迹、小数据单 epoch 训练及实际 ONNX/Core ML 输出检查。正式人工数据与三端真机评估仍未完成。


## 本土化双语实验 v2

本机已完成两轮从随机权重训练。当前中文/英文模型分别位于 `artifacts/zh-v2`、`artifacts/en-v2`，首轮保留在 `*-first`；模型文件 Git 忽略，完整 reports 也被忽略；数据、文档与 docs/evidence 的紧凑指标快照受版本控制。每份包含 ONNX、Core ML、MindSpore、词表、清单、训练报告与源代码快照。清单 `.invalid` URL 是本地实验占位，不能给 APP 下载，也没有更改三端线上模型地址。

- `data/zh`：10,136 条合成训练、900 条固定开发样例；`data/en`：10,137 条合成训练、900 条固定开发样例。开发集首次生成后冻结，结构迭代不重划。
- `data/zh-challenge` / `data/en-challenge`：34 / 38 条人工编写方式的助手挑战，仍是未人工复核的开发数据；包括跨月跨年午夜、方言、歧义、否定和标题时间词。
- `data/sources/massive`：27,094 条公开原始中英文 train/dev 样例，其中 8,022 条进入复核队列。保留原始意图、槽位、来源分区与平行组；原始标签尚未映射为完整 Deadliner 提案，**没有用于这两份模型训练**，官方 test 未导入。数据许可 CC BY 4.0 与署名随目录保存。
- KMP 按 manifest 的 `codec.locale` 对模型提取的字段归一化。模型负责意图与边界，日期算术、24:00 跨日及周期映射采用确定性逻辑；不重新从标题里抽日期。旧清单默认 zh。未完成原生客户端按语言下载/选模型及 UI i18n。

```sh
# 合成训练只能显式 opt-in，避免被误当成人工数据
.venv/bin/python -m quickadd_training.train --data-dir data/zh \
 --output artifacts/zh-v2 --locale zh --epochs 20 --coreml --allow-synthetic
.venv/bin/python -m quickadd_training.train --data-dir data/en \
 --output artifacts/en-v2 --locale en --epochs 20 --coreml --allow-synthetic
# 先构建、指纹化 KMP 适配器，再重复运行实际权重评估
.venv/bin/python -m quickadd_training.prepare_adapter --core ../DeadlinerCore --output /tmp/quickadd-adapter.json
.venv/bin/python -m quickadd_training.benchmark --core ../DeadlinerCore \
 --data-dir data/en-challenge --model-dir artifacts/en-v2 --streaming \
 --adapter-metadata /tmp/quickadd-adapter.json --output reports/en-v2-challenge
```

详细结果见 [双语实验报告](docs/localized-bilingual-experiment.md)，词义来源见 [来源目录](docs/localization-sources.md)。实验结果不满足发布门禁：需要独立人工最终集、规则回退策略修正、三端真机性能与端到端验收。实验模型没有发布到 Releases。

## 当前验收交付

用户已授权用v2重打验收包，现已上传GitHub预发布、三端清单接线，并去掉模型语义拒绝后的规则回退。详见实验报告最后的更新章节；上面的未发布/未接线说明记录上一阶段。正式门禁仍需要独立人工最终集和真机证据。

## 公开语料辅助训练与预训练对照 g3

见 [本轮实验](docs/public-corpus-generalization.md)。中英文业务训练加入空格扰动和新标题，MASSIVE原始train用于辅助监督，另外比较两个预训练小BERT。新增评估集只代表助手编写的标题词汇泛化检查；不是真实用户盲测。预训练教师没有部署三端。实际完整提案结果与权重SHA见docs/evidence/g3-generalization.json。

## 核心字段与标准化原型（2026-10-19）

用户确认先覆盖标题、时间、地点转备注，允许部分表达需要确认。已完成[片段抽取原型](docs/core-span-prototype-v1.md)及[标准化后规则识别对比](docs/normalization-comparison-v1.md)，包含相同数据BIO对照、真实KMP日期归一化、独立新实体留出与逐条失败。两条路线都能处理“明天下午 4 点要在逸夫楼开会”；目前标准化路线会漏复制房间号或字段尾字，不能据已知回归的较高成绩替换现有模型。

本轮是中文离线实验，生产v3、三个客户端与在线清单保持现有版本。片段模型已导出ONNX并核对完整提案，标准化模型仅做PyTorch实验。模型/完整报告在Git忽略的artifacts/reports，摘要在docs/evidence。新语料增加LOCATION/CONNECTOR角色，由span_data.validate校验，不适用旧九类BIO监督入口。

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 ../DeadlinerCore/build/quickadd-env/bin/python \
  -m quickadd_training.span_predict '明天下午 4 点要在逸夫楼开会'
PYTHONPATH=src HF_HUB_OFFLINE=1 ../DeadlinerCore/build/quickadd-env/bin/python \
  -m quickadd_training.normalizer_predict '明天下午 4 点要在逸夫楼开会'
```

已完成[标准化原文保真校验复验](docs/normalization-fidelity-v1.md)。固定权重、第三批新实体集上，原阈值的错误接受由55条降至2条，147条正确接受全部保留；dev重新校准后完整正确199/384、错误接受7/206，仍存在地点尾部被归入标题等角色边界错误。全套50项测试通过；这批合成数据不替代人工/真机验收。

已审核的设计：[提案置信度架构与Mermaid图](docs/proposal-confidence-architecture-v1.md)、[70%覆盖与低错误接受实验方案](docs/proposal-confidence-experiment-v1.md)。完整但低置信度的草案允许 `ready=true, needsAI=true`，隐藏AI还需通过硬校验及独立校准阈值。

最新实现：[TASK/HABIT联合原型报告v3](docs/task-habit-joint-prototype-v3.md)，承接[联合设计v2](docs/task-habit-joint-design-v2.md)。一次BERT编码，分类型片段头与正确性头，复用真实Core的星期/窗口/计次归一化。开发消融去掉了降低覆盖的学习式重排。

主合成留出TASK/HABIT正确接受覆盖97.56%/92.44%，均零观察错误；2,000条保护集零误接受。但旧表达回放只有244/384（63.54%），当前候选池上限也仅263/384（68.49%），**真实70%泛化目标尚未证明**。64项Core、62项Python行为检查通过，8,014条ONNX完整提案一致；没有客户端接入或发布。

```sh
PYTHONPATH=artifacts/zh-joint-v3-runtime/src HF_HUB_OFFLINE=1 ../DeadlinerCore/build/quickadd-env/bin/python \
  -m quickadd_training.joint_predict '工作日每周跑步三次' \
  --prototype reports/zh-joint-v3/type42-base/prototype.json --core artifacts/zh-joint-v3-runtime
```

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 ../DeadlinerCore/build/quickadd-env/bin/python \
  -m quickadd_training.normalizer_predict '明天下午 4 点要在逸夫楼开会' \
  --fidelity-settings reports/zh-fidelity-compare-v1/prototype.json
```
