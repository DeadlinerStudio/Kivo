# Kivo Tiny 接入准备 v1

2026-10-19。品牌为 Kivo，训练仓库为 `DeadlinerStudio/Kivo`；本地目录和 Python 导入名暂保留。本文件是实验客户端的实现依据，不表示客户端已接入或模型已通过发布验收。

当前冻结候选为 `zh-joint-v3-type-seed42`：单次共享 BERT 编码、分类型片段头、片段分数组合、两种正确性头。没有学习式重排。主合成留出 TASK/HABIT 正确接受覆盖为 97.56%/92.44%，但旧表达回放只有 63.54%；真实跨表达 70% 目标未证明。详见[完整实验报告](task-habit-joint-prototype-v3.md)。

## 仓库与产物边界

Kivo 保存训练、数据、评测、导出、协议与回归样例；Core 保存业务日期、执行星期、计次窗口和目标次数的归一化。客户端负责 tokenizer、张量运行、桥接与 UI。不要在三个客户端分别实现日期算术。

模型、完整报告与冻结原生测试运行时保留在本地 `artifacts/` / `reports/`，不会随着 Git clone 下载。仓库的 `docs/evidence/` 保存紧凑证据，`integration/joint-v1/` 保存已曝光回归与 tokenizer 样例。这些样例用于实现一致性检查，不是独立准确率证据。

本地预览包生成与校验：

```sh
PYTHONPATH=src python3 -m quickadd_training.integration_bundle \
  --prototype reports/zh-joint-v3/type42-base/prototype.json \
  --onnx-dir reports/zh-joint-v3/onnx-parity \
  --runtime artifacts/zh-joint-v3-runtime \
  --output artifacts/kivo-tiny-v1-preview
PYTHONPATH=src python3 -m quickadd_training.integration_bundle \
  --verify artifacts/kivo-tiny-v1-preview
```

生成器先核验权重、评分头、tokenizer/config、冻结 Python 策略、Core 测试二进制与 ONNX parity 的 SHA，再复制 3 个图和 5 个 tokenizer/config 文件，共 14,620,389 字节。任何过期或不一致都拒绝生成。输出已存在也拒绝覆盖。`kivo-preview.json` 标记 `experimental=true`、`enabledByDefault=false`、`formalGatePassed=false`、`productionEligible=false`。它不是线上下载清单，也不包含 `.pt` 或原生测试程序。

当前证据只覆盖桌面 ONNX：8,014 条完整提案一致。Core ML、MindSpore、三个客户端 tokenizer/codec 与真机性能尚未完成。新设备不应把此包接到旧 BIO 模型下载入口；目前仅用于开发者显式选择的本地实验。上游权重许可状态沿用历史报告记录，本轮没有公开分发权重。

## 推理协议

编码器输入均为 int32：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `input_ids` | `[1,96]` | 固定 tokenizer 的 WordPiece ID，含特殊符号与 padding |
| `attention_mask` | `[1,96]` | 真实 token 为 1，padding 为 0 |
| `preferred_kind` | `[1]` | TASK=0，HABIT=1，必须来自当前入口上下文 |

输出均为 float32：`intent_logits[1,3]`、`span_logits[1,8,96,96]`、`token_features[1,96,128]`。意图顺序 TASK/HABIT/CONFIRM；角色顺序 TITLE/TIME/LOCATION/NOTE/CONNECTOR/DAYS/GOAL_WINDOW/COUNT。

空输入输出 `ready=false, needsAI=false`；原文超过 126 个 Unicode code point 或含特殊符号的编码长度超过 96 时保持确认，不截断后接受。offset 是原文 Unicode code point 的 `[start,end)`，不是 UTF-16 下标、UTF-8 字节或 Swift 字素序号。Kotlin/JS 需要 code point→UTF-16 映射；Swift 需要原文索引映射。不得先改写空白或规范化 Unicode 后复用旧 offset。

仅考虑非特殊 token、有非空原文 offset、上三角且 token 距离小于 64 的 span。每角色取分数最高 12 个、丢弃分数低于 -2 的候选，再用每角色前 4 个做 beam=32 的不重叠组合。TITLE 必须一个；其他字段允许缺失，TIME 可为两个不重叠片段。排序与并列分数处理必须对齐冻结参考实现。不要把 span 分数或意图 softmax 当作整条提案置信度。

CONFIRM 转确认结果；HABIT 入口被预测成 TASK 时硬校验失败。TASK 需要时间字段；无日程的 HABIT 只有在 HABIT 入口才可采用默认周期。覆盖校验逐个原文位置检查，只允许参考实现中的有限连接词规则，不能用通用正则丢掉未知内容。

## Core 归一化与提案

将模型选出的 TIME 片段按原文顺序用一个空格连接；DAYS/GOAL_WINDOW/COUNT 各自无分隔连接，传给 Core。HABIT 的四个字段分别传入 `normalizeHabitFields(time, days, window, count, locale, now, zone)`；TASK 禁止带习惯字段，沿用时间归一化。`now` 与 `zone` 必须与本轮输入上下文一致。

执行日和目标窗口独立，例如“工作日每周跑步三次”应为 `weekdaysMask=31, period=WEEKLY, targetCount=3`。星期 mask 的 bit0 为周一，bit6 为周日。默认 DAILY/1/127，未写提醒时间不生成提醒。

原型调用带标题占位的真实 Core 规则，只接收 `ready && !needsAI` 且标题未被残留字段污染的归一化结果；复用 dueAt/period/targetCount/weekdaysMask/reminderTime，不让 Core 重读原始标题。生产桥接仍需在 Core 内提供稳定入口，当前 helper 是 internal。

TITLE/NOTE/LOCATION 都从原文 offset 复制；地点写入 `地点：<LOCATION>`，与 NOTE 用 `；` 连接。业务字段和 hardReasons 全部确定后才运行正确性头。

## 正确性特征与门控

基础特征 1515 维，顺序精确参照冻结 `joint_decode.features`：CLS 128 维；随后 TITLE/TIME/LOCATION/NOTE/DAYS/GOAL_WINDOW/COUNT 各 192 维，分别为片段 token 均值 128 维和边缘向量 64 维。边缘向量由前一个、起点、终点、后一个 token 各前 16 维拼接，越界钳制到 `[0,95]`；多 TIME 取片段向量的均值，缺字段填零。

最后 43 维依次为 3 个意图概率、TASK/HABIT 两个 kind indicator、HABIT 入口 indicator；七字段各 present/maxScore÷10/总字符长度÷96/最小 start÷96/最大 end÷96；原文覆盖比例和片段总分÷50。

正确性特征在其后追加 15 维：draftValid、DAILY、WEEKLY、周一至周日七 bit、`log1p(max(0,targetCount))/10`、有 dueAt、有 reminderTime、去重 hardReasons 数÷10、rankingMargin÷10，共 1530 维。rankingMargin 是第一、第二名候选总分差，只有一个候选时为 0；不可全部填 0。

`quality-TASK.onnx` / `quality-HABIT.onnx` 输入 `features[N,1530]` float32，输出 `score[N]` float32，内部已包含特征标准化。按最终预测 kind 选择图，再用同一 kind 的 Platt 参数计算 `c=sigmoid(slope*score+bias)`，sigmoid 输入限制在 `[-700,700]`。参数和阈值以包内清单为准，当前 TASK T=0.999，HABIT T=0.9。

```text
ready = draftValid
localTrusted = draftValid && finite(c) && 0 <= c <= 1 && T可用 && c >= T
needsAI = 输入非空 && !localTrusted
```

允许 `ready=true, needsAI=true`：显示草案、允许用户确认添加，同时保留 AI 按钮。置信度缺失、非有限或阈值不可用时不隐藏按钮。高置信度不能覆盖任何硬校验失败。

参考输出 source 为 `JOINT_PROTOTYPE`；接入现有 DTO 时映射为 `LOCAL`，不要未经契约变更扩展线上 source 枚举。测试比较业务字段、ready/needsAI、状态及原因；排除这项 source 命名差异，置信度容差先用 1e-5。

## Core 补丁与后续步骤

本次 Core 提交为 `bb2c8cf356b9a35179556db13756f6cf89f4501c`，父提交 `89e63b5562e647eccd2c028cc335ac1d4228202c`。可审阅[补丁](../integration/core/0001-kivo-structured-habit-fields.patch)，仅涉及结构化习惯归一化、中文“十”、星期邻接时钟修复及相关测试。

Core 本地 `byos` 的基础代码已有 6 个未推送提交，远端 `origin/byos` 尚无所需 Quick Add 模块；本次没有连带推送它们。此补丁不能直接应用到缺少基础模块的旧远端。应先对齐 Core 的基础提交，再合入该提交，不能仅复制 helper 假装完成接入。

1. Core 补齐八角色解码、覆盖校验、特征构造、校准与桥接，使用本仓 fixtures 对齐参考实现。
2. 选择一个客户端接本地实验包，校验 tokenizer、各图 logits、完整提案及异步输入版本；保留文本/入口/时区变化、取消和旧结果丢弃保护。
3. Android 当前提交条件和 iOS Quick Add 当前 `canSubmit` 都包含 `!needsAI`，接入时改为按 ready 与当前草案有效性判断，AI 按钮继续按 needsAI 控制；UI 阶段按共享规格先补该端原型。
4. 补齐新表达家族数据与负迁移对照，完成正式人工/三格式/真机门禁后再决定发布。父需求保持未完成。
