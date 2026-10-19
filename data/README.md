# 数据契约

每行 JSON：id、group（表达家族）、text、preferredKind、now（UTC）、timezone、provenance、split、expected。
expected.decision 为 accept/defer。accept 必须包含 proposal 的全部业务字段：kind/title/dueAt/note/period/targetCount/weekdaysMask/reminderTime。
可附 spans：[{start,end,label}]，Unicode 字符索引，end 不含，label=TITLE/TIME/REPEAT/NOTE。监督训练的 accept 例必须有 TITLE，真正非添加请求的 defer 例须显式审核 trainingIntent=OTHER，且 spans=[]、slots=O。复杂但可添加的 defer 暂不参与监督训练，应保留作评估负例。

当前 dev 为助手手工编写的公开开发挑战集，不是真实用户语料或独立盲测。train/test 刻意为空：避免把开发例包装成训练/最终测试。填充真实人工审核数据后才能开始正式训练与发布。

同 group 不得跨 train/dev/test；规范化相同文本不允许跨 split。训练词表只由 train 构建，未知字由 [UNK] 表示。test 不进入训练入口、词表或阈值选择。最终测试应由另一名审核者保存，冻结后记录 SHA-256，发布时一次性评估；反复调参后此集合应降级为 dev。
