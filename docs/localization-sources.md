# 本土表达来源与语义政策（2026-10-10）

只记录表达及自编例句，不复制完整文章/词典例句。样本 provenance=assistant-authored-synthetic-localization，annotationStatus=synthetic；没有假称人工审核或真实用户分布。

- 转钟：[武汉市政府新闻](https://3g.wuhan.gov.cn/sy/whyw/202502/t20250201_2528096.shtml) 的深夜工作语境。本项目按用户约定：某日晚的转钟 = 该日24:00 = 次日00:00；转钟后是区间，没有精确钟点时拒绝自动设置。
- 明儿：[教育部修订本词典](https://dict.revised.moe.edu.tw/dictView.jsp?ID=32486&la=1&powerMode=0) 同时列明天/以后；有明确钟点的安排按明天，无确定安排的泛指以后需确认。
- 晌午：[教育部台语词典对应释义](https://sutian.moe.edu.tw/zh-hant/poosiu/poosiu/72/%E6%99%9D/) 为昼间语义，不自动编造12:00；仅“晌午12点”这类有钟点的组合可精确化。
- 差一刻：[Vista Higher Learning 中文教学教材](https://learn.vistahigherlearning.com/journeys/assets/CHN1e_V1_SE_U03_114-171_spread_low.pdf) 给出减去15分钟的报时方法；下午差一刻三点 =>14:45。
- midnight / noon：[英国国家物理实验室](https://www.npl.co.uk/resources/q-a/is-midnight-12am-or-12pm) 说明午夜日界和12am/pm的混淆。因此本工程明确12am=00:00、12pm=12:00、tonight at midnight=次日00:00；midnight tomorrow日期歧义需确认。
- at/on/in 时间介词：[Cambridge Grammar](https://dictionary.cambridge.org/grammar/british-grammar/at-on-in-time)。英文训练独立覆盖语序、介词、大小写、am/pm、quarter past/to、half past、weekdays/weekends。

其余普通表达和全部任务标题由项目自行编写。没有稳定钟点的下班后、天亮、擦黑、一早、EOD、COB；未能表示的隔天/fortnightly等周期；缺少年份/地区的3/4日期，均先放入需要确认的挑战样例，不悄悄猜默认值。

- 英式/美式报时：[Cambridge词典作者说明](https://dictionaryblog.cambridge.org/2018/02/14/what-time-is-it-how-to-say-the-time/)，记录 quarter past/after/to 与英国 half eight 用法；裸 half eight 的地区歧义仍需确认。
- 礼拜：[教育部修订本词典](https://dict.revised.moe.edu.tw/dictView.jsp?ID=62078&q=1&word=%E7%A6%AE%E6%8B%9C) 的星期语义。
- 真实公开中英文原始语料：[MASSIVE官方工程](https://github.com/alexa/massive)、[数据许可说明](https://raw.githubusercontent.com/alexa/massive/main/NOTICE.md)。导入 train/dev，保留原始标签和分区、平行组；不导入 test、不保留 worker IDs。当前仅待复核，没有进入实验权重训练。
