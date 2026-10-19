"""Synthetic bootstrap corpus. Labels are generated from known spans, never from the rule parser."""
import random

TASKS = ['交报告', '买牛奶', '提交作业', '参加会议', '完成论文', '给妈妈打电话', '整理房间', '预约体检', '修理电脑',
         '回复邮件', '寄快递', '取快递', '交电费', '还信用卡', '看牙医', '买生日礼物', '检查代码', '提交申请',
         '打印文件', '准备演讲', '写总结', '准备面试', '更新简历', '完成设计稿', '去银行', '买车票', '修复问题',
         '续费会员', '预约会议室', '完成实验', '修改方案', '提交合同', '交房租', '买菜', '发工资', '清理照片',
         '备份数据', '确认订单', '办理手续', '归还图书', '阅读明天的天气预报', '整理周一的会议记录', '规划下一次旅行',
         '填写表格', '查收包裹', '查看账单', '联系客户', '安排出差', '预订酒店', '审核材料', '写代码', '发布新版本']
HABITS = ['读书', '跑步', '喝水', '练琴', '学英语', '运动', '冥想', '背单词', '写日记', '记账', '散步', '阅读',
          '复习功课', '打扫卫生', '早起', '整理书桌', '练字', '做瑜伽', '练习编程', '学习数学', '拉伸', '刷牙',
          '记录心情', '收拾房间', '学习法语', '做俯卧撑', '听英语', '背诗', '学习历史', '健身', '写作', '听播客']
DATES = ['今天', '明天', '后天', '下周一', '下周二', '下周三', '下周四', '下周五', '周六', '周日',
         '星期一', '星期二', '星期三', '星期四', '星期五', '10月20日', '11月3日', '12月12日']
TIMES = ['上午9点', '上午10点', '下午3点', '下午6点', '晚上8点', '晚上9点', '早上7点', '下午三点',
         '晚上八点', '上午九点半', '下午三点半', '18:30', '09:00', '21:30', '中午12点']
REPEATS = ['每天', '每日', '每个工作日', '每个周末', '每周一', '每周二', '每周三', '每周四', '每周五', '每周六', '每周日',
           '每周一三五', '每周二四', '每周六日', '每月', '每周', '每天2次', '每天三次', '每周3次']
NOTES = ['带上电脑', '记得带身份证', '提前准备材料', '带上文件', '不要忘记钥匙', '记得充电', '顺便带上雨伞', '带上笔记本']
PREFIXES = ['', '请', '帮我', '我要', '记得', '提醒我', '添加任务', '创建任务', '帮我记下', '给我添加', '我想', '我需要', '记一下']
SHARED_TITLES = TASKS + HABITS + ['交报告书', '写工作总结', '提交日报', '检查库存', '核对账目', '整理资料', '回访客户', '给爸爸回电话', '归还资料', '练习钢琴']
OTHER = ['今天的天气怎么样', '你觉得明天会下雨吗', '这是什么意思', '我今天很开心', '如何安排时间', '我不知道做什么',
         '谢谢你', '你好', '帮我想一个名字', '读书有什么好处', '明天放假吗', '我完成了报告', '昨天已经买了牛奶',
         '会议已经结束了', '我要取消提醒', '不要创建任务', '我只是随便说说', '查询我的待办', '看看本周计划', '我不想运动']


def render(parts, intent):
    text, labels = '', []
    for value, group in parts:
        if not value:
            continue
        text += value
        labels += ([0] * len(value) if group == 0 else [group] + [group + 1] * (len(value) - 1))
    return dict(text=text, intent=intent, slots=labels)


def generate(count=18000, seed=42):
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        intent = rng.choices([0, 1, 2], [0.45, 0.35, 0.2])[0]
        if intent == 2:
            if rng.random() < 0.5:
                text = rng.choice(OTHER)
            else:
                cue = rng.choice(DATES + REPEATS)
                text = cue + rng.choice(['不需要', '不要', '不用', '已经', '取消']) + rng.choice(SHARED_TITLES)
            result.append(render([(text, 0)], 2))
            continue
        # Intent follows the request/schedule, never a title dictionary: daily reporting is a habit too.
        title = rng.choice(SHARED_TITLES)
        prefix = rng.choice(PREFIXES)
        moment = rng.choice(DATES) + (rng.choice(TIMES) if rng.random() < 0.75 else '') if intent == 0 else rng.choice(TIMES)
        # Preserve span offsets while teaching realistic spaces inside clock expressions.
        if rng.random() < 0.6:
            import re
            gap = rng.choice([' ', '  ', '\u3000'])
            moment = re.sub(r'([0-9一二三四五六七八九十两]+)([点时:：])', lambda m: gap + m[1] + gap + m[2], moment)
        recurring = rng.choice(REPEATS) if intent == 1 else ''
        # Expression families are shared by the random bootstrap split; this is not independent evaluation.
        layout = rng.randrange(6)
        if intent == 0:
            variants = [
                [(prefix, 0), (moment, 3), (title, 1)],
                [(prefix, 0), (title, 1), ('，截止到', 0), (moment, 3)],
                [('帮我把', 0), (title, 1), ('安排到', 0), (moment, 3)],
                [('添加一个任务：', 0), (title, 1), ('，时间是', 0), (moment, 3)],
                [(moment, 3), ('记得', 0), (title, 1)],
                [(prefix, 0), (title, 1)],
            ]
        else:
            variants = [
                [(prefix, 0), (recurring, 5), (moment, 3), (title, 1)],
                [(prefix, 0), (recurring, 5), (title, 1)],
                [('养成', 0), (title, 1), ('的习惯，', 0), (recurring, 5)],
                [('我想', 0), (recurring, 5), (moment, 3), (title, 1)],
                [(title, 1), ('，', 0), (recurring, 5), (moment, 3)],
                [('添加习惯：', 0), (title, 1), ('，', 0), (recurring, 5)],
            ]
        parts = variants[layout]
        if rng.random() < 0.35:
            parts = [(value + (' ' if group == 0 and value else ''), group) for value, group in parts]
        if rng.random() < 0.18:
            parts = parts + [('，备注：', 0), (rng.choice(NOTES), 7)]
        result.append(render(parts, intent))
    # Entire multi-task expressions are OTHER; never accept just one member silently.
    for _ in range(count // 12):
        a, b = rng.sample(TASKS, 2)
        result.append(render([(rng.choice(DATES) + a + rng.choice(['然后', '以及', '；']) + b, 0)], 2))
    rng.shuffle(result)
    return result


# Authored separately from generation. No precision claim is based solely on training-family samples.
ACCEPTANCE = [
    render([('明天下午3点', 3), ('交报告', 1)], 0),
    render([('提醒我', 0), ('明天上午9点', 3), ('提交作业', 1)], 0),
    render([('帮我把', 0), ('回复邮件', 1), ('安排到', 0), ('后天下午6点', 3)], 0),
    render([('买牛奶', 1)], 0),
    render([('每天', 5), ('晚上8点', 3), ('读书', 1)], 1),
    render([('每个工作日', 5), ('早上7点', 3), ('跑步', 1)], 1),
    render([('养成', 0), ('记账', 1), ('的习惯，', 0), ('每天', 5)], 1),
    render([('交报告', 1), ('，截止到', 0), ('明天', 3), ('，备注：', 0), ('带上电脑', 7)], 0),
    render([('明天不需要开会', 0)], 2),
    render([('今天的天气怎么样', 0)], 2),
    render([('明天交报告然后买牛奶', 0)], 2),
    render([('帮我记录', 0), ('后天', 3), ('给爸爸回电话', 1)], 0),
    render([('在', 0), ('下周三下午两点', 3), ('归还资料', 1)], 0),
    render([('每周二四', 5), ('练习钢琴', 1)], 1),
    dict(render([('每天', 5), ('下午 3 点', 3), ('交报告书', 1)], 1), reminderTime='15:00'),
    dict(render([('每天', 5), ('下午3点', 3), ('交报告书', 1)], 1), reminderTime='15:00'),
    dict(render([('每天', 5), ('下午三点', 3), ('交报告书', 1)], 1), reminderTime='15:00'),
    dict(render([('每个工作日', 5), ('上午 9 点', 3), ('提交日报', 1)], 1), reminderTime='09:00'),
    render([('每天不要交报告书', 0)], 2),
]
