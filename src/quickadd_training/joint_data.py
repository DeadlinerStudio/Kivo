"""Versioned synthetic TASK/HABIT screening partitions, authored gold not parser labels."""
import argparse
import copy
import datetime as dt
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import unicodedata
from zoneinfo import ZoneInfo

from .corpus import BASE, TITLE_POOL
from .data import digest, load, validate as validate_business
from .span_data import make_row, temporal, LOCATIONS

ROLES = ('TITLE', 'TIME', 'LOCATION', 'NOTE', 'CONNECTOR', 'DAYS', 'GOAL_WINDOW', 'COUNT')
KINDS = ('TASK', 'HABIT', 'CONFIRM')
SIZES = dict(train=20000, rank=4000, quality=6000, dev=2000, cal_fit=2000,
             cal_gate=4000, test=4000, protection=2000)
DAYS = [('每天', 127), ('工作日', 31), ('周末', 96), ('每周一三五', 21), ('每周二四', 10),
        ('每周三', 4), ('每周六日', 96), ('每周一二四', 11)]
NOTES = ['带好材料', '提前做好准备', '使用自己的设备', '带上笔记本', '记得穿运动鞋', '记录完成情况']


def title_pools():
    objects = {
        '练习': '法语口语 德语听力 英语发音 日语书写 西班牙语 意大利语 俄语单词 韩语拼写 吉他和弦 钢琴音阶 小提琴指法 架子鼓 节奏训练 水彩绘画 素描写生 毛笔书法 硬笔书法 陶笛 长笛 萨克斯 象棋围棋 羽毛球 篮球投篮 乒乓球 排球发球 瑜伽动作 平衡动作 游泳转身 跳绳 太极拳'.split(),
        '阅读': '历史资料 地理教材 科普文章 数学笔记 文学评论 经济案例 建筑设计 城市规划 植物图鉴 天文入门 物理实验 化学资料 生物笔记 软件文档 旅行指南 古诗词 法律常识 摄影教程 逻辑学 语言学 教育理论 音乐史 电影评论 美术史 人物传记 自然观察 文化研究 考古资料 民俗资料 设计案例'.split(),
        '整理': '学习资料 办公文件 会议纪要 读书笔记 课程作业 实验记录 活动照片 出差清单 旅行账单 家庭相册 旧报纸 收藏卡片 书架 收据 工具箱 书桌 花园 厨房 客厅 阳台 衣柜 邮件 项目记录 工作计划 课程大纲 参考书目 通讯录 联系名单 购物清单 物品目录'.split(),
        '复习': '三次函数 二次函数 微积分 线性代数 概率论 数理统计 离散数学 编译原理 操作系统 数据结构 计算机网络 数据库原理 电路分析 机械设计 工程制图 材料力学 理论力学 流体力学 热力学 大学物理 普通化学 有机化学 遗传学 生态学 中国历史 世界历史 现代文学 古代文学 英语语法 数字逻辑'.split(),
        '检查': '没有签字的文件 没修好的设备 未提交的材料 不能访问的链接 已取消的预约 尚未完成的清单 缺少的凭证 项目进展 库存清单 装备 电源 接线 零部件 实验仪器 培训记录 学习进度 发音 拼写 标点 日期 账单 发票 印章 申请表 报告 数据 日志 备份 通知 消息'.split(),
        '完成': '晨间拉伸 夜间拉伸 户外散步 室内训练 核心训练 呼吸训练 冥想训练 早间阅读 晚间阅读 口语训练 绘画练习 书法练习 听力练习 阅读练习 写作练习 清洁工作 物品归位 手工制作 拼图练习 数独练习 脑力训练 记忆训练 课程练习 乐理练习 节奏练习 室内步行 放松练习 视力休息 肩颈练习 平衡训练'.split(),
    }
    values = [verb + noun for verb, nouns in objects.items() for noun in nouns]
    random.Random(7361).shuffle(values)
    return {s: values[i::len(SIZES)] for i, s in enumerate(SIZES)}


def location_pool(index):
    buildings = ['文澜', '笃实', '嘉木', '清溪', '星河', '青禾', '云亭', '明远',
                 '紫藤', '听松', '海棠', '墨池', '银竹', '雨荷', '知行', '锦书']
    return [f'{b}{suffix}' for b in buildings[index * 2:index * 2 + 2]
            for suffix in ['楼A203', '楼B419', '活动中心', '社区接待处', '阅览室317', '报告厅二楼']]


def validate(splits):
    validate_business({s: [dict(r, spans=[v for v in r['spans'] if v['label'] in ['TITLE', 'TIME', 'NOTE']])
                          for r in rows] for s, rows in splits.items()})
    entities = {s: {k: set() for k in ['TITLE', 'LOCATION']} for s in splits}
    for split, rows in splits.items():
        for row in rows:
            used = set()
            for span in row['spans']:
                a, b = span['start'], span['end']
                if span['label'] not in ROLES or not 0 <= a < b <= len(row['text']) or used & set(range(a, b)):
                    raise ValueError('Invalid/overlapping joint span')
                used.update(range(a, b))
                if span['label'] in entities[split]:
                    entities[split][span['label']].add(row['text'][a:b])
            if row['expected']['decision'] == 'accept':
                title = [row['text'][s['start']:s['end']] for s in row['spans'] if s['label'] == 'TITLE']
                if title != [row['expected']['proposal']['title']]:
                    raise ValueError('Gold title differs from authored span')
    for i, a in enumerate(splits):
        for b in list(splits)[i + 1:]:
            for role in ['TITLE', 'LOCATION']:
                if entities[a][role] & entities[b][role]:
                    raise ValueError('Cross-partition entity leak: ' + a + '/' + b + '/' + role)


def build(sizes=None, seed=20261019, revision=2):
    rng, pools, splits = random.Random(seed), title_pools(), {}
    legacy = []
    if revision >= 3:
        legacy = load(Path(__file__).resolve().parents[2] / 'data/zh-core-spans-v1/train.jsonl')
        replay_titles = set(TITLE_POOL['zh']) | {r['expected']['proposal']['title'] for r in regression() if r['expected']['decision'] == 'accept'}
        replay_titles |= {r['expected']['proposal']['title'] for r in legacy if r['expected']['decision'] == 'accept' and r['expected']['proposal']['dueAt']}
        for split in pools:
            if split != 'train':
                pools[split] = [t for t in pools[split] if t not in replay_titles]
        pools['train'] = sorted(set(pools['train']) | replay_titles)
    for index, (split, size) in enumerate((sizes or SIZES).items()):
        rows, seen, attempts = [], set(), 0
        locations = location_pool(list(SIZES).index(split))
        if revision >= 3 and split == 'train':
            locations = locations + LOCATIONS['train'] + ['逸夫楼']
        while len(rows) < size:
            attempts += 1
            if attempts > size * 500:
                raise ValueError('Unable to create enough unique authored inputs: ' + split)
            i = len(rows)
            domain = 'TASK' if i % 2 == 0 else 'HABIT'
            title, loc = rng.choice(pools[split]), rng.choice(locations)
            day, clock, sep, due, now, zone = temporal(rng)
            if revision >= 3 and domain == 'TASK' and rng.random() < .2:
                today = dt.datetime.fromisoformat(now.replace('Z', '+00:00')).astimezone(ZoneInfo(zone)).date()
                next_week = rng.randrange(2)
                weekday = rng.randrange(today.weekday(), 7) if not next_week else rng.randrange(7)
                day = ('下周' if next_week else '本周') + '一二三四五六日'[weekday]
                due = (today + dt.timedelta(days=weekday - today.weekday() + next_week * 7)).isoformat() + 'T' + due.split('T')[1]
                sep = ''
            preferred = 'HABIT' if domain == 'HABIT' and rng.random() < .5 else 'TASK'
            d, mask = rng.choice(DAYS)
            window = '每周' if rng.random() < .35 else ''
            count = rng.choice(['一次', '两次', '三次', '十次', '2次', '5次']) if rng.random() < .65 else ''
            target = {'一次': 1, '两次': 2, '三次': 3, '十次': 10, '2次': 2, '5次': 5}.get(count, 1)
            has_clock = domain == 'TASK' or rng.random() < .6
            has_loc = rng.random() < (.7 if domain == 'TASK' else .25)
            ptime = (day + sep + clock if domain == 'TASK' else clock) if has_clock else ''
            reminder = due.split('T')[1] if has_clock else None
            t, time, place = (title, 'TITLE'), (ptime, 'TIME'), (loc, 'LOCATION')
            if domain == 'TASK':
                mode = rng.randrange(6)
                parts = [time, ('在', 'CONNECTOR'), place, t] if has_loc else [time, t]
                if mode == 1:
                    parts = [t, ('，', None), time] + ([('在', 'CONNECTOR'), place] if has_loc else [])
                elif mode == 2:
                    parts = [('请提醒我', 'CONNECTOR')] + parts
                elif mode == 3 and has_loc:
                    parts = [('在', 'CONNECTOR'), place, t, ('，时间是', 'CONNECTOR'), time]
                elif mode == 4 and has_loc:
                    parts = [(day, 'TIME'), ('在', 'CONNECTOR'), place, t, ('，', None), (clock, 'TIME')]
                elif mode == 5 and has_loc:
                    parts = [time, t, ('，地点：', 'CONNECTOR'), place]
                proposal = dict(BASE, title=title, dueAt=due)
            else:
                # Weekly goal alone has all execution days unless a separate DAYS is authored.
                if window and rng.random() < .5:
                    d, mask = '', 127
                parts = [(d, 'DAYS'), (window, 'GOAL_WINDOW'), time]
                if has_loc:
                    parts += [('在', 'CONNECTOR'), place]
                parts += [t, (count, 'COUNT')]
                mode = rng.randrange(4)
                if mode == 1:
                    parts = [t, ('，', None), (d, 'DAYS'), (window, 'GOAL_WINDOW'), (count, 'COUNT')]
                    if has_clock:
                        parts += [('，提醒时间', 'CONNECTOR'), time]
                    if has_loc:
                        parts += [('，地点', 'CONNECTOR'), place]
                elif mode == 2:
                    parts = [('创建习惯', 'CONNECTOR')] + parts
                elif mode == 3 and d and not window:
                    goal = rng.choice(['每个执行日', '每天']) if revision >= 3 else '每个执行日'
                    parts = [(d, 'DAYS'), time, t, ('，', None), (goal, 'GOAL_WINDOW'), (count, 'COUNT')]
                    if has_loc:
                        parts += [('，地点', 'CONNECTOR'), place]
                if preferred == 'HABIT' and rng.random() < .03:
                    parts, mask, window, target, reminder, has_loc, preferred = [t], 127, '', 1, None, False, 'HABIT'
                proposal = dict(BASE, kind='HABIT', title=title, period='WEEKLY' if window else 'DAILY',
                                targetCount=target, weekdaysMask=mask, reminderTime=reminder)
            note = ''
            if rng.random() < .2:
                note = rng.choice(NOTES)
                parts += [('，备注：', 'CONNECTOR'), (note, 'NOTE')]
            proposal['note'] = ('地点：' + loc if has_loc else '') + ('；' if has_loc and note else '') + note
            category = domain.lower() + '-supported'
            if split == 'protection' or i % 10 in [8, 9]:
                control = rng.randrange(8)
                if control == 0:
                    parts = [(rng.choice(['不要', '取消', '别']), 'CONNECTOR')] + parts
                    category = 'negation'
                elif control == 1:
                    parts = [('如果有空就', 'CONNECTOR')] + parts
                    category = 'conditional'
                elif control == 2:
                    parts += [('然后', 'CONNECTOR'), (rng.choice(pools[split]), 'TITLE')]
                    category = 'multiple-events'
                elif control == 3:
                    parts = [('每隔两天', 'DAYS'), t]
                    category = 'unsupported-interval'
                elif control == 4:
                    parts = [('每天', 'DAYS'), t, ('30分钟', 'COUNT')]
                    category = 'unsupported-unit'
                elif control == 5:
                    parts = [('明天开始', 'TIME'), ('每天', 'DAYS'), t]
                    category = 'unsupported-start'
                elif control == 6:
                    parts = [time if ptime else ('早上8点', 'TIME'), ('在', 'CONNECTOR'), place]
                    category = 'missing-title'
                else:
                    parts = [('每月', 'GOAL_WINDOW'), t, ('两次', 'COUNT')]
                    category = 'unsupported-window'
                proposal = None
            row = make_row(parts, f'joint-{split}-{i:05}', split, category, proposal, now, zone)
            row.update(preferredKind=preferred, domain=domain, provenance='authored-synthetic-joint-v2-not-human-reviewed')
            key = unicodedata.normalize('NFKC', row['text']).strip().casefold()
            if key in seen or len(row['text']) > 90:
                continue
            seen.add(key)
            row['group'] = 'source-' + hashlib.sha256(key.encode()).hexdigest()[:24]
            rows.append(row)
        splits[split] = rows
    if revision >= 3:
        old = [r for r in legacy if r['expected']['decision'] == 'defer' or r['expected']['proposal']['dueAt']]
        replay = [dict(r, domain='TASK') for r in old[:1500]] + regression()
        used = {unicodedata.normalize('NFKC', r['text']).strip().casefold() for r in splits['train']}
        replaced = set()
        for source in replay:
            key = unicodedata.normalize('NFKC', source['text']).strip().casefold()
            if key in used:
                continue
            for i, original in enumerate(splits['train']):
                if i not in replaced and original['domain'] == source['domain'] and original['expected']['decision'] == source['expected']['decision']:
                    value = copy.deepcopy(source)
                    value.update(id=original['id'], split='train', domain=source['domain'],
                                 group='source-' + hashlib.sha256(key.encode()).hexdigest()[:24],
                                 provenance='synthetic-known-training-replay-not-human-reviewed')
                    used.add(key); replaced.add(i); splits['train'][i] = value
                    break
    validate(splits)
    return splits


def regression():
    cases = [
        ([('每天', 'DAYS'), ('早上7点', 'TIME'), ('跑步', 'TITLE')], 'HABIT', dict(title='跑步', reminderTime='07:00')),
        ([('每周一三五', 'DAYS'), ('跑步', 'TITLE')], 'HABIT', dict(title='跑步', weekdaysMask=21)),
        ([('每周', 'GOAL_WINDOW'), ('跑步', 'TITLE'), ('三次', 'COUNT')], 'HABIT', dict(title='跑步', period='WEEKLY', targetCount=3)),
        ([('工作日', 'DAYS'), ('早上8点', 'TIME'), ('背单词', 'TITLE'), ('，', None), ('每天', 'GOAL_WINDOW'), ('两次', 'COUNT')], 'HABIT', dict(title='背单词', weekdaysMask=31, reminderTime='08:00', targetCount=2)),
        ([('工作日', 'DAYS'), ('每周', 'GOAL_WINDOW'), ('跑步', 'TITLE'), ('三次', 'COUNT')], 'HABIT', dict(title='跑步', weekdaysMask=31, period='WEEKLY', targetCount=3)),
        ([('本周三晚上7点', 'TIME'), ('跑步', 'TITLE')], 'TASK', dict(title='跑步', dueAt='2026-10-21T19:00')),
        ([('每周三', 'DAYS'), ('晚上7点', 'TIME'), ('跑步', 'TITLE')], 'HABIT', dict(title='跑步', weekdaysMask=4, reminderTime='19:00')),
        ([('每天', 'DAYS'), ('复习三次函数', 'TITLE')], 'HABIT', dict(title='复习三次函数')),
        ([('明天上午十点', 'TIME'), ('提交每周报告', 'TITLE')], 'TASK', dict(title='提交每周报告', dueAt='2026-10-20T10:00')),
        ([('明天下午 4 点', 'TIME'), ('要在', 'CONNECTOR'), ('逸夫楼', 'LOCATION'), ('开会', 'TITLE')], 'TASK', dict(title='开会', dueAt='2026-10-20T16:00', note='地点：逸夫楼')),
        ([('跑步', 'TITLE')], 'HABIT', dict(title='跑步')),
        ([('每隔两天', 'DAYS'), ('跑步', 'TITLE')], 'HABIT', None),
        ([('每天', 'DAYS'), ('读书', 'TITLE'), ('30分钟', 'COUNT')], 'HABIT', None),
        ([('不要', 'CONNECTOR'), ('每天', 'DAYS'), ('跑步', 'TITLE')], 'HABIT', None),
    ]
    return [dict(make_row(parts, f'joint-reg-{i:03}', 'regression', 'known-regression',
                          dict(BASE, kind=kind, **values) if values else None),
                 preferredKind='HABIT' if i == 10 else 'TASK', domain=kind)
            for i, (parts, kind, values) in enumerate(cases)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/zh-joint-v2'))
    parser.add_argument('--revision', type=int, choices=[2, 3], default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to replace frozen data')
    splits = build(revision=args.revision)
    args.output.mkdir(parents=True)
    for name, rows in dict(splits, regression=regression()).items():
        (args.output / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    provenance = dict(humanReviewed=False, realUserData=False, scope='synthetic engineering screening only', revision=args.revision,
                      knownRegressionReplayed=args.revision >= 3,
                      sourceGroupsAndExactEntitiesDisjoint=True, syntax='shared synthetic layouts, not syntax-family blind test',
                      formalGateEligible=False, sourceSha256=digest(Path(__file__)),
                      partitions={s: dict(rows=len(rs), uniqueTexts=len({r['text'] for r in rs}),
                                          classes=dict(Counter(r['expected'].get('proposal', {}).get('kind', 'CONFIRM') for r in rs)),
                                          sha256=digest(args.output / (s + '.jsonl'))) for s, rs in splits.items()})
    (args.output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
