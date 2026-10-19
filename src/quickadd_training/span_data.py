"""Chinese core-field span experiment. Authored synthetic data, never human gold."""
import argparse
import datetime as dt
import json
import random
from pathlib import Path
from zoneinfo import ZoneInfo

from .corpus import BASE, TITLE_POOL
from .data import digest, validate as validate_business

ROLES = ('TITLE', 'TIME', 'LOCATION', 'NOTE', 'CONNECTOR')
PROVENANCE = 'assistant-authored-core-spans-not-human-reviewed'
TITLES = {
    'train': TITLE_POOL['zh'][:48] + [
        '开会', '讨论作业', '参加培训', '提交申请表', '归还设备', '办理手续',
        '参加面试', '领取材料', '讨论项目方案', '检查有没有库存', '退掉不合适的衣服',
        '记录取消原因', '确认没完成的工作', '讨论不能上线的原因', '检查不合格的产品',
        '确认有没有空教室', '整理明天的演讲稿', '讨论图书馆改造方案', '记录昨天的问题',
        '和老师讨论论文', '与同事核对预算', '给爸爸打电话', '取快递', '修复登录问题',
        '更换不能使用的电池', '处理没有签字的合同', '确认不参加活动的人数',
        '领取体检报告', '购买学习资料', '准备项目答辩', '核对报名信息', '检查消防设施'],
    'dev': ['参加小组研讨', '核对租赁账单', '领取补办的学生证', '检查没有登记的资料',
            '记录不能报销的原因', '讨论礼堂维修方案', '整理后天的交流提纲',
            '和导师核对实验安排', '归还借用的镜头', '办理课程退选', '确认有没有剩余名额', '领回修好的电脑'],
    'test': ['参加读书分享会', '办理证件补办', '归还租借的设备', '核对社团活动经费',
             '讨论逸夫楼改造方案', '检查没有归档的凭证', '记录不予受理的原因',
             '确认有没有预约成功', '整理明天的竞赛材料', '和学长讨论实习安排',
             '领回遗失的雨伞', '参加志愿者说明会', '更新不能访问的链接', '检查没修好的打印机',
             '购买不含糖的饮料', '讨论取消活动的影响'],
}
LOCATIONS = {
    'train': ['图书馆', '行政楼', '实验楼', '科技楼', '会议室', '办公室', '体育馆', '食堂',
              '校医院', '教学楼', '报告厅', '服务中心', '创新楼', '活动中心', '东门', '南门',
              '一号楼203室', '二号楼301室', '三楼会议室', '学院办公室', '东校区', '西校区',
              '华东大厦', '人民路营业厅', '社区服务站', '综合楼B205', '教学楼A308',
              '创业园', '车站大厅', '咖啡店', '书店', '公司前台'],
    'dev': ['明德楼', '博雅楼', '学生活动大厅', '北门接待室', '研发中心C307', '社区阅览室',
            '城南办事处', '市民中心二楼', '新校区展厅', '办公楼D402', '西苑食堂', '江滨书屋'],
    'test': ['求是楼', '致远楼', '文澜阁', '学术交流中心', '东湖服务大厅', '梅园咖啡馆',
             '南区实验室E406', '锦城大厦12层', '新图书馆206室', '北校区礼堂',
             '综合服务楼F102', '银杏广场', '艺术中心二楼', '红枫报告厅', '湖畔自习室', '清源楼'],
}
NOTES = ['带身份证', '带上小票', '提前十分钟到场', '记得带纸笔', '携带申请材料', '带上电脑', '准备好相关凭证']
LAYOUTS = {
    'train': ['time-modal-loc-title', 'time-loc-title', 'loc-title-time', 'title-time-loc',
              'request-time-loc-title', 'time-title-explicit-loc', 'title-loc-time',
              'day-title-clock-loc', 'day-loc-title-clock', 'time-loc-title-note'],
    'dev': ['time-modal-loc-title', 'time-loc-title', 'loc-title-time', 'title-time-loc',
            'request-time-loc-title', 'time-title-explicit-loc', 'title-loc-time',
            'day-title-clock-loc', 'day-loc-title-clock', 'time-loc-title-note'],
    'test': ['time-modal-loc-title', 'time-loc-title', 'loc-title-time', 'title-time-loc',
             'request-time-loc-title', 'time-title-explicit-loc', 'title-loc-time',
             'day-title-clock-loc', 'day-loc-title-clock', 'time-loc-title-note',
             'held-location-first', 'held-request-final', 'held-location-between-time'],
}


def make_row(parts, row_id, split, category, proposal, now='2026-10-19T04:00:00Z',
             timezone='Asia/Shanghai', family=None):
    text, spans = '', []
    for token, label in parts:
        start = len(text)
        text += token
        if label and token:
            spans.append(dict(start=start, end=len(text), label=label))
    return dict(id=row_id, locale='zh', split=split, group=family or row_id, category=category,
                text=text, spans=spans, preferredKind='TASK', now=now, timezone=timezone,
                provenance=PROVENANCE, annotationStatus='synthetic',
                expected=dict(decision='accept', proposal=proposal) if proposal else dict(decision='defer'))


def temporal(rng):
    now = rng.choice(['2026-10-19T04:00:00Z', '2026-12-31T15:30:00Z', '2028-02-28T02:00:00Z'])
    zone = rng.choice(['Asia/Shanghai', 'UTC'])
    today = dt.datetime.fromisoformat(now.replace('Z', '+00:00')).astimezone(ZoneInfo(zone)).date()
    offset = rng.randrange(4)
    day = ['今天', '明天', '后天', '大后天'][offset]
    hour = rng.choice([7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21])
    minute = rng.choice([0, 0, 0, 15, 30, 45])
    gap = rng.choice(['', '', ' ', '  ', '\t', '\u3000'])
    if rng.random() < .25:
        clock = f'{hour:02}:{minute:02}'
    else:
        part = '上午' if hour < 12 else '下午' if hour < 18 else '晚上'
        h = hour if hour < 12 else hour - 12
        numeral = str(h) if rng.random() < .65 else ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一'][h]
        tail = '' if minute == 0 else '半' if minute == 30 else '一刻' if minute == 15 else '四十五分'
        clock = part + gap + numeral + gap + '点' + tail
    if rng.random() < .12:
        day = (today + dt.timedelta(days=offset)).isoformat()
        separator = ' '
    else:
        separator = ''
    return day, clock, separator, (today + dt.timedelta(days=offset)).isoformat() + f'T{hour:02}:{minute:02}', now, zone


def layout_parts(layout, title, location, day, clock, separator, rng):
    t, l, time = (title, 'TITLE'), (location, 'LOCATION'), (day + separator + clock, 'TIME')
    layouts = {
        'time-modal-loc-title': [time, (rng.choice(['要在', '准备在', '打算在', '需要在']), 'CONNECTOR'), l, t],
        'time-loc-title': [time, ('在', 'CONNECTOR'), l, t],
        'loc-title-time': [('在', 'CONNECTOR'), l, t, ('，', None), time],
        'title-time-loc': [t, ('，', None), time, ('在', 'CONNECTOR'), l],
        'request-time-loc-title': [(rng.choice(['提醒我', '请提醒我', '麻烦提醒我', '记得', '帮我安排']), 'CONNECTOR'), time, ('在', 'CONNECTOR'), l, t],
        'time-title-explicit-loc': [time, t, ('，地点：', 'CONNECTOR'), l],
        'title-loc-time': [t, ('，安排在', 'CONNECTOR'), l, ('，时间是', 'CONNECTOR'), time],
        'day-title-clock-loc': [(day, 'TIME'), t, ('，', None), (clock, 'TIME'), ('在', 'CONNECTOR'), l],
        'day-loc-title-clock': [(day, 'TIME'), ('在', 'CONNECTOR'), l, t, ('，', None), (clock, 'TIME')],
        'time-loc-title-note': [time, ('在', 'CONNECTOR'), l, t, ('，备注：', 'CONNECTOR'), (rng.choice(NOTES), 'NOTE')],
        'held-location-first': [('地点定在', 'CONNECTOR'), l, ('，', None), time, t],
        'held-request-final': [('请把', 'CONNECTOR'), t, ('安排到', 'CONNECTOR'), day and (day, 'TIME'), ('的', 'CONNECTOR'), (clock, 'TIME'), ('，地点在', 'CONNECTOR'), l],
        'held-location-between-time': [(day, 'TIME'), ('，在', 'CONNECTOR'), l, ('，', None), (clock, 'TIME'), t],
    }
    return layouts[layout]


def build(count=5000, seed=20261019):
    rng = random.Random(seed)
    splits = {}
    for split, total in [('train', count), ('dev', 360), ('test', 480)]:
        rows, seen = [], set()
        while len(rows) < total:
            title = rng.choice(TITLES[split])
            location = rng.choice(LOCATIONS[split])
            day, clock, sep, due, now, zone = temporal(rng)
            mode = len(rows) % 10
            layout = rng.choice(LAYOUTS[split])
            parts = layout_parts(layout, title, location, day, clock, sep, rng)
            note = next((s for s, role in parts if role == 'NOTE'), '')
            proposal = dict(BASE, title=title, dueAt=due, note='地点：' + location + ('；' + note if note else ''))
            category = 'held-syntax' if layout.startswith('held-') else 'scheduled-location'
            if mode == 6:
                parts = [(rng.choice(['', '提醒我', '记得', '请', '麻烦', '请帮我', '帮我', '我要', '打算', '准备']), 'CONNECTOR'), (title, 'TITLE')]
                proposal = dict(BASE, title=title)
                category = 'plain-title'
            elif mode == 7:
                parts = [(day + sep + clock, 'TIME'), (title, 'TITLE')]
                proposal = dict(BASE, title=title, dueAt=due)
                category = 'scheduled-no-location'
            elif mode in [8, 9]:
                control = rng.randrange(5)
                if control == 0:
                    parts = [(rng.choice(['不要', '别', '不用', '取消']), 'CONNECTOR')] + parts
                    category = 'negation'
                elif control == 1:
                    parts = [(rng.choice(['如果有空就', '看情况再', '也许会']), 'CONNECTOR')] + parts
                    category = 'conditional'
                elif control == 2:
                    parts += [('然后', 'CONNECTOR'), (rng.choice(TITLES[split]), 'TITLE')]
                    category = 'multiple-events'
                elif control == 3:
                    parts = [('过几天', 'TIME'), ('在', 'CONNECTOR'), (location, 'LOCATION'), (title, 'TITLE')]
                    category = 'uncertain-time'
                else:
                    parts = [(day + sep + clock, 'TIME'), ('在', 'CONNECTOR'), (location, 'LOCATION')]
                    category = 'missing-event'
                proposal = None
            row = make_row(parts, f'span-{split}-{len(rows):05}', split, category, proposal, now, zone)
            if row['text'] in seen:
                continue
            seen.add(row['text'])
            rows.append(row)
        splits[split] = rows
    return splits


def regression():
    """Known user examples and minimal pairs; never counted as independent holdout."""
    cases = [
        ([('明天下午 4 点', 'TIME'), ('要在', 'CONNECTOR'), ('逸夫楼', 'LOCATION'), ('开会', 'TITLE')], '开会', '2026-10-20T16:00', '地点：逸夫楼'),
        ([('明天下午4点', 'TIME'), ('在', 'CONNECTOR'), ('逸夫楼', 'LOCATION'), ('开会', 'TITLE')], '开会', '2026-10-20T16:00', '地点：逸夫楼'),
        ([('在', 'CONNECTOR'), ('逸夫楼', 'LOCATION'), ('明天下午4点', 'TIME'), ('开会', 'TITLE')], '开会', '2026-10-20T16:00', '地点：逸夫楼'),
        ([('明天下午4点', 'TIME'), ('开会', 'TITLE'), ('，地点', 'CONNECTOR'), ('逸夫楼', 'LOCATION')], '开会', '2026-10-20T16:00', '地点：逸夫楼'),
        ([('明天下午4点', 'TIME'), ('开会', 'TITLE'), ('，备注：地点', 'CONNECTOR'), ('逸夫楼', 'LOCATION')], '开会', '2026-10-20T16:00', '地点：逸夫楼'),
        ([('明天', 'TIME'), ('在', 'CONNECTOR'), ('逸夫楼', 'LOCATION'), ('开会', 'TITLE'), ('，', None), ('下午四点', 'TIME')], '开会', '2026-10-20T16:00', '地点：逸夫楼'),
        ([('明天下午  4  点', 'TIME'), ('要在', 'CONNECTOR'), ('逸夫楼', 'LOCATION'), ('开会', 'TITLE')], '开会', '2026-10-20T16:00', '地点：逸夫楼'),
        ([('明天下午4点', 'TIME'), ('讨论逸夫楼改造方案', 'TITLE')], '讨论逸夫楼改造方案', '2026-10-20T16:00', ''),
        ([('明天下午4点', 'TIME'), ('退掉不合适的衣服', 'TITLE')], '退掉不合适的衣服', '2026-10-20T16:00', ''),
        ([('明天下午4点', 'TIME'), ('检查商品有没有库存', 'TITLE')], '检查商品有没有库存', '2026-10-20T16:00', ''),
        ([('明天下午4点', 'TIME'), ('记录取消原因', 'TITLE')], '记录取消原因', '2026-10-20T16:00', ''),
        ([('明天下午4点', 'TIME'), ('整理明天的演讲稿', 'TITLE')], '整理明天的演讲稿', '2026-10-20T16:00', ''),
        ([('不要', 'CONNECTOR'), ('明天下午4点', 'TIME'), ('在', 'CONNECTOR'), ('逸夫楼', 'LOCATION'), ('开会', 'TITLE')], None, None, ''),
        ([('如果有空', 'CONNECTOR'), ('明天下午4点', 'TIME'), ('在', 'CONNECTOR'), ('逸夫楼', 'LOCATION'), ('开会', 'TITLE')], None, None, ''),
        ([('明天下午4点', 'TIME'), ('在', 'CONNECTOR'), ('逸夫楼', 'LOCATION'), ('开会', 'TITLE'), ('然后', 'CONNECTOR'), ('取快递', 'TITLE')], None, None, ''),
        ([('明天下午', 'TIME'), ('在', 'CONNECTOR'), ('逸夫楼', 'LOCATION'), ('开会', 'TITLE')], None, None, ''),
    ]
    return [make_row(parts, f'span-regression-{i:03}', 'dev', 'known-regression',
                     dict(BASE, title=title, dueAt=due, note=note) if title else None)
            for i, (parts, title, due, note) in enumerate(cases)]


def validate(splits):
    adapted = {s: [dict(r, spans=[v for v in r['spans'] if v['label'] not in ['LOCATION', 'CONNECTOR']]) for r in rows]
               for s, rows in splits.items()}
    validate_business(adapted)
    reserved = {s: {role: set() for role in ['TITLE', 'LOCATION']} for s in splits}
    for split, rows in splits.items():
        for row in rows:
            occupied = set()
            if not 1 <= len(row['text']) <= 126:
                raise ValueError('Text outside prototype context')
            for span in row['spans']:
                a, b = span['start'], span['end']
                if span['label'] not in ROLES or not 0 <= a < b <= len(row['text']):
                    raise ValueError('Invalid span')
                positions = set(range(a, b))
                if occupied & positions:
                    raise ValueError('Overlapping gold spans')
                occupied |= positions
                if span['label'] in reserved[split]:
                    reserved[split][span['label']].add(row['text'][a:b])
            if row['expected']['decision'] == 'accept':
                titles = [row['text'][s['start']:s['end']] for s in row['spans'] if s['label'] == 'TITLE']
                if titles != [row['expected']['proposal']['title']]:
                    raise ValueError('Title mismatch')
    keys = list(splits)
    for i, split in enumerate(keys):
        for other in keys[i + 1:]:
            for role in ['TITLE', 'LOCATION']:
                if reserved[split][role] & reserved[other][role]:
                    raise ValueError(f'Entity leakage: {split}/{other}/{role}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/zh-core-spans-v1'))
    parser.add_argument('--train-count', type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to replace an existing experiment corpus')
    splits = build(args.train_count)
    validate(splits)
    args.output.mkdir(parents=True)
    for split, rows in splits.items():
        (args.output / (split + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (args.output / 'regression.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in regression()))
    report = dict(seed=20261019, counts={s: len(rs) for s, rs in splits.items()},
                  sha256={s: digest(args.output / (s + '.jsonl')) for s in splits},
                  humanReviewed=False, realUserData=False, entitiesDisjoint=True,
                  syntaxHoldout='three test-only layouts; other layouts shared',
                  regression='known exposed examples; not holdout', roles=ROLES)
    (args.output / 'provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
