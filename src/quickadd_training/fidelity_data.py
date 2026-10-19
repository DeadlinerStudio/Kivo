"""Third entity holdout, frozen before fidelity-policy calibration or evaluation."""
import argparse
import json
from pathlib import Path
import shutil

from . import span_data
from .data import digest, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/zh-fidelity-compare-v1'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to replace a frozen comparison')
    titles = ['领取重新开具的证明', '办理设备报废登记', '确认没有录入的成绩', '讨论取消预约的处理办法',
              '记录不允许入场的原因', '整理明天的培训清单', '核对交流会参会名单', '归还校庆借用的服装',
              '预约无法启动的电脑维修', '和同学确认课程替换方案', '检查有没有准备足够材料', '购买无糖代餐粉',
              '参加校友职业分享', '签收遗漏的实习材料', '讨论清晖楼消防方案', '核实未到账的补助']
    locations = ['柏园阅览室317', '教学服务楼J209', '云岭大厦16层', '清晖楼', '笃志楼', '听雨学术馆',
                 '文津讨论室', '翠竹社区接待处', '梧桐报告厅', '东篱活动中心', '北苑展览馆二楼',
                 '新行政楼K512', '青云服务站', '西山咖啡屋', '求真楼B407', '星澜广场']
    old_t, old_l = span_data.TITLES['test'], span_data.LOCATIONS['test']
    try:
        span_data.TITLES['test'], span_data.LOCATIONS['test'] = titles, locations
        rows = span_data.build()['test']
    finally:
        span_data.TITLES['test'], span_data.LOCATIONS['test'] = old_t, old_l
    source = Path('data/zh-core-spans-v1')
    splits = {s: load(source / (s + '.jsonl')) for s in ['train', 'dev']}
    splits['test'] = rows
    span_data.validate(splits)
    for previous in ['zh-core-spans-v1', 'zh-normalization-compare-v1']:
        previous_rows = load(Path('data') / previous / 'test.jsonl')
        for role in ['TITLE', 'LOCATION']:
            values = lambda rs: {r['text'][s['start']:s['end']] for r in rs for s in r['spans'] if s['label'] == role}
            if values(previous_rows) & values(rows):
                raise ValueError('Entity overlaps exposed evaluation: ' + previous + '/' + role)
    args.output.mkdir(parents=True)
    for name in ['train.jsonl', 'dev.jsonl', 'regression.jsonl']:
        shutil.copyfile(source / name, args.output / name)
    (args.output / 'test.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = dict(counts={s: len(rs) for s, rs in splits.items()},
                  sha256={s: digest(args.output / (s + '.jsonl')) for s in splits},
                  humanReviewed=False, realUserData=False, syntax='previously exposed templates',
                  titleAndLocationDisjointFromTrainDevAndBothPreviousTests=True,
                  frozenBeforeFidelityPolicyCalibration=True, generatorSha256=digest(Path(__file__)),
                  training='no new training; all model weights fixed', trainDevIdenticalTo='zh-core-spans-v1')
    (args.output / 'provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
