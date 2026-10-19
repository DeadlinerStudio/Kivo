"""Freeze new entity holdout before training the normalization comparison."""
import argparse
import json
from pathlib import Path
import shutil

from .data import digest, load
from . import span_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/zh-core-spans-v1'))
    parser.add_argument('--output', type=Path, default=Path('data/zh-normalization-compare-v1'))
    args = parser.parse_args()
    source, output = args.source, args.output
    if output.exists():
        raise ValueError('Refusing to replace frozen comparison data')
    titles = ['签收补寄的毕业证明', '参加摄影作品交流', '核实奖学金到账情况', '办理自行车年检',
              '领取没有寄出的包裹', '讨论不能开门的原因', '确认有没有空余座位', '记录取消订单的经过',
              '整理后天的采访问题', '讨论知行楼供电方案', '和辅导员核对转专业材料', '修理没恢复的投影仪',
              '购买不带花生的点心', '参加社区急救培训', '归还备用充电设备', '核对展览报名名单']
    locations = ['知行楼', '敦行楼', '望江阁', '松林交流中心', '青禾社区大厅', '紫藤咖啡屋',
                 '南苑综合楼G305', '云帆大厦八楼', '城西阅览室208', '北辰学术厅',
                 '政务服务楼H106', '海棠广场', '美术馆三楼', '星河报告厅', '竹园讨论室', '澄明楼']
    old_t, old_l = span_data.TITLES['test'], span_data.LOCATIONS['test']
    try:
        span_data.TITLES['test'], span_data.LOCATIONS['test'] = titles, locations
        rows = span_data.build()['test']
    finally:
        span_data.TITLES['test'], span_data.LOCATIONS['test'] = old_t, old_l
    splits = {s: load(source / (s + '.jsonl')) for s in ['train', 'dev']}
    splits['test'] = rows
    span_data.validate(splits)
    previous = load(source / 'test.jsonl')
    for role in ['TITLE', 'LOCATION']:
        values = lambda rs: {r['text'][s['start']:s['end']] for r in rs for s in r['spans'] if s['label'] == role}
        assert not values(previous) & values(rows), role
    output.mkdir()
    for name in ['train.jsonl', 'dev.jsonl', 'regression.jsonl']:
        shutil.copyfile(source / name, output / name)
    (output / 'test.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = dict(counts={s: len(rs) for s, rs in splits.items()},
                  sha256={s: digest(output / (s + '.jsonl')) for s in splits},
                  trainDevIdenticalTo='zh-core-spans-v1', frozenBeforeNormalizerTraining=True,
                  humanReviewed=False, realUserData=False, newTitlesAndLocations=True,
                  syntax='previously exposed templates; not new syntax or real-user blind test')
    (output / 'provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
