import argparse
from pathlib import Path
from .data import load, validate, digest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path,nargs='?',default=Path('data'))
    args=parser.parse_args()
    splits={s:load(args.directory/(s+'.jsonl')) for s in ('train','dev','test')}
    validate(splits)
    for split,rows in splits.items():
        print(split,len(rows),digest(args.directory/(split+'.jsonl')))

if __name__=='__main__':
    main()
