"""Build once, fingerprint exact sources and native executable for repeated corpus runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .data import digest


def source_digest(core):
    return hashlib.sha256(b''.join(str(p.relative_to(core)).encode()+p.read_bytes() for p in sorted((core/'shared/src').rglob('*.kt')) if 'quickadd' in str(p))).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();core=args.core.resolve()
    before=source_digest(core)
    result=subprocess.run([str(core/'gradlew'),':shared:linkDebugTestMacosArm64','-x',':shared:generateKmpBuildMetadata','--offline','--no-daemon','--console=plain'],cwd=core,env=os.environ,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.build.log').write_text(result.stdout)
    if result.returncode or before!=source_digest(core):raise SystemExit('Adapter build failed or sources changed')
    binary=core/'shared/build/bin/macosArm64/debugTest/test.kexe'
    args.output.write_text(json.dumps(dict(core=str(core),sourceSha256=before,binary=str(binary),binarySha256=digest(binary)),indent=2))
    print(args.output)

if __name__=='__main__':main()
