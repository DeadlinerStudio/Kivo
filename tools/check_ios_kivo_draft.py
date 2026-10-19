"""Exercise actual Swift draft override code on the host; no simulator/device use."""
from pathlib import Path
import subprocess
import tempfile

source = Path('../Deadliner-iOS/Deadliner/Features/Main/Sheets/QuickAddDraft.swift').read_text()
# Only the external Shared enum is substituted; the real override/decoding code is used unchanged.
source = source.replace('import Shared', 'enum HabitPeriod { case weekly, monthly, ebbinghaus, once, daily }', 1)
checks = '''
var draft = QuickAddDraft()
draft.title = "开会"
draft.ready = true
draft.needsAI = true
draft.source = "MODEL"
draft.recognitionState = "READY_WITH_AI"
var overrides = QuickAddOverrides()
overrides.dueAt = Date().addingTimeInterval(86400)
let low = overrides.applying(to: draft)
precondition(low.ready && low.needsAI && low.title == "开会", "Deadline edit hid Kivo uncertainty")
draft.ready = false
draft.recognitionState = "KIVO_PARTIAL"
let partial = overrides.applying(to: draft)
precondition(!partial.ready && partial.needsAI, "Deadline edit accepted a hard-failed Kivo draft")
draft.recognitionState = nil
draft.source = "LOCAL"
let legacy = overrides.applying(to: draft)
precondition(legacy.ready && !legacy.needsAI, "Legacy manual entry changed")
print("Swift Kivo override checks passed")
'''
with tempfile.TemporaryDirectory(prefix='kivo-swift-') as directory:
    root = Path(directory)
    (root / 'main.swift').write_text(source + checks)
    subprocess.run(['swiftc', '-module-cache-path', str(root / 'module-cache'),
                    str(root / 'main.swift'), '-o', str(root / 'check')], check=True)
    subprocess.run([str(root / 'check')], check=True)
