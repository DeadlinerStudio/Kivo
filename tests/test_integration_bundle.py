import json
from pathlib import Path
import tempfile
import unittest

from quickadd_training.integration_bundle import GRAPHS, build_bundle, digest, verify_bundle


class BundleIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'model'
        self.onnx = self.root / 'onnx'
        self.runtime = self.root / 'runtime'
        self.prototype = self.root / 'prototype.json'
        self.output = self.root / 'output'
        def asset(root, name):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            return digest(path)
        config = dict(modelDirectory=str(self.model), modelSha256=asset(self.model, 'joint.pt'),
                      nativeSha256=asset(self.runtime, 'shared/build/bin/macosArm64/debugTest/test.kexe'),
                      policySources={'policy.py': asset(self.runtime / 'src/quickadd_training', 'policy.py')},
                      headSha256={'quality.pt': asset(self.root, 'quality.pt')},
                      modelAssetSha256={'tokenizer/tokenizer.json': asset(self.model, 'tokenizer/tokenizer.json')},
                      calibration={}, thresholds={})
        self.prototype.write_text(json.dumps(config))
        graphs = {n: dict(sha256=asset(self.onnx, n), bytes=(self.onnx / n).stat().st_size) for n in GRAPHS}
        parity = dict(prototypeSha256=digest(self.prototype), nativeSha256=config['nativeSha256'],
                      graphs=graphs, completeProposalDifferences=[], examples=14, runtime='test fixture')
        (self.onnx / 'parity.json').write_text(json.dumps(parity))

    def build(self):
        return build_bundle(self.prototype, self.onnx, self.runtime, self.output)

    def test_changed_onnx_is_rejected_before_creating_output(self):
        (self.onnx / 'encoder-spans.onnx').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.build()
        self.assertFalse(self.output.exists())

    def test_changed_decoder_cannot_reuse_calibration(self):
        (self.runtime / 'src/quickadd_training/policy.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.build()
        self.assertFalse(self.output.exists())

    def test_stale_or_failed_parity_cannot_create_bundle(self):
        path = self.onnx / 'parity.json'
        parity = json.loads(path.read_text())
        for field, value in [('prototypeSha256', 'stale'), ('completeProposalDifferences', ['case-1'])]:
            path.write_text(json.dumps(dict(parity, **{field: value})))
            with self.assertRaisesRegex(ValueError, 'evidence'):
                self.build()
            self.assertFalse(self.output.exists())

    def test_verifier_detects_tampering_extra_files_and_path_escape(self):
        self.build()
        verify_bundle(self.output)
        extra = self.output / 'extra.pt'
        extra.write_bytes(b'unexpected')
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            verify_bundle(self.output)
        extra.unlink()
        path = self.output / 'kivo-preview.json'
        manifest = json.loads(path.read_text())
        manifest['assets']['../quality.pt'] = dict(sha256=digest(self.root / 'quality.pt'), bytes=10)
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'outside root'):
            verify_bundle(self.output)

    def test_existing_bundle_is_preserved(self):
        self.build()
        before = digest(self.output / 'kivo-preview.json')
        with self.assertRaisesRegex(ValueError, 'replace'):
            self.build()
        self.assertEqual(digest(self.output / 'kivo-preview.json'), before)
