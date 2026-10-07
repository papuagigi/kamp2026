"""Import the integrity-checked Colab transfer without replacing local results."""
import json
import shutil
import time
import zipfile
from pathlib import Path
from xray_config import ROOT
from xray_recovery import atomic_copy, atomic_json, sha256

REPORT = ROOT / 'reports/direct_training_20261006'
RUN = 'v2_direct20_rfdetr_cuda_20261006'
PARTS = [
    (200000000, '23c64467ada2997c26da728132f22ce5cbe82e684d271cff9ab92731c6a74c84'),
    (200000000, 'ee88f66b41cea079fff2829f661c38d6dec873bc2f14b3d71231ca685b37c995'),
    (200000000, 'be6636da7f3823c9e4c93d800c4d3d212622f351d94ce14a3621a45318d970f9'),
    (193122163, '1841da0c0eaa9188ca52f868275bac7ecbefcddcc19d26fdd86af623d1764d30'),
]
ARCHIVE_SHA = 'ceb449c14d361f187c8cb17184e48506189fa903ac06288be2130f834c16a832'


def main():
    deadline = time.monotonic() + 1800
    paths = [REPORT / f'xray_mps_transfer_20261007.part{i:02d}' for i in range(4)]
    while not all(p.exists() and p.stat().st_size == size for p, (size, _) in zip(paths, PARTS)):
        if time.monotonic() > deadline:
            raise TimeoutError('Download incomplete; original Colab process untouched')
        time.sleep(5)
    for path, (_, digest) in zip(paths, PARTS):
        assert sha256(path) == digest, f'Damaged transfer part: {path.name}'
    archive = REPORT / 'xray_mps_transfer_20261007.zip'
    if not archive.exists():
        with archive.with_suffix('.pending').open('wb') as dest:
            for path in paths:
                with path.open('rb') as src:
                    shutil.copyfileobj(src, dest)
        archive.with_suffix('.pending').replace(archive)
    assert sha256(archive) == ARCHIVE_SHA
    backup = REPORT / 'colab_transfer_backup'
    copied = []
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('TRANSFER_MANIFEST.json'))
        assert manifest['run'] == RUN and manifest['epoch'] == 14
        assert set(z.namelist()) == set(manifest['files']) | {'TRANSFER_MANIFEST.json'}
        for name, info in manifest['files'].items():
            rel = Path(name)
            assert not rel.is_absolute() and '..' not in rel.parts
            if rel.parts[0] == 'runs':
                assert len(rel.parts) >= 3 and rel.parts[1] == RUN
                dest = ROOT / rel
            else:
                assert len(rel.parts) == 1
                dest = backup / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                pending = dest.with_name(dest.name + '.importing')
                with z.open(name) as src, pending.open('wb') as out:
                    shutil.copyfileobj(src, out)
                assert pending.stat().st_size == info['bytes'] and sha256(pending) == info['sha256']
                pending.replace(dest)
            assert dest.stat().st_size == info['bytes'] and sha256(dest) == info['sha256']
            copied.append(str(dest.relative_to(ROOT)))
    run = ROOT / 'runs' / RUN
    selection = json.loads((run / 'selection.json').read_text())
    record = json.loads((run / 'validation' / f"epoch_{selection['best_epoch']:03d}.json").read_text())
    selected = ROOT / selection['checkpoint']
    native = ROOT / record['checkpoint']
    assert native.is_relative_to(run) and sha256(selected) == record['checkpoint_sha256']
    if not native.exists():
        atomic_copy(selected, native)
    assert sha256(native) == record['checkpoint_sha256']
    atomic_json(REPORT / 'mps_transfer_import.json', dict(status='verified', epoch=14,
                archive_sha256=ARCHIVE_SHA, files=copied, fingerprint=manifest['fingerprint'],
                selected_native_copy=str(native.relative_to(ROOT)), imported_unix=time.time()))
    print('MPS_TRANSFER_IMPORTED', len(copied), 'epoch', 14, flush=True)


if __name__ == '__main__':
    main()
