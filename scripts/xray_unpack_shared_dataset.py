"""Unpack the supplied split ZIP bundle on macOS without running its BAT file."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def default_folder():
    return next(p for p in (ROOT / 'preprocessed_data').iterdir()
                if p.is_dir() and unicodedata.normalize('NFC', p.name) == '전처리된데이터셋')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder', type=Path, default=default_folder())
    a = ap.parse_args()
    folder = a.folder.resolve()
    archives = sorted(folder.glob('*.zip'))
    if not archives:
        raise RuntimeError('No ZIP archives')
    destinations = {}
    info = []
    # Validate every archive before writing any member.
    for archive in archives:
        record = {'archive': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                  'files': 0, 'bytes': 0}
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in item.filename or not path.parts:
                    raise ValueError(f'Unsafe member: {item.filename}')
                if path.parts[0] != 'X-ray_데이터셋' or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError(f'Unexpected member: {item.filename}')
                if item.is_dir():
                    continue
                target = folder.joinpath(*path.parts)
                if not target.resolve().is_relative_to(folder):
                    raise ValueError(f'Outside destination: {item.filename}')
                data = z.read(item)  # validates the ZIP CRC
                digest = hashlib.sha256(data).hexdigest()
                key = unicodedata.normalize('NFC', item.filename)
                if key in destinations and destinations[key] != digest:
                    raise ValueError(f'Conflicting archives: {item.filename}')
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Existing different file: {item.filename}')
                destinations[key] = digest
                record['files'] += 1
                record['bytes'] += len(data)
        info.append(record)
    written = skipped = 0
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if item.is_dir():
                    continue
                target = folder.joinpath(*PurePosixPath(item.filename).parts)
                if target.exists():
                    skipped += 1
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as f:
                        f.write(z.read(item))
                    written += 1
                key = unicodedata.normalize('NFC', item.filename)
                assert hashlib.sha256(target.read_bytes()).hexdigest() == destinations[key]
    report = {'archives': info, 'unique_files': len(destinations), 'written': written, 'skipped_identical': skipped,
              'destination': (folder / 'X-ray_데이터셋').relative_to(ROOT).as_posix(),
              'verified_crc_and_extracted_sha256': True}
    out = ROOT / 'reports/provided_preprocessing_audit_20261002'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'extraction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
