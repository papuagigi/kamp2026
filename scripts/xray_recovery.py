"""Validated epoch-boundary recovery for trusted project checkpoints.

Publish to the run directory (on mounted Drive for Colab). A pending directory
is never a recovery candidate. Keep two full states so a damaged newest state
does not destroy the previous one. This does not reconnect a cloud runtime.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import uuid

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    with Path(source).open('rb') as src, temp.open('wb') as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    temp.replace(destination)


@contextmanager
def run_lock(directory):
    """Prevent duplicate writers in the same OS; not a cross-VM Drive lease."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Keep advisory locks on the local filesystem, not Google Drive's FUSE mount.
    locks = Path(tempfile.gettempdir()) / 'xray-training-locks'
    locks.mkdir(exist_ok=True)
    key = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()
    with (locks / (key + '.lock')).open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This run already has an active writer') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def capture_rng(device='cpu', generator=None):
    ns = np.random.get_state()
    state = dict(python=random.getstate(), numpy=(ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
                 torch=torch.get_rng_state(), device=str(device))
    if generator is not None:
        state['loader'] = generator.get_state()
    if str(device).startswith('cuda'):
        state['cuda'] = torch.cuda.get_rng_state_all()
    if str(device) == 'mps':
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng(state, generator=None):
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state['torch'])
    if generator is not None and 'loader' in state:
        generator.set_state(state['loader'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state:
        torch.mps.set_rng_state(state['mps'])


def checkpoint_epoch(path, model):
    # YOLO stores its model object; all files here must be our trusted outputs.
    ck = torch.load(path, map_location='cpu', weights_only=model != 'yolo')
    keys = {
        'yolo': ['epoch', 'optimizer', 'ema'],
        'faster': ['epoch', 'model', 'optimizer', 'scheduler'],
        'rfdetr': ['epoch', 'state_dict', 'optimizer_states', 'lr_schedulers', 'loops'],
        'dfine': ['last_epoch', 'model', 'optimizer', 'lr_scheduler'],
    }[model]
    if any(k not in ck or ck[k] is None for k in keys):
        raise ValueError('Incomplete training checkpoint; best/weights-only files cannot resume')
    if model == 'rfdetr' and not ck['optimizer_states']:
        raise ValueError('Lightning optimizer state missing')
    epoch = int(ck['last_epoch'] if model == 'dfine' else ck['epoch'])
    if model != 'faster':
        epoch += 1
    if epoch < 1:
        raise ValueError('No completed epoch in checkpoint')
    return epoch


def dataset_digest(directory):
    """Read train/val images and labels, never test; paths are portable."""
    directory = Path(directory)
    h = hashlib.sha256()
    for split in ['train', 'val']:
        for kind in ['images', 'labels']:
            files = sorted((directory / kind / split).glob('*'))
            files = [p for p in files if p.suffix in {'.png', '.txt'}]
            if not files:
                raise ValueError(f'Missing {split}/{kind}')
            for p in files:
                h.update(str(p.relative_to(directory)).encode())
                h.update(sha256(p).encode())
    return h.hexdigest()


class RecoveryStore:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        binding = self.directory / 'binding.json'
        if binding.exists():
            if json.loads(binding.read_text())['fingerprint'] != self.fingerprint:
                raise ValueError('Resume configuration/data/code/version differs; use a new run name')
        else:
            atomic_json(binding, dict(config=config, fingerprint=self.fingerprint))

    def publish(self, checkpoint, epoch, rng):
        if checkpoint_epoch(checkpoint, self.config['model']) != epoch:
            raise ValueError('Checkpoint epoch does not match commit epoch')
        pending = self.directory / ('.pending-' + uuid.uuid4().hex)
        pending.mkdir()
        files = {}
        source = Path(checkpoint)
        name = 'checkpoint' + source.suffix
        with source.open('rb') as src, (pending / name).open('wb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        if sha256(source) != sha256(pending / name):
            raise IOError('Checkpoint copy verification failed')
        atomic_torch(pending / 'rng.pt', rng)
        for filename in [name, 'rng.pt']:
            p = pending / filename
            files[filename] = dict(bytes=p.stat().st_size, sha256=sha256(p))
        manifest = dict(epoch=epoch, fingerprint=self.fingerprint, checkpoint=name, files=files)
        atomic_json(pending / 'manifest.json', manifest)
        final = self.directory / f'epoch-{epoch:04d}-{uuid.uuid4().hex}'
        pending.replace(final)
        # Only prune this module's committed, verified generations, never native archives.
        valid = self._valid()
        for old, _ in valid[2:]:
            shutil.rmtree(old)
        return final

    def _valid(self):
        valid = []
        for p in self.directory.glob('epoch-*'):
            try:
                m = json.loads((p / 'manifest.json').read_text())
                if m['fingerprint'] != self.fingerprint:
                    continue
                if set(m['files']) != {m['checkpoint'], 'rng.pt'}:
                    continue
                for name, info in m['files'].items():
                    if Path(name).name != name:
                        raise ValueError('Invalid snapshot member')
                    f = p / name
                    if f.stat().st_size != info['bytes'] or sha256(f) != info['sha256']:
                        raise ValueError('Damaged snapshot')
                valid.append((p, m))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(valid, key=lambda x: (x[1]['epoch'], x[0].name), reverse=True)

    def latest(self):
        candidates = self._valid()
        if not candidates:
            if list(self.directory.glob('epoch-*')):
                raise RuntimeError('No verified recovery checkpoint remains')
            return None
        p, m = candidates[0]
        return dict(epoch=m['epoch'], checkpoint=p / m['checkpoint'],
                    rng=torch.load(p / 'rng.pt', map_location='cpu', weights_only=True))
