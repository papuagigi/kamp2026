"""Recovery fault tests; no competition model is trained here."""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from xray_recovery import (RecoveryStore, atomic_torch, capture_rng, restore_rng,
                           checkpoint_epoch, dataset_digest, run_lock)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.store=RecoveryStore(self.root/'recovery',dict(model='faster',epochs=20,data='fixed'))

    def publish(self, epoch):
        p=self.root/'last.pt'
        atomic_torch(p,dict(epoch=epoch,model={'w':torch.ones(2)*epoch},optimizer={},scheduler={}))
        return self.store.publish(p,epoch,capture_rng())

    def test_latest_complete_epoch(self):
        self.publish(1);self.publish(2)
        self.assertEqual(self.store.latest()['epoch'],2)

    def test_partial_save_is_ignored(self):
        self.publish(1)
        p=self.root/'recovery/.pending-interrupted';p.mkdir();(p/'checkpoint.pt').write_bytes(b'partial')
        self.assertEqual(self.store.latest()['epoch'],1)

    def test_corrupt_latest_falls_back(self):
        self.publish(1);p=self.publish(2)
        (p/'checkpoint.pt').write_bytes(b'corrupt')
        self.assertEqual(self.store.latest()['epoch'],1)

    def test_no_valid_checkpoint_does_not_restart_silently(self):
        p=self.publish(1);(p/'rng.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError,'No verified'):self.store.latest()

    def test_failed_copy_preserves_prior(self):
        self.publish(1)
        with patch('xray_recovery.shutil.copyfileobj',side_effect=OSError('storage disconnected')):
            with self.assertRaises(OSError):self.publish(2)
        self.assertEqual(self.store.latest()['epoch'],1)

    def test_keep_two_valid_full_states(self):
        for ep in range(1,5):self.publish(ep)
        self.assertEqual([x[1]['epoch'] for x in self.store._valid()],[4,3])

    def test_changed_config_refused(self):
        with self.assertRaisesRegex(ValueError,'differs'):
            RecoveryStore(self.root/'recovery',dict(model='faster',epochs=30,data='fixed'))

    def test_dataset_bytes_affect_digest(self):
        for split in ['train','val']:
            for kind,suffix in [('images','.png'),('labels','.txt')]:
                d=self.root/split/kind;d.mkdir(parents=True)
        # Match the actual project layout.
        for kind,suffix in [('images','.png'),('labels','.txt')]:
            for split in ['train','val']:
                p=self.root/kind/split;p.mkdir(parents=True);(p/('a'+suffix)).write_bytes(b'1')
        before=dataset_digest(self.root)
        (self.root/'labels/train/a.txt').write_bytes(b'2')
        self.assertNotEqual(before,dataset_digest(self.root))

    def test_epoch_mismatch_refused(self):
        p=self.root/'last.pt';atomic_torch(p,dict(epoch=2,model={},optimizer={},scheduler={}))
        with self.assertRaisesRegex(ValueError,'epoch'):
            self.store.publish(p,3,capture_rng())

    def test_four_checkpoint_contracts_and_weights_only_rejection(self):
        payloads={
            'yolo':dict(epoch=6,optimizer={},ema=torch.nn.Linear(1,1)),
            'faster':dict(epoch=7,model={},optimizer={},scheduler={}),
            'rfdetr':dict(epoch=6,state_dict={},optimizer_states=[{}],lr_schedulers=[{}],loops={}),
            'dfine':dict(last_epoch=6,model={},optimizer={},lr_scheduler={}),
        }
        for name,payload in payloads.items():
            with self.subTest(model=name):
                p=self.root/(name+'.pt');atomic_torch(p,payload)
                self.assertEqual(checkpoint_epoch(p,name),7)
                payload.pop('optimizer_states' if name=='rfdetr' else 'optimizer')
                atomic_torch(p,payload)
                with self.assertRaises(ValueError):checkpoint_epoch(p,name)

    def test_same_os_duplicate_writer_refused(self):
        with run_lock(self.root):
            code='from xray_recovery import run_lock; from pathlib import Path\nwith run_lock(Path('+repr(str(self.root))+')): pass'
            result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('active writer',result.stderr)

    def test_rng_and_optimizer_resume_matches_uninterrupted_tiny_training(self):
        torch.manual_seed(0);random.seed(0);np.random.seed(0)
        model=torch.nn.Linear(3,1);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.8)
        generator=torch.Generator().manual_seed(0)
        def step(model,opt,scheduler):
            x=torch.rand((4,3),generator=generator)+torch.rand(1)+random.random()+float(np.random.rand())
            opt.zero_grad();loss=model(x).square().mean();loss.backward();opt.step();scheduler.step()
        step(model,opt,scheduler)
        p=self.root/'last.pt';atomic_torch(p,dict(epoch=1,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict()))
        self.store.publish(p,1,capture_rng(generator=generator))
        step(model,opt,scheduler);expected=copy.deepcopy(model.state_dict())
        saved=self.store.latest();ck=torch.load(saved['checkpoint'],weights_only=True)
        model=torch.nn.Linear(3,1);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.8)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        restore_rng(saved['rng'],generator);step(model,opt,scheduler)
        self.assertTrue(all(torch.equal(expected[k],model.state_dict()[k]) for k in expected))


if __name__=='__main__':unittest.main()
