"""Short role-specific RF-DETR adaptation with archived full-state epoch recovery."""
import argparse
import json
import os
import time
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
from xray_config import ROOT, V2_SPLIT_MD5
from xray_model_io import seed_all
from xray_recovery import RecoveryStore, atomic_json, capture_rng, restore_rng, sha256, dataset_digest, run_lock


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['candidate','miss'],required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();seed_all()
    from rfdetr import RFDETRSmall
    import rfdetr.training as training
    from pytorch_lightning.callbacks import Checkpoint
    data=ROOT/'data/xray_review_roles_20261005'/a.role
    out=ROOT/'runs'/f'v2_review_{a.role}_rfdetr_mps_20261005'
    checkpoint=ROOT/'runs/v2_common20_rfdetr_mps_20261004/selected/best.ckpt'
    binding=dict(model='rfdetr',role=a.role,initial_checkpoint_sha256=sha256(checkpoint),data_sha256=dataset_digest(data),
        split_md5=V2_SPLIT_MD5,seed=0,epochs=3,batch=4,lr=1e-5,lr_encoder=1e-6,
        freeze_encoder=True,script_sha256=sha256(__file__),purpose='exploratory role-input adaptation; not final specialist selection')
    with run_lock(out):
        store=RecoveryStore(out/'recovery',binding);saved=store.latest()
        if saved and not a.resume:raise FileExistsError('Use --resume')
        if saved and saved['epoch']>=3:
            print('Already complete');return
        model=RFDETRSmall.from_checkpoint(str(checkpoint),device='mps',resolution=512,
            amp=False,fused_optimizer=False,freeze_encoder=True)
        state=dict(binding,status='running',completed_epochs=saved['epoch'] if saved else 0,started_unix=time.time())
        atomic_json(out/'execution.json',state)
        class SaveRole(Checkpoint):
            def on_train_start(self,trainer,module):
                if saved:restore_rng(saved['rng'])
            def on_train_epoch_end(self,trainer,module):
                ep=int(trainer.current_epoch)+1;native=out/f'checkpoint_{ep-1}.ckpt'
                if not native.exists():raise RuntimeError('Missing archived native checkpoint')
                store.publish(native,ep,capture_rng('mps'))
                state.update(completed_epochs=ep,checkpoint=str(native.relative_to(ROOT)),updated_unix=time.time())
                atomic_json(out/'execution.json',state)
                print('ROLE_EPOCH',a.role,ep,flush=True)
        original=training.build_trainer
        def builder(*args,**kwargs):
            trainer=original(*args,**kwargs);trainer.callbacks.append(SaveRole());return trainer
        training.build_trainer=builder
        try:
            model.train(dataset_dir=str(data),dataset_file='yolo',output_dir=str(out),epochs=3,
                batch_size=4,grad_accum_steps=1,device='mps',num_workers=0,seed=0,
                resolution=512,multi_scale=False,expanded_scales=False,aug_config={},
                lr=1e-5,lr_encoder=1e-6,weight_decay=1e-4,use_ema=False,amp_dtype=None,
                tensorboard=False,wandb=False,run_test=False,early_stopping=False,
                checkpoint_interval=1,progress_bar=None,compute_val_loss=False,
                resume=str(saved['checkpoint']) if saved else '')
            state['status']='complete'
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));raise
        finally:
            training.build_trainer=original
            state['seconds']=time.time()-state['started_unix'];atomic_json(out/'execution.json',state)

if __name__=='__main__':main()
