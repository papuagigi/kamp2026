"""Create a local, blind-review source/data/checkpoint package without changing inputs."""
import ast
import hashlib
import io
import json
import pickletools
import re
import struct
import zipfile
from pathlib import Path
from xray_config import ROOT

OUT=ROOT/'outputs/submission_review_20261004'
BASE=ROOT/'reports/design_review_20261004'


def sha(data):return hashlib.sha256(data).hexdigest()


def clean_text(text):
    text=text.replace(str(ROOT)+'/', '').replace(str(ROOT),'.')
    return re.sub(r'/Users/[^/\s"\']+', '<USER_HOME>',text)


def export_checkpoint(data):
    """Redact protocol-2 string metadata only; copy tensor storage bytes exactly."""
    source=zipfile.ZipFile(io.BytesIO(data));changed=False;entries={}
    for name in source.namelist():
        body=source.read(name)
        if name.endswith('.pkl') and b'/Users/' in body:
            if body[:2]!=b'\x80\x02':raise ValueError('Unsupported checkpoint pickle protocol')
            operations=list(pickletools.genops(body));parts=[]
            for i,(op,arg,pos) in enumerate(operations):
                end=operations[i+1][2] if i+1<len(operations) else len(body)
                if isinstance(arg,str) and clean_text(arg)!=arg:
                    if op.name!='BINUNICODE':raise ValueError('Unsupported string opcode')
                    b=clean_text(arg).encode('utf-8');parts.append(b'X'+struct.pack('<I',len(b))+b);changed=True
                else:parts.append(body[pos:end])
            body=b''.join(parts);list(pickletools.genops(body))
            assert b'/Users/' not in body
        entries[name]=body
    if not changed:return data
    target=io.BytesIO()
    with zipfile.ZipFile(target,'w',zipfile.ZIP_STORED) as z:
        for name,body in entries.items():z.writestr(name,body)
    with zipfile.ZipFile(io.BytesIO(target.getvalue())) as z:
        for name in source.namelist():
            if not name.endswith('.pkl'):assert z.read(name)==source.read(name)
    return target.getvalue()


def main():
    files=set()
    # Fixed processed inputs only. Original raw BMP files remain outside the export.
    for folder in ['data/xray_v2','data/xray_combined_20261003']:
        base=ROOT/folder
        files.update(p for p in base.rglob('*') if p.is_file() and
                     (p.suffix in {'.png','.txt','.csv','.json'} or p.name=='data.yaml')
                     and 'runtime' not in p.name)
    seeds=['xray_submission_replay.py','xray_train_resumable.py','xray_train_latest_yolo.py',
           'xray_dfine.py','xray_secondary_review_live.py','xray_inspection_gate.py',
           'xray_check_dataset.py','xray_prepare_v2.py','xray_prepare_roadmap.py',
           'xray_prepare_combined.py','xray_pseudo_label.py','xray_package_submission.py']
    seeds += [p.name for p in (ROOT/'scripts').glob('test_xray*.py')]
    queue=[ROOT/'scripts'/x for x in seeds]
    while queue:
        p=queue.pop()
        if p in files or not p.exists():continue
        files.add(p)
        for node in ast.walk(ast.parse(p.read_text())):
            names=([node.module] if isinstance(node,ast.ImportFrom) else
                   [n.name for n in node.names] if isinstance(node,ast.Import) else [])
            for name in names:
                if name:
                    local=ROOT/'scripts'/(name.split('.')[0]+'.py')
                    if local.exists():queue.append(local)
    for name in ['requirements.txt','requirements.lock.txt','.python-version','setup.sh']:
        files.add(ROOT/name)
    vendor=ROOT/'reports/common_epoch_20261004/vendor/D-FINE'
    files.update(p for p in vendor.rglob('*') if p.is_file() and '.git' not in p.parts and
                 (p.suffix in {'.py','.yaml','.yml'} or p.name in {'LICENSE','README.md','requirements.txt'}))
    common=json.loads((ROOT/'reports/common_epoch_20261004/evaluation/frozen_selection.json').read_text())
    latest=json.loads((BASE/'proposal_frozen.json').read_text())
    specs={**common['runs'],**latest['specs']}
    policy=[]
    for run,s in specs.items():
        d=ROOT/'runs'/run
        for name in ['selection.json','execution.json']:files.add(d/name)
        checkpoint=ROOT/s['checkpoint'];assert sha(checkpoint.read_bytes())==s.get('checkpoint_sha256',s.get('sha256'))
        files.add(checkpoint)
        # Preserve the full epoch-wise selection curve but export selected weights only.
        files.update(p for p in (d/'validation').glob('*') if p.suffix in {'.csv','.json'})
        if run in common['runs']:
            pred=ROOT/'reports'/f'preds_{run}_test.csv'
            ev=ROOT/'reports/common_epoch_20261004/evaluation'/f'{run}_evaluation.json'
            record=json.loads(ev.read_text());m=record['metrics']['iou50']['test']
            threshold=s['thresholds']['iou50'];files.add(ev)
        else:
            pred=BASE/'inference'/run/'test/preds_repeat0.csv'
            row=next(x for x in json.loads((BASE/'latest_test_descriptive.json').read_text())['models'] if x['run']==run)
            m=row['metrics'];threshold=row['threshold']
        files.add(pred)
        policy.append(dict(name=run,predictions=str(pred.relative_to(ROOT)),threshold=threshold,
                           expected={k:m[k] for k in ['tp','fp','fn','f1','precision','recall','ap']}))
    files.add(ROOT/'reports/common_epoch_20261004/evaluation/frozen_selection.json')
    files.add(ROOT/'reports/common_epoch_20261004/evaluation_policy.json')
    for name in ['reports/overnight_20261003/diagnostics/model_selection_comparison.json',
                 'docs/evidence/condition_audit_20261003/summary.json']:
        files.add(ROOT/name)
    for p in (ROOT/'weights').iterdir():
        if p.is_file() and p.name!='yolov8s.pt':files.add(p)
    for name in ['data_integrity.json','six_model_validation.json','latest_validation_comparison.json',
                 'latest_validation_uncertainty.json','latest_inference_validation.json',
                 'latest_test_descriptive.json','proposal_frozen.json','food_fn_priority_proposal.json',
                 'pretrained_sources.json','comparison_policy.json']:
        files.add(BASE/name)
    for folder in ['yolo8_faster_all','yolo11_rfdetr_all','shared_error_audit','latest_visual']:
        files.update(p for p in (BASE/folder).rglob('*') if p.is_file() and p.suffix in {'.csv','.json','.png'})
    payload={};checkpoint_exports={}
    redacted=[]
    for p in sorted(files):
        data=p.read_bytes();name=str(p.relative_to(ROOT))
        if p.suffix in {'.json','.csv','.yaml','.yml','.txt','.md','.py','.sh'}:
            text=data.decode('utf-8')
            if p.suffix=='.json':text=json.dumps(json.loads(text),ensure_ascii=False,indent=2)
            clean=clean_text(text)
            if clean!=text:redacted.append(name)
            data=clean.encode('utf-8')
        if p.suffix in {'.pt','.pth','.ckpt'} and 'runs' in p.relative_to(ROOT).parts:
            original_hash=sha(data);data=export_checkpoint(data)
            if sha(data)!=original_hash:checkpoint_exports[name]=dict(original_sha256=original_hash,export_sha256=sha(data))
        payload[name]=data
    # Exported policies must point to the exported checkpoint metadata hash.
    for name,data in list(payload.items()):
        if Path(name).suffix=='.json':
            text=data.decode('utf-8')
            for pair in checkpoint_exports.values():text=text.replace(pair['original_sha256'],pair['export_sha256'])
            payload[name]=text.encode('utf-8')
    payload['reproduction_policy.json']=json.dumps(dict(models=policy,matching='iou50',threshold_source='validation',
        previous_test_exposure=True,real_normal_test_images=0),ensure_ascii=False,indent=2).encode()
    payload['README.md']=README.encode()
    payload['EXPORT_NOTES.json']=json.dumps(dict(redacted_text_files=redacted,
        note='Exported text replaces host paths. Original records remain local; embedded original source hashes describe originals, not redacted exports.',
        checkpoint_exports=checkpoint_exports,tensor_storage_bytes_unchanged=True,
        checkpoint_metadata_redacted=True,selected_checkpoints_only=True,full_resume_archives_included=False),indent=2).encode()
    payload['PACKAGE_SHA256.json']=json.dumps(dict(files={n:sha(v) for n,v in payload.items()}),indent=2).encode()
    destination=OUT/'Xray_재현자료_FN위험보완.zip'
    if destination.exists():raise FileExistsError(destination)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,data in payload.items():z.writestr(name,data)
    with zipfile.ZipFile(destination) as z:assert z.testzip() is None
    result=dict(path=str(destination.relative_to(ROOT)),files=len(payload),bytes=destination.stat().st_size,
                sha256=sha(destination.read_bytes()),redacted_files=len(redacted),crc_check='passed')
    (BASE/'submission_package.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


README='''# X-ray 이물 탐지 재현 자료

이 묶음은 여섯 모델의 선택 저장본, 고정 가공 데이터, 예측 결과와 실행 코드를 연결한다. 공장 출하 프로그램이 아니다.

## 먼저 확인할 결과
1. ZIP을 새 폴더에 푼다. 폴더 전체를 유지한다.
2. Python 3.12.14와 uv가 있으면 `bash setup.sh`로 환경을 설치한다. 최초 설치에는 인터넷이 필요하다.
3. `.venv/bin/python scripts/xray_submission_replay.py`를 실행한다. 모든 파일 해시를 확인하고, 고정 임계값으로 여섯 테스트 예측을 기존 채점기로 재채점한다.
4. `replay_result.json`에서 통과 여부를 확인한다. 이것은 새 학습이나 새 추론이 아니다.

## 범위와 데이터
- 가공 학습 2453장, 검증 107장, 테스트 97장. data/xray_v2는 공식 라벨 제공 500장의 분할과 마스킹본이다. data/xray_combined_20261003은 실제 학습 입력이다. 두 폴더를 합쳐 추가 학습하지 않는다.
- 검증·테스트는 고정 촬영 묶음이며 split MD5는 8e58184ae24dfd1b48da2e9dfd88fedc다.
- 실제 정상 제품은 검증·테스트에 없다. 이물 전부 제거 82장은 합성 학습 자료다.
- IoU 0.5가 주 채점 기준이다. IoU FN과 실제 이물 중심 표시 누락을 구분한다. AP는 COCO AP가 아니다.
- 모델과 임계값은 검증에서 선택했다. 테스트는 이전에 본 자료여서 새 독립 시험이라고 주장하지 않는다.
- 텍스트와 선택 저장본의 개인 장치 경로를 정규화했다. 텐서 저장 바이트는 그대로 복사했다. EXPORT_NOTES.json에 변경 파일과 저장본의 원본/내보낸 해시를 기록했다. 내보낸 정책의 저장본 해시는 새 파일에 맞췄다. PACKAGE_SHA256.json은 내보낸 파일의 해시다. 그 외 과거 기록 안의 원본 해시는 원본 자료를 뜻한다.
- 학습 중단 복구 전체 묶음은 로컬·Drive에 별도 보존했다. 여기에는 추론용 선택 가중치와 epoch별 검증 기록만 포함한다. 이 ZIP으로 이전 학습을 그대로 이어가는 명령을 사용하지 않는다.

## 실제 두 모델 검사 재실행
맥북 MPS에서 다음을 실행한다. 인터넷 없이 선택 가중치를 불러온다. 새로운 실행 이름을 사용한다.
```
.venv/bin/python scripts/xray_secondary_review_live.py --second faster --scope all --name reproduced_full_review --repeats 1
```
공식 검증 107장을 다시 추론한다. 첫 경고와 두 모델 출력을 보존한다. 결과가 같거나 두 모델 모두 무검출이어도 자동 PASS를 허용하지 않는다. scripts/xray_inspection_gate.py는 제품 ID·사진 해시·지연·오류를 다루는 상태 처리 시제품이며 PLC나 GPU 강제 복구 기능은 아니다.

## 처음부터 학습할 때
기존 runs 이름을 덮어쓰지 말고 새 이름을 사용한다. GPU 종류·패키지·수치 연산에 따라 결과가 달라질 수 있다. 아래는 기존 설정을 적용하는 명령이다. 새 환경에서 전체 재학습은 이번 묶음 점검에 포함하지 않았다.
```
.venv/bin/python scripts/xray_train_resumable.py --model yolo --name reproduced_yolo --epochs 20 --batch 8 --resolution 512 --device mps --epoch-validation
.venv/bin/python scripts/xray_train_resumable.py --model faster --name reproduced_faster --epochs 20 --batch 4 --resolution 512 --device cuda --epoch-validation
.venv/bin/python scripts/xray_train_resumable.py --model rfdetr --name reproduced_rfdetr --epochs 20 --batch 4 --resolution 512 --device mps --epoch-validation
.venv/bin/python scripts/xray_dfine.py --name reproduced_dfine --epochs 20 --batch 4 --resolution 512 --device cuda
.venv/bin/python scripts/xray_train_latest_yolo.py --architecture yolo11n --name reproduced_yolo11n --device mps
.venv/bin/python scripts/xray_train_latest_yolo.py --architecture yolo26n --name reproduced_yolo26n --device mps
```
동시에 여러 MPS 학습을 시작하지 않는다. 여기서 새로 시작한 run을 이어갈 때만 같은 명령에 --resume을 추가한다. 복구 검증이 실패하면 우회하지 않는다.

## 환경과 권리
requirements.txt와 requirements.lock.txt는 실제 맥북 환경 기록이다. CUDA 설치는 해당 장치의 PyTorch 공식 배포 지침과 reports/common_epoch_20261004의 기존 설치 기록을 참고한다. 학습에 사용된 공개 모델·D-FINE 소스의 이용 조건은 각 공식 프로젝트를 따른다. 데이터 공개 배포 권한과 실제 공장 상용 배포의 라이선스는 제출·도입 전에 별도 확인한다. 이 ZIP은 승인된 팀 검토·대회 제출용 로컬 자료다.
'''


if __name__=='__main__':main()
