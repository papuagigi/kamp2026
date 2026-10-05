"""Add the approved full-frame policy and reviewed documents to a verified export.

Preserves the previous ZIP and original weights. The base export already removes
host-path metadata without changing checkpoint tensor storage.
"""
import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path
from xray_config import ROOT
from xray_package_submission import clean_text

BASE=ROOT/'outputs/submission_review_20261004/Xray_재현자료_FN위험보완.zip'
OUT=ROOT/'outputs/submission_review_20261004/Xray_재현자료_전체사진이중검사.zip'
REPORT=ROOT/'reports/fullframe_policy_20261005'


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    if OUT.exists():raise FileExistsError(OUT)
    expected=json.loads((ROOT/'reports/design_review_20261004/submission_package.json').read_text())
    if sha(BASE.read_bytes()) != expected['sha256']:raise ValueError('Previous ZIP hash mismatch')
    with zipfile.ZipFile(BASE) as z:
        manifest=json.loads(z.read('PACKAGE_SHA256.json'))['files']
        payload={}
        for name,digest in manifest.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError(name)
            b=z.read(name)
            if sha(b)!=digest:raise ValueError(name)
            payload[name]=b
    files=set()
    for folder in ['docs/evidence','docs/figures']:
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.md','.json','.csv','.png','.svg'})
    for name in ['19_데이터전처리_정리본.md','20_모델선정과_평가_정리본.md',
                 '21_진행계획과_작업이력.md','22_대회규칙과_제출준비.md','README.md']:
        files.add(ROOT/'docs'/name)
    # Include selected current report/deck and their user-facing guide, not private reference PDFs.
    submission=ROOT/'docs/submission_review_20261004'
    for pattern in ['*전체사진이중검사.*','README.md']:
        files.update(submission.glob(pattern))
    for name in ['xray_fullframe_policy.py','xray_package_fullframe.py','xray_approved_pair_evidence.py',
                 'xray_secondary_review_audit.py']:
        files.add(ROOT/'scripts'/name)
    live=ROOT/'reports/design_review_20261004/yolo8_rfdetr_all_20261005'
    files.update(p for p in live.iterdir() if p.suffix in {'.csv','.json'})
    # Exploratory outputs remain available without presenting them as deployed role experts.
    for p in (ROOT/'scripts').glob('*review_role*.py'):files.add(p)
    files.add(ROOT/'reports/review_roles_20261005/summary.json')
    files.update((ROOT/'scripts').glob('test_xray*.py'))
    files.add(ROOT/'run.sh')
    files.update(ROOT/name for name in re.findall(r'scripts/[A-Za-z0-9_]+\.py', (ROOT/'run.sh').read_text()))
    queue=[p for p in files if p.suffix=='.py']
    scanned=set()
    while queue:
        p=queue.pop()
        if p in scanned:continue
        scanned.add(p)
        for node in ast.walk(ast.parse(p.read_text())):
            names=([node.module] if isinstance(node,ast.ImportFrom) else
                   [n.name for n in node.names] if isinstance(node,ast.Import) else [])
            for name in names:
                if name:
                    dep=ROOT/'scripts'/(name.split('.')[0]+'.py')
                    if dep.exists() and dep not in files:
                        files.add(dep);queue.append(dep)
    notes=json.loads(payload['EXPORT_NOTES.json']);exports=notes['checkpoint_exports']
    added=[]
    for p in sorted(files):
        name=str(p.relative_to(ROOT));b=p.read_bytes()
        if p.suffix in {'.md','.json','.csv','.py','.sh'}:
            text=clean_text(b.decode('utf-8'))
            if p.suffix=='.json':
                for pair in exports.values():text=text.replace(pair['original_sha256'],pair['export_sha256'])
            b=text.encode('utf-8')
        payload[name]=b;added.append(name)
    # Update hashes embedded in exported audit/live records after path normalization.
    livekey='reports/design_review_20261004/yolo8_rfdetr_all_20261005'
    execution=json.loads(payload[livekey+'/execution.json'])
    execution['policy_sha256']=sha(payload[livekey+'/policy.json'])
    payload[livekey+'/execution.json']=json.dumps(execution,ensure_ascii=False,indent=2).encode()
    summary=json.loads(payload[livekey+'/summary.json']);summary['execution_sha256']=sha(payload[livekey+'/execution.json'])
    payload[livekey+'/summary.json']=json.dumps(summary,ensure_ascii=False,indent=2).encode()
    policykey='docs/evidence/fullframe_policy_20261005/policy.json'
    auditkey='docs/evidence/fullframe_policy_20261005/audit.json'
    audit=json.loads(payload[auditkey]);audit['policy_sha256']=sha(payload[policykey])
    payload[auditkey]=json.dumps(audit,ensure_ascii=False,indent=2).encode()
    readme=payload['README.md'].decode().replace('--second faster --scope all','--second rfdetr --scope all')
    readme += '''\n## 2026-10-05 확정한 전체 사진 검사\n- 기본 조합: YOLOv8n과 RF-DETR-S가 모두 전체 사진 검사. 확대·분할은 비교 실험으로만 보존한다.\n- 보고서·발표자료 최신 파일은 docs/submission_review_20261004의 전체사진이중검사 파일이다.\n- 저장 예측으로 정책을 재검증하려면 `.venv/bin/python scripts/xray_fullframe_policy.py`를 실행한다. 입력 해시·저장본·임계값을 확인하고 검증 107장·테스트 97장의 상태를 재생한다. 새 학습이나 추론은 아니다.\n- 원래 정책을 다시 생성하는 --freeze는 사용하지 않는다. 내보낸 가중치의 메타데이터 해시에 맞춘 정책이 이미 들어 있다.\n- 상태 재생은 docs/evidence/fullframe_policy_20261005/audit.json과 image_states.csv를 다시 쓴다. 파일 해시 검사 xray_submission_replay.py를 먼저 실행한다.\n- run.sh는 과거 비교용 진입점이다. 현재 전체 사진 정책에는 위의 xray_secondary_review_live.py 명령을 사용한다.
- 실제 정상 제품·새 독립 촬영 자료·설비 연동은 미검증이다. 두 모델의 무후보만으로 AI가 정상 출하를 승인하지 않는다.\n'''
    payload['README.md']=readme.encode()
    notes['fullframe_update']=dict(source_zip_sha256=expected['sha256'],updated_files=added,
        note='Latest policies use exported checkpoint hashes. Historical source hashes in evidence describe original records.',
        current_policy=policykey,role_weights_included=False)
    payload['EXPORT_NOTES.json']=json.dumps(notes,ensure_ascii=False,indent=2).encode()
    payload['PACKAGE_SHA256.json']=json.dumps(dict(files={n:sha(b) for n,b in payload.items()}),indent=2).encode()
    with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,b in payload.items():z.writestr(name,b)
    with zipfile.ZipFile(OUT) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC check failed')
    result=dict(path=str(OUT.relative_to(ROOT)),sha256=sha(OUT.read_bytes()),bytes=OUT.stat().st_size,
        files=len(payload),hashed_files=len(payload)-1,crc_check='passed',base_zip_preserved=True)
    REPORT.mkdir(exist_ok=True);(REPORT/'package.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
