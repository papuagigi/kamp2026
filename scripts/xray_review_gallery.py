"""Build a local, image-by-image review UI from the AI decisions (no label writes)."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from xray_config import ROOT

REPORT=ROOT/'reports/roadmap_20261002/pseudo'
REASONS={
 'models_or_flip_disagree_count':'모델 또는 좌우 반전 결과의 이물 개수가 다름',
 'models_or_flip_disagree_location':'모델 또는 좌우 반전 결과의 위치·크기가 다름',
 'equipment_candidate_unresolved':'장비가 표시한 후보 중 AI 박스가 대응하지 않는 곳이 있음',
 'unexplained_dark_point':'AI 박스 밖에 설명되지 않은 어두운 점이 있음',
 'no_detection_not_normal':'충분한 점수의 검출이 없음. 정상이라는 뜻은 아님',
 'calibration_insufficient':'공식 정답을 이용한 사전 검사에서 자동 채택 기준을 충족하지 못함',
 'unusual_box_size_or_boundary':'박스 크기나 영상 경계가 기준 밖임',
 'duplicate_boxes':'겹치는 중복 박스가 있음'}

def png(path):
    folder=REPORT/'review/assets';folder.mkdir(parents=True,exist_ok=True)
    name=hashlib.sha256(str(path.relative_to(ROOT)).encode()).hexdigest()[:24]+'.png'
    dest=folder/name
    if not dest.exists():
        with Image.open(path) as im:im.convert('RGB').save(dest,format='PNG')
    return 'assets/'+name


def main():
    decisions=pd.read_csv(REPORT/'decisions.csv').fillna('');summary=json.loads((REPORT/'summary.json').read_text())
    out=REPORT/'review';out.mkdir(exist_ok=True);records=[]
    for r in decisions.itertuples():
        rows=[list(map(float,line.split())) for line in (ROOT/r.proposed_label).read_text().splitlines() if line.strip()]
        records.append({'stem':r.stem,'status':r.status,'reason':r.reason,'reason_text':' / '.join(REASONS.get(s,s) for s in r.reason.split(':')),
                        'boxes':rows,'original':png(ROOT/r.source),'clean':png(ROOT/r.image)})
    payload=json.dumps(records,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>AI 라벨 의심 사진 확인</title>
<style>body{margin:24px;font:16px/1.6 -apple-system,sans-serif;background:#f4f6fa;color:#172a3a}main{max-width:1200px;margin:auto}h1{font-size:25px}button,select,input{font:inherit;padding:8px;margin:3px}button{cursor:pointer}.panels{display:grid;grid-template-columns:1fr 1fr;gap:20px}.panel{background:white;padding:14px;border-radius:12px}canvas{width:100%;height:auto;image-rendering:pixelated}#info{padding:14px;background:#fff4d6;border-radius:10px}.small{font-size:14px;color:#46546a}#caption{font-weight:600}@media(max-width:750px){.panels{grid-template-columns:1fr}}</style>
<main><h1>AI 라벨 의심 사진 확인</h1><p>원본과 색 네모를 지운 사진을 나란히 봅니다. 오른쪽 파란 박스는 AI의 제안이며 확정 정답이 아닙니다. 점이 빠졌는지, 다른 곳을 가리키는지, 크기가 적절한지만 확인하세요.</p>
<p class="small">‘보류’는 AI 근거가 부족한 사진입니다. 모든 보류 사진을 지금 검사할 필요는 없습니다. 검사하지 않은 사진은 학습에서 제외됩니다. 이 화면의 선택은 학습 라벨을 자동으로 바꾸지 않습니다.</p>
<select id="filter"><option value="review">의심 사진</option><option value="held">보류 사진</option><option value="accepted">자동 채택 사진</option><option value="all">전체</option></select>
<button id="prev">← 이전</button><button id="next">다음 →</button><input id="search" placeholder="사진 이름 검색"><span id="count"></span>
<div id="info"></div><div class="panels"><div class="panel"><h2>원본 · 색 네모 포함</h2><canvas id="original"></canvas></div><div class="panel"><h2>색 네모 제거 · AI 제안 박스</h2><canvas id="clean"></canvas></div></div>
<p id="caption"></p><button data-decision="correct">제안이 맞음</button><button data-decision="needs_fix">누락·위치·크기 수정 필요</button><button data-decision="uncertain">판단 어려움</button><button id="export">검수 메모 JSON 저장</button><p class="small">선택은 현재 브라우저 메모리에만 보관됩니다. 창을 닫기 전에 JSON을 저장하세요. 저장 파일을 기준으로 별도 라벨 수정을 진행할 수 있습니다.</p></main>
<script>
const rows=PAYLOAD,notes={},statusNames={review:'의심',held:'보류',accepted:'자동 채택'};let subset=[],index=0,renderId=0;
function draw(id,src,boxes,token){const image=new Image();image.onload=()=>{if(token!==renderId)return;const c=document.getElementById(id);c.width=image.width;c.height=image.height;const ctx=c.getContext('2d');ctx.drawImage(image,0,0);ctx.strokeStyle='#087bff';ctx.fillStyle='#087bff';ctx.lineWidth=1;ctx.font='10px sans-serif';boxes.forEach((b,j)=>{const x=(b[1]-b[3]/2)*c.width,y=(b[2]-b[4]/2)*c.height;ctx.strokeRect(x,y,b[3]*c.width,b[4]*c.height);ctx.fillText(j+1,x,y-3);});};image.src=src;}
function render(){renderId++;const r=subset[index];document.getElementById('count').textContent=subset.length?`${index+1} / ${subset.length}`:'0 / 0';if(!r){document.getElementById('info').textContent='해당 조건의 사진이 없습니다.';for(const id of ['original','clean']){const c=document.getElementById(id);c.getContext('2d').clearRect(0,0,c.width,c.height);}document.getElementById('caption').textContent='';return;}
document.getElementById('info').textContent=`${r.stem} | ${statusNames[r.status]} | ${r.reason_text||'자동 검사 기준 통과. 오류 없음이 보장되지는 않습니다.'}`;
draw('original',r.original,[],renderId);draw('clean',r.clean,r.boxes,renderId);document.getElementById('caption').textContent=`AI 제안 ${r.boxes.length}개 · 검수 메모: ${notes[r.stem]||'아직 선택하지 않음'}`;}
function filter(){const f=document.getElementById('filter').value,q=document.getElementById('search').value;subset=rows.filter(r=>(f==='all'||r.status===f)&&r.stem.includes(q));index=0;render();}
document.getElementById('filter').onchange=filter;document.getElementById('search').oninput=filter;document.getElementById('prev').onclick=()=>{index=Math.max(0,index-1);render();};document.getElementById('next').onclick=()=>{index=Math.min(subset.length-1,index+1);render();};
document.querySelectorAll('[data-decision]').forEach(b=>b.onclick=()=>{const r=subset[index];if(r){notes[r.stem]=b.dataset.decision;render();}});
document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify({created:new Date().toISOString(),notes},null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='AI라벨_검수메모.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
if(!rows.some(r=>r.status==='review'))document.getElementById('filter').value='held';filter();
</script></html>'''.replace('PAYLOAD',payload)
    (out/'AI라벨_검수.html').write_text(html)
    (out/'gallery_validation.json').write_text(json.dumps({'images':len(records),'statuses':summary['statuses'],'source':'actual AI proposals','self_contained':False,'local_assets_directory':'assets','external_network_required':False,'label_files_modified':False},ensure_ascii=False,indent=2))
    print('Saved',out.relative_to(ROOT),'images',len(records))

if __name__=='__main__':main()
