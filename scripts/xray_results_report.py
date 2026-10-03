"""Build a blind-review manuscript/PDF from completed common-scoring JSON only."""
import argparse
import csv
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape
from xray_config import ROOT

REPORT=ROOT/'reports/roadmap_20261002'
OUT=ROOT/'outputs/submission_draft'
NAMES={'yolo':'YOLOv8n','faster':'Faster R-CNN ResNet50-FPN v2','rfdetr':'RF-DETR-S'}
VARIANTS={'official':'기본 학습 데이터셋만','geometry':'사진 자르기·회전','remove_all_refined':'이물 전부 제거 추가',
          'remove_partial_refined':'이물 일부 제거 추가','geometry_removal_refined':'자르기·회전·이물 제거',
          'offbar':'막대 밖 후보 위치에 이물 무늬 추가','pseudo':'AI 의사 라벨 추가','geometry_pseudo':'자르기·회전 + AI 의사 라벨'}


def number(x):return '해당 없음' if x is None else f'{x:.4f}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--pdf-libs',help='Optional bundled Python site-packages for document libraries')
    a=p.parse_args()
    if a.pdf_libs:sys.path.append(a.pdf_libs)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak,KeepTogether
    final=json.loads((REPORT/'final_comparison.json').read_text())
    ablation=json.loads((REPORT/'ablation_results.json').read_text())
    ds=json.loads((REPORT/'dataset_selection.json').read_text())
    pseudo=json.loads((REPORT/'pseudo/summary.json').read_text())
    refined=json.loads((ROOT/'data/xray_roadmap_20261002/refined_removal_summary.json').read_text())
    executions={r['run']:json.loads((ROOT/'runs'/r['run']/'execution.json').read_text()) for r in final}
    inference=json.loads((REPORT/'inference_benchmark.json').read_text())
    # Rank with validation evidence only, then deployment burden on the same inference device.
    selected=max(final,key=lambda r:(r['validation']['f1'],r['validation']['recall'],
                 -r['validation']['fp_per_image'],-inference[r['run']]['mean_seconds']))
    model_name=NAMES[executions[selected['run']]['model']]
    report_selection={'recommended_run':selected['run'],'model':model_name,'selection_uses_test':False,
        'rule':'validation F1, recall, fewer FP, then same-device inference mean latency; human final approval',
        'dataset_variant':ds['selected_variant'],'operational_status':'research prototype; not factory validated',
        'threshold':selected['validation']['thr']}
    (REPORT/'model_selection.json').write_text(json.dumps(report_selection,ensure_ascii=False,indent=2))
    OUT.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont('Body','/System/Library/Fonts/Supplemental/AppleMyungjo.ttf'))
    pdfmetrics.registerFont(TTFont('Sans','/System/Library/Fonts/AppleSDGothicNeo.ttc'))
    styles={'body':ParagraphStyle('body',fontName='Body',fontSize=14,leading=22.4,spaceAfter=9,wordWrap='CJK'),
            'title':ParagraphStyle('title',fontName='Sans',fontSize=23,leading=30,spaceAfter=18,wordWrap='CJK'),
            'head':ParagraphStyle('head',fontName='Sans',fontSize=17,leading=25,spaceAfter=14,wordWrap='CJK'),
            'small':ParagraphStyle('small',fontName='Sans',fontSize=10,leading=15,spaceAfter=7,wordWrap='CJK'),
            'cell':ParagraphStyle('cell',fontName='Sans',fontSize=10,leading=14,wordWrap='CJK')}
    story=[];markdown=[]
    def text(s,kind='body'):
        story.append(Paragraph(escape(s).replace('\n','<br/>'),styles[kind]));markdown.append(s+'\n')
    def head(s):text(s,'head')
    def table(rows,widths=None):
        content=[[Paragraph(escape(str(c)),styles['cell']) for c in row] for row in rows]
        t=Table(content,colWidths=widths or [475/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eff5')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#899baa')),
            ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d7dfe5'))]))
        story.extend([t,Spacer(1,12)])
        markdown.append('| '+' | '.join(map(str,rows[0]))+' |\n')
        markdown.append('| '+' | '.join(['---']*len(rows[0]))+' |\n')
        markdown.extend(['| '+' | '.join(map(str,row))+' |\n' for row in rows[1:]])
    def figure(name,caption):
        path=ROOT/'docs/figures'/name
        from PIL import Image as PILImage
        with PILImage.open(path) as im:w,h=im.size
        story.append(Image(str(path),width=475,height=475*h/w));text(caption,'small')
        markdown.append(f'![{caption}](../../docs/figures/{name})\n')
    def page():story.append(PageBreak())

    text('제6회 K-인공지능 제조데이터 분석 경진대회\n결과보고서 검토본','title')
    text('프로젝트명: X-ray 이물 탐지와 재검사 판단을 위한 구조별 모델 비교')
    text('팀명: ____________________')
    text(f'내용 요약: 라벨 제공 데이터셋 500장을 촬영 묶음으로 분리하고 색 네모를 제거했다. CNN 단일 단계·CNN 두 단계·Transformer 탐지기를 비교했다. 검증 자료에서 선택한 시제품 후보는 {model_name}이며, 최종 테스트 결과와 조건별 한계를 함께 보고한다.')
    text('추가 사진의 AI 라벨은 공식 정답과 분리하고, 자르기·회전·이물 제거·이물 추가를 같은 학습량으로 비교했다. 효과가 확인되지 않은 기법을 성능 개선으로 설명하지 않는다.')
    text('2026년 ____월 ____일\n팀장: ____________________ (서명)\n팀원: ____________________ (서명)\n팀원: ____________________ (서명)')
    text('검토본: 팀명·성명·서명과 설문 완료 화면은 팀이 제출 전 기입·첨부한다. 휴먼명조 글꼴이 없어 본 검토본에는 애플명조 14pt·줄간격 160%를 사용했다. 최종 양식의 글꼴을 확인한다.','small')
    page();head('제1장. 데이터 이해 및 진단')
    text('제공 영상은 세 X-ray 검사기가 저장한 완제품 검사 사진이다. 정답 사진의 작은 검은 점은 시험편 막대에 부착된 금속구이며, AI의 학습 대상은 점의 위치다. 장비가 영상 위에 그린 색 테두리는 별도의 검사기 판단이므로 학습 입력에서 제거했다.')
    table([['항목','수량·의미'],['원본 BMP','2,809개 파일 / 파일 이름과 내용으로 중복을 제거한 사진 2,532장'],['공식 정답','사진 500장 / 이물 박스 1,147개'],['고정 분할','학습 296장 · 검증 107장 · 테스트 97장'],['라벨링 후보 데이터셋','1,900장. 위치 이상 105장과 평가 묶음의 27장 제외']], [120,355])
    text('사진 단위로 무작위 분리하면 거의 같은 연속 사진을 학습과 평가에 나누어 넣을 수 있다. 같은 호기에서 60초 이내 간격으로 이어지는 사진을 하나의 촬영 묶음으로 만들고, 묶음 전체가 같은 분할에 속하게 했다.')
    text('학습·검증·테스트의 목표 비율은 60:20:20이다. 검사기·시험편 구성을 고려해 촬영 묶음을 통째로 배정하여 실제 장수는 296·107·97장이 됐다. 최적 비율로 입증한 것은 아니며 학습과 두 평가 용도의 자료를 확보하려는 선택이다. 라벨 누락 의심 사진이 포함된 6장짜리 묶음 하나는 평가 오염을 피하려고 학습에 고정했다. 시드 0에서 동일 분할을 재현했고 세트 간 묶음·내용 중복은 없었다.')
    text('공식 TXT는 이물마다 종류 번호·박스 중심의 가로와 세로 위치·너비·높이를 기록한다. 위치와 크기는 사진 전체 크기에 대한 비율이다. 종류 0은 defect를 의미하며, 정상이라는 뜻이 아니다.')
    figure('19_색표시와_정답의_차이.png','그림 1. 실제 검은 점, 장비의 빨간 표시, TXT 좌표를 그린 설명용 파란 박스. 학습 사진에는 파란 박스를 넣지 않는다.')
    page();head('제1장 계속. 전처리 방법과 품질 검사')
    text('원본 BMP의 팔레트 244~255번에 해당하는 색 네모 픽셀만 선택하고, 나비에-스토크스 인페인팅으로 주변 밝기·경계가 이어지도록 메웠다. 주변까지 삭제 범위를 넓히지 않아 작은 이물 주변의 회색 정보를 불필요하게 바꾸지 않았다. 가려진 원래 픽셀을 정확히 복원한다고 보장하지 않는다.')
    table([['방법','추가 사진','처리 내용'],['자르기','295','이물 박스가 모두 남는 80~95% 범위 선택 후 확대'],['전체 회전','590','−7.5도·+7.5도, 제품·막대·점을 함께 회전'],['전부 제거',str(refined['counts'].get('remove_all_refined',0)),'국소 어두운 성분 제거 후 잔류 검사, 빈 TXT'],['일부 제거',str(refined['counts'].get('remove_partial_refined',0)),'선택한 점만 지우고 해당 라벨만 삭제'],['이물 추가','295','학습 영상의 어두운 성분을 다른 제품 위치에 추가']], [100,65,310])
    text('모든 증강은 기본 학습 데이터셋의 사진에서만 생성했다. 누락 의심 사진 1장은 증강에서 제외했다. 잘린 사진의 좌표 역변환, 회전한 모서리 좌표, 제거 후 남은 라벨과 평가 파일의 동일성을 검사했다. 이물 제거본은 합성 음성 자료이며 실제 정상 생산품을 대신하지 않는다.')
    text(f"AI 의사 라벨: 학습 내부에서 234장으로 모델을 학습하고 다른 14개 촬영 묶음의 61장으로 기준을 조정했다. 두 구조의 예측·사진 전체 좌우 반전 후 좌표 일치·장비 후보·잔여 어두운 점을 함께 확인했다. 후보 {pseudo['candidates']}장 중 처리 상태는 {pseudo['statuses']}이며, 자동 채택 박스는 {pseudo['accepted_boxes']}개다.")
    text('자동 기준의 정확도는 기준 조정에 쓴 자료에서 관찰한 값이다. 알려지지 않은 새 사진에서도 오류가 없다는 의미가 아니다. 의심·보류 사진은 자동 학습에 넣지 않았다.')
    page();head('전처리 실제 사진')
    figure('19_실행한_전처리_사진예시.png','그림 2. 같은 학습 사진에 적용한 처리. 색 박스는 저장된 라벨을 보여 주는 설명선이다. 생성한 자료를 모두 최종 학습에 채택한 것은 아니다.')
    text('단순히 그림을 늘리면 성능이 오른다는 가설을 검증하기 위해, 전처리 종류별 데이터와 공식 라벨 기준 데이터를 분리했다. 특히 이물 추가·제거는 영상의 물리적 특성을 완벽히 재현한다고 전제하지 않았다.')
    page();head('제2장. AI 예측모델 개발 및 성능평가')
    table([['모델','구조와 비교 목적'],['YOLOv8n','CNN의 단일 단계 탐지. 빠르고 작은 기준 모델'],['Faster R-CNN ResNet50-FPN v2','CNN의 두 단계 탐지. 의심 영역을 제안한 뒤 각 영역의 특징으로 다시 판단'],['RF-DETR-S','Vision Transformer와 DETR 탐지 구조. 여러 위치의 특징 관계와 사전학습 표현 활용']], [155,320])
    text('공식 사전학습 가중치를 가져와 이물 탐지에 파인튜닝했다. 정지 영상의 공간적 위치를 찾는 문제여서 시계열용 RNN·LSTM을 비교 모델에 추가하지 않았다. 최신성이나 구조 이름만으로 성능 우위를 가정하지 않았다.')
    text('모든 모델은 같은 학습 부모·검증·테스트 분할을 사용했다. 입력 기준 해상도는 512이며 모델별 크기 맞춤·정규화를 적용했다. Faster R-CNN은 작은 이물에 맞춰 FPN 단계의 앵커 크기를 8/16/32/64/128픽셀로 설정하고 후보 수를 제한했다. RF-DETR은 이물 1종 출력 구조를 명시했다.')
    text('채점은 하나의 공통 스크립트를 사용했다. 주지표는 IoU 0.5 이상의 일대일 대응에서 계산한 F1이다. 점수가 높은 예측부터 아직 대응하지 않은 정답 중 IoU가 가장 큰 것을 고른다. 아래 AP는 이 규칙의 all-points AP이며 표준 COCO AP가 아니다. 이전의 8픽셀 또는 IoU 0.3 기준은 보조 비교로만 보존했다.')
    text('이물 신뢰도 점수는 해당 예측 구역이 이물이라는 모델의 분류 출력이며, 정답과 대조해 측정한 정확도가 아니다. 검증 데이터셋에서 F1이 가장 높은 탐지 임계값을 선택하고, 동점에서는 가장 높은 임계값을 고른 뒤 테스트에 고정 적용했다. 검증 최고 F1은 개발 단계의 점수이며 테스트 성능과 구분한다.')
    text('라벨 제공 데이터셋 500장에는 이물 라벨 1,147개가 있다. 기본 학습 296장에 679개, 검증 107장에 243개, 테스트 97장에 225개다. 500장 모두 이물 라벨이 1개 이상 있어 실제 정상 제품의 오경보율과 제품 단위 분류 F1은 검증하지 못했다.')
    figure('19_색네모_정답박스_예측박스.png','같은 실제 사진에서 원본의 색 네모, 공식 TXT 좌표, 모델 예측 좌표를 구분한 예시. 설명용 박스는 학습 입력에 없다.')
    page();head('제2장 계속. 박스 대응과 측정 결과')
    text('IoU는 두 박스가 겹친 면적을 두 박스가 차지하는 전체 면적으로 나눈 비율이다. 0.5와 0.75는 COCO 평가에서도 사용되는 위치 정확도 기준을 참고했다. 이를 대회가 지정한 기준으로 주장하지 않는다. 초기 중심 거리 8픽셀 또는 IoU 0.3은 작은 이물의 위치 오차를 허용하는 자체 기준으로, 그 수치의 최적성을 실험으로 입증한 근거가 없어 보조 지표로 내렸다.')
    figure('19_IoU와_중복예측_설명.png','설명용 도식. 같은 이물을 두 번 예측해도 정답 하나는 한 번만 맞힌 것으로 센다. 남은 중복 예측은 오탐이며, 찾지 못한 다른 정답은 미탐이다.')
    table([['모델','검증 F1','테스트 F1','테스트 P/R','테스트 AP'],*[[NAMES[executions[r['run']]['model']],number(r['validation']['f1']),number(r['test']['f1']),number(r['test']['precision'])+' / '+number(r['test']['recall']),number(r['test']['ap'])] for r in final]],[160,65,65,115,70])
    text(f'추천 모델은 {model_name}다. 검증 F1·재현율·오탐과 같은 장치의 추론 시간을 순서대로 비교했다. 테스트 성능을 보고 모델을 다시 고르지 않았다. 채택 데이터 조건은 “{VARIANTS[ds["selected_variant"]]}”다.')
    page();head('제2장 계속. 전처리 효과의 통제 비교')
    text('전처리 비교는 YOLOv8n에서 난수 시드 0, 배치 8, 740회 실제 업데이트를 고정했다. 업데이트별 학습률도 0.001에서 0.00001로 동일하게 줄였다. 데이터 수가 많아서 학습 횟수까지 늘어난 영향을 분리하기 위한 설정이다. 마지막 배치가 작은 경우 실제 사진 노출 수는 조금 달라지므로 실행 기록에 함께 남겼다.')
    table([['데이터 조건','검증 F1','재현율','오탐 수'],*[[VARIANTS[r['variant']],number(r['f1']),number(r['recall']),str(r['fp'])] for r in ablation]],[265,70,70,70])
    off=next(r for r in ablation if r['variant']=='offbar');base=ablation[0];delta=off['f1']-base['f1']
    if delta>1e-6:
        text(f'막대 밖 후보 위치에 이물 무늬를 추가한 조건은 검증 F1이 {base["f1"]:.4f}에서 {off["f1"]:.4f}로 높아졌다. 다만 한 데이터·학습량 조건의 결과이며 실제 막대 밖 혼입 이물에 대한 검증을 대신하지 않는다.')
    elif delta< -1e-6:
        text(f'막대 밖 후보 위치에 이물 무늬를 추가하면 성능이 좋아질 것으로 예상했지만, 검증 F1은 {base["f1"]:.4f}에서 {off["f1"]:.4f}로 낮아졌다. 이 실험 조건에서는 가설이 지지되지 않아 해당 합성 자료를 최종 학습에서 제외했다.')
    else:text('막대 밖 후보 위치의 이물 추가 조건은 검증 F1 개선을 보이지 않았다. 악화했다고 쓰지는 않으며, 개선 근거가 없고 합성의 현실성도 추가 확인이 필요해 최종 학습에서 제외했다.')
    text('각 조건은 시드 0의 단일 학습 결과다. 반복 시드의 신뢰구간이나 새로운 생산품에서의 일반화 개선을 검증한 것은 아니다.')
    text('F1이 같으면 기본 학습 데이터셋만 사용하는 조건을 유지했다. 검증 점수를 기준으로 자료와 모델을 선택했으므로 검증 최고 점수는 선택 효과를 포함한다. 새로운 생산품과 실제 정상 제품을 포함한 외부 평가 및 학습 내부 촬영 묶음 교차 검증이 추가로 필요하다.')
    page();head('제3장. 영향요인 및 오류분석')
    test=selected['test'];conditions=test['recall_by_condition']
    text(f'추천 모델의 고정 임계값은 {selected["validation"]["thr"]:.6f}다. 이를 테스트에 그대로 적용했을 때 정답 {test["n_gt"]}개 중 탐지 {test["tp"]}개, 미탐 {test["fn"]}개, 오탐 {test["fp"]}개였다.')
    rows=[['조건','재현율','정답 박스 수']]
    for group in ['machine','size_bin','contrast_bin','edge_bin']:
        for key,value in conditions.get(group,{}).items():rows.append([{'machine':'호기','size_bin':'박스 크기','contrast_bin':'대비','edge_bin':'가장자리 거리'}[group]+' · '+key,number(value[0]),str(value[1])])
    table(rows,[280,95,100])
    text('호기·크기·대비·제품 가장자리와의 거리는 관찰한 조건이며 인과 효과를 식별한 실험은 아니다. 표본이 적은 조건에서 점수가 높아도 해당 조건을 충분히 검증했다고 단정할 수 없다. 데이터에는 대표적인 실제 정상 제품 표본이 없어 공장 오경보율은 산출할 수 없다.')
    text('테스트 자료는 과거 프로젝트에서 이미 확인한 이력이 있다. 이번 실험에서 선택에는 사용하지 않았지만 완전히 새로운 외부 블라인드 평가라고 표현하지 않는다.')
    page();head('제4장. 현장 활용방안')
    text('제품 단위 출력은 탐지 박스·점수·이상 의심 점수·판정 상태를 함께 제공한다. 가장 높은 이물 신뢰도 점수를 제품의 이상 의심 점수로 요약할 수 있지만, 실제 불량 확률로 보정됐다고 부르지 않는다. 확률 보정에는 대표적인 실제 정상·이상 제품과 발생 비율이 추가로 필요하다.')
    table([['입력·출력 조건','제안 조치'],['임계값 이상 이물 후보','위치가 표시된 영상을 저장하고 제품 분리·재검사'],['낮은 점수, 모델 간 불일치, 잔여 의심점','확정 정상 대신 재검사·보류'],['영상 읽기 실패, 잘림, 포화, 알려지지 않은 입력','오류 기록 후 보류. 무검출을 자동 정상으로 바꾸지 않음'],['자동 통과 운영','실제 정상 제품 오경보율과 요구 미탐률을 확보한 뒤 별도 승인']], [210,265])
    timing=inference[selected['run']]
    text(f'현재 측정 장치는 {timing["device"]}이며 이미지 읽기·변환·추론·후처리 평균은 {timing["mean_seconds"]*1000:.1f}ms, 95백분위는 {timing["p95_seconds"]*1000:.1f}ms다. 학습이 끝난 뒤 같은 검증 사진 107장을 모델별로 순차 측정했다. 모델별 예열 3장은 시간 집계에서 제외했다. 이 측정에는 원본 색 네모 제거와 공장 통신 시간이 포함되지 않아 생산 라인의 전체 지연으로 해석하지 않는다.')
    text('실제 적용 전 생산 속도·허용 지연·미탐 비용·재검사 인력과 코드 및 가중치의 사용 조건을 확인한다. 현 결과는 시험편 중심의 연구용 시제품 검증이며 공장 배치 완료를 뜻하지 않는다.')
    page();head('제5장. 창의성 및 차별성')
    text('색 네모를 모델 입력의 지름길로 사용하지 않도록 표시 픽셀만 제거했다. 사진과 별도의 공식 TXT를 보존하고, 촬영 묶음으로 학습·평가 누출을 줄였다. 두 탐지 구조와 좌우 반전 일관성, 장비 위치 후보와 잔여 어두운 점 검사를 조합해 의사 라벨의 자동 채택과 사람의 의심 사례 검수를 구분했다.')
    text('또한 전처리 자료를 많이 만드는 데 그치지 않고 같은 학습량에서 효과를 분리해 비교했다. 도움이 되지 않은 합성과 제거를 채택하지 않을 수 있도록 실행 전에 판단 기준을 정했다. 이러한 절차적 차별성과 실제 점수 개선 여부는 구분해 보고한다.')
    head('제6장. 코드 구성 및 재현성')
    table([['단계','실행 파일'],['전처리','xray_prepare_roadmap.py / xray_refine_removal.py'],['좌표·분할 검사','xray_validate_roadmap.py'],['모델 학습','xray_train_controlled.py'],['공통 예측·채점','xray_predict_controlled.py / xray_eval.py'],['의사 라벨','xray_prepare_teacher_split.py / xray_pseudo_label.py'],['실험 이력·보고서','xray_run_roadmap_queue.py / xray_results_report.py']],[130,345])
    text('Python 3.12.14와 프로젝트 잠금 파일의 패키지를 사용했다. 학습 시드와 분할 목록을 저장하고, 모델별 실제 장치·설정·체크포인트·추론 결과·채점 JSON을 기록했다. CPU와 Apple GPU의 수치 차이와 다른 장치에서의 학습 시간을 동일하다고 가정하지 않는다.')
    text('주지표 채점 버전: iou50_v1. 보존된 초기 비교: custom_ap_v2. 고정 분할 확인값: 8e58184ae24dfd1b48da2e9dfd88fedc. 실패한 연결 시험·중단 실행은 최종 성능표에서 제외했다.','small')
    text('평가 참고: scikit-learn classification_threshold 및 cross_validation 공식 문서, COCO API의 cocoeval.py. IoU 0.5/0.75 위치 기준을 참고하되, 본 AP는 all-points 계산이므로 COCO AP와 동일하지 않다.','small')
    text('참고 자료: 대회 공식 안내문·결과보고서 양식·X-ray 추가 안내, Ultralytics YOLOv8 공식 문서, torchvision Faster R-CNN 공식 문서, RF-DETR 공식 저장소. 선행 모델의 공개 성능을 본 데이터의 측정 결과로 대체하지 않았다.','small')
    page();head('제출 전 팀 확인')
    text('팀명·참가자 성명·서명을 표지에 기입한다. 만족도 조사 완료 화면을 이 페이지에 첨부한다. 모든 제출물의 소속·학교명·로고·홈 경로 등 참가자 식별 정보를 다시 확인한다. 검토본의 대체 글꼴을 공식 양식의 휴먼명조로 적용한다.')
    text('설문 완료 화면 첨부 영역\n\n\n\n\n\n\n\n')
    text('포털 제출·서명·설문 응답은 팀의 최종 확인 후 진행한다. 이 파일은 자동으로 제출되지 않았다.','small')
    pdf=OUT/'결과보고서_검토본.pdf'
    def footer(canvas,doc):
        canvas.setFont('Sans',9);canvas.setFillColor(colors.HexColor('#526476'))
        canvas.drawString(60,32,'X-ray 이물 탐지 | 결과보고서 검토본');canvas.drawRightString(A4[0]-60,32,str(doc.page))
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=60,leftMargin=60,topMargin=50,bottomMargin=50,
                      title='X-ray 이물 탐지 결과보고서',author='').build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/'결과보고서_본문.md').write_text('\n'.join(markdown))
    (OUT/'문서_검증정보.json').write_text(json.dumps({'source':'completed scoring JSON','selection':report_selection,
        'font_substitution':'AppleMyungjo used because Human Myeongjo not installed; final template check needed',
        'requires_team_fields_signatures_survey':True,'pdf':pdf.name},ensure_ascii=False,indent=2))
    print(pdf.relative_to(ROOT))


if __name__=='__main__':main()
