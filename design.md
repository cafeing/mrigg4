# 전립선 bpMRI 고등급 암(GG≥4) 판정 AI의 제조사·딥러닝 재구성 강건성 — 연구 설계·코드 지시서

버전 2026-11-17 · 프로젝트 전립선암_안성아 · Claude Code(VS Code, 오프라인 GPU 서버)용
함께 쓰는 문서: `claude/mri_staging_design.md`(전처리·인코더 파이프라인이 같음)

---

## 0. 이 문서 사용법과 원칙

- VS Code에서 Claude Code에게 그대로 넘기는 작업 지시서다. §9의 각 단계는 스크립트 하나이며, 앞 단계의 **[확인]**을 출력한 뒤 다음 단계로 간다. **[중단 조건]**에 걸리면 멈추고 보고한다.
- **환자 단위 데이터(영상, 연구번호, 날짜, 판독문)는 기관 내부 밖으로 내보내지 않는다.** 출력은 집계표·그림·로그만.
- GPU 서버는 인터넷이 없다. 가중치·wheel은 로컬에서 받아 업로드한다(§10).
- **외부검증 데이터(서울)와 시간 검증 데이터(광명 2025–2026)는 주 모델을 잠그기(`LOCKED_MODEL.yaml`) 전까지 성능 계산에 쓰지 않는다.** 전처리와 QC만 허용한다.
- 모든 스크립트는 고정 seed, yaml 설정, 로그, git commit 해시를 남긴다.
- `external1`처럼 파일이 많은 폴더에서 `ls`나 재귀 탐색을 하지 않는다. 코호트 파일의 `folder` 열로 필요한 폴더만 찾는다.

---

## 1. 연구 질문과 가설

| 항목 | 내용 |
|---|---|
| 제목(가제) | Robustness of MRI foundation-model features for detecting high-grade prostate cancer across scanner vendors and vendor deep-learning reconstruction |
| 주 질문 | 치료 전 bpMRI에서 사전학습 인코더의 고정 특징으로 만든 GG≥4 판정 모델이, **제조사(Philips→Siemens)**와 **제조사 딥러닝 재구성(DLR)**이 바뀌어도 성능을 유지하는가 |
| 주 결과 | 환자 단위 GG≥4(ISUP 4–5) 대 GG≤3. 라벨 = max(MRI ±120일 생검 GG, MRI 후 0–180일 RP 병리 GG) |
| 보조 결과 | GG≥3, 생검만으로 정한 GG≥4 |
| 입력 | MRI만: 축상 T2, ADC, 고 b값 DWI |
| 핵심 분석 축 | ① 제조사가 다른 외부검증(광명 Philips → 서울 Siemens, 반대 방향 보조) ② **같은 장비 안 DLR 전환**(광명 2022–2024 일반 재구성 → 2025–2026 SmartSpeed 재구성) ③ **같은 검사 안 재구성 쌍**(고 b값 일반·DLR 버전이 모두 있는 151건) ④ 인코더 사전학습 도메인 비교(MRI 기반모델 대 CNN 대 2D 일반 기반모델) ⑤ 강건성 학습법 비교 |
| H1 | 잠근 주 모델의 서울 외부검증 AUROC가 광명 CV AUROC보다 0.05 이상 떨어지지 않는다(비열등 한계 −0.05) |
| H2 | DLR 전환 시험(축 ②)에서 강건성 학습법(R1–R3)이 기본 선형 탐침(R0)보다 성능 하락이 작다 |
| H3 | MRI로 사전학습한 인코더(Triad)가 DINOv2·MedicalNet보다 제조사·재구성 이동에 덜 민감하다(특징 이동 지표와 AUROC 하락 모두) |

**왜 이 질문인가**
- MRI로 고등급 암을 찾는 과제 자체는 PI-CAI 등으로 이미 많이 연구됐다. 여기서의 기여는 "얼마나 잘 맞히나"보다 **"장비·재구성이 바뀌면 무너지는가, 무엇이 덜 무너지는가"**다.
- 제조사 차이가 딥러닝 성능과 특징에 큰 영향을 준다는 보고가 있다(Radiology: AI 2024, doi:10.1148/ryai.230555).
- 딥러닝 재구성이 AI 성능에 미치는 영향은 체계적 문헌고찰이 남은 과제로 꼽았다(J Pers Med 2025, doi:10.3390/jpm15070284). 우리 광명 자료는 같은 장비에서 재구성만 바뀐 시점이 있어 이 질문을 직접 시험할 수 있다.

---

## 2. 코호트

### 2.1 정의
1. 전립선 MRI DICOM(StudyDescription에 Prostate)이 있고, 검사일은 SeriesInstanceUID에서 복원한 날짜를 쓴다.
2. 라벨: MRI 검사일 ±120일 생검 병리 GG, 또는 MRI 후 0–180일 RP 병리 GG가 있다. 둘 다 있으면 높은 쪽(RP가 생검 과소추정을 보정).
3. **치료 전 MRI만:** MRI 이전에 ADT(GnRH 작용제·길항제, 항안드로겐, ARPI)·RT·RP가 없다.
4. 환자당 가장 이른 적격 MRI 1건.
5. 기관: 연구번호 ≤ R000002139 = 서울(Siemens), 그 이상 = 광명(Philips).

### 2.2 규모 (2026-11-17, 맥 집계)

| | 광명 (개발) | 서울 (외부검증) | 전체 |
|---|---|---|---|
| 환자 | 308 | 133 | 441 |
| GG≥4 | 115 (37%) | 59 (44%) | 174 |
| GG≥3 | 144 | 89 | 233 |
| T2 DLR 사용 | 112 (2025–2026 전부) | 19 (Deep Resolve) | 131 |
| 고 b값 일반·DLR 쌍 | — | — | 151 |
| 검사 연도 | 2022–2026 | 2021–2026 | |

- 라벨 출처: max(생검, RP) 285명, 생검만 113명, RP만 43명.
- 생검 후 MRI(biopsy-first) 328명, 생검→MRI 간격 중앙값 16일 → 출혈 영향 하위군 분석 필요.
- 치료 후 MRI를 빼기 전 528명이었다. ADT·RT 후 MRI는 종양 신호가 바뀌어 라벨과 맞지 않으므로 뺐다.
- **광명 DLR 전환:** 2022–2024년 196명(GG≥4 75)은 일반 재구성, 2025–2026년 112명(GG≥4 40)은 전부 `_smpr_recon` T2다.

**[확인]** 서버에서 위 표를 재현한다(파일이 실제로 있는 환자 기준). 차이가 5% 넘으면 원인을 보고한다.

### 2.3 파일 (맥 `dataset/tab_benchmark/` → 서버로 복사, 반출 금지)

| 파일 | 단위 | 주요 열 |
|---|---|---|
| `gg4_cohort_study.csv` | 환자 1행 (441) | pid, site, vendor, StudyInstanceUID, folder, exam_date, date_src, pick_T2ax, T2ax_is_dlr, T2ax_alt_recon, pick_ADC, ADC_is_dlr, ADC_alt_recon, pick_HB, HB_is_dlr, HB_alt_recon, seq_complete, dlr_any, bx_gg, bx_to_mri_days, biopsy_first, rp_gg, mri_to_rp_days, gg_final, label_source, y_gg4, y_gg3, y_gg4_bxonly, age, psa |
| `gg4_cohort_series.csv` | 시리즈 1행 | pid, StudyInstanceUID, SeriesInstanceUID, folder, SeriesDescription, seq(T2ax/ADC/HB/other), dlr, n_inst, thk, PixelSpacing, ImageType |
| `gg4_cohort.py` | — | 코호트 생성 코드(재현용) |

- `pick_*`: 시퀀스별 기본 시리즈. **일반 재구성을 우선**하고, 없으면 DLR, 그다음 instance 수가 많은 것.
- `*_alt_recon`: 같은 검사에서 재구성 방식이 반대인 시리즈(쌍 분석용).
- `psa`: MRI −180~+30일 중 가장 가까운 PSA. 검사 자료가 생검 이후만 있어 135명이 결측이다.
- 시퀀스 규칙(대표 SeriesDescription)
  - 광명: `T2 TSE AX` / `T2 TSE AX_smpr_recon`, `dDWI_3b_ZOOM ADC` 또는 `dADC`, `ssDWI ZOOM b-1400` / `…_smpr_recon`, `cDWI2000`
  - 서울: `T2 AX TSE` / `DRB T2 AX TSE 1mm`, `…_ADC`, `…_CALC_BVAL`(C1500), `DL DWI …`
- DLR 플래그: SeriesDescription에 `smpr`(Philips SmartSpeed 계열로 추정), `DRB`·`DL`(Siemens Deep Resolve 계열로 추정). 서버에서 DICOM 사설 태그나 ImageType으로 재구성 방식을 확인할 수 있으면 플래그를 보정한다.

---

## 3. 선행연구와 차별점

| 연구 | 내용 | 우리와의 차이 |
|---|---|---|
| PI-CAI (Lancet Oncol 2024) | 대규모 bpMRI csPCa 검출 AI, 다기관 | 검출 성능 중심. 재구성 변화는 다루지 않음 |
| Radiology: AI 2024 (doi:10.1148/ryai.230555) | 제조사·코일이 bpMRI 공격성 분류 딥러닝 성능에 큰 영향 | 영향 존재를 보임. 기반모델·강건성 학습법 비교 없음 |
| J Pers Med 2025 체계적 고찰 (doi:10.3390/jpm15070284) | DLR은 영상 품질을 높이지만 기존 데이터로 학습한 AI가 DLR 영상에서 덜 정확할 수 있음 | "AI 성능에 미치는 영향"을 남은 과제로 제시 → 우리가 직접 시험 |
| Eur J Radiol 2023, Diagn Interv Imaging 2025 | DLR이 영상 품질·radiomics 특징을 바꿈 | 판독 품질·radiomics 수준. 딥러닝 판정 성능은 아님 |
| ProFound·ProViCNet 전이 한계 분석 (arXiv 2026) | 전립선 MRI 기반모델의 과제별 이득이 고르지 않음 | 장비·재구성 이동에 대한 강건성은 보지 않음 |

차별점 세 가지의 조합으로 주장한다.
1. **같은 장비에서 재구성만 바뀐 실제 임상 전환**(광명 2025)과 같은 검사 안 재구성 쌍을 이용한 DLR 영향 평가
2. **제조사가 다른 외부검증**(Philips 단일 장비 학습 → Siemens 두 기종)
3. **MRI 기반모델 대 CNN 대 2D 일반 기반모델**, 그리고 **강건성 학습법**을 같은 파이프라인에서 비교

---

## 4. 데이터 분할과 분석 설계

| 분석 | 학습 | 평가 | 목적 |
|---|---|---|---|
| **A. 제조사 외부검증 (주)** | 광명 전체 308 (CV로 모델 선택) | 서울 133 (1회) | H1 |
| A′. 반대 방향 (보조) | 서울 133 | 광명 308 | 학습 기관 의존성 |
| **B. 같은 장비 DLR 전환** | 광명 2022–2024 일반 재구성 196 | 광명 2025–2026 DLR 112 | H2 |
| B-대조 | 광명 2022–2024에서 무작위 112명을 떼어 낸 내부 검증(같은 크기, 50회 반복) | | 시간·표본 변동만으로 생기는 하락의 기준선 |
| **C. 같은 검사 재구성 쌍** | (B와 같은 학습 모델) | 고 b값 일반·DLR 쌍 151건에서 같은 환자 예측 비교 | 재구성만의 효과 |
| D. 서울 내부 DLR | (A의 잠근 모델) | 서울 DLR 19 대 일반 114 | 기술 통계 |

- 모델 선택(인코더, 풀링, 서열 조합, 강건성 방법, L2 강도)은 **광명 2022–2024 일반 재구성 196명의 nested CV(5-fold × 5회, 안쪽 3-fold)**에서만 한다. 그 결과로 주 모델 1개를 `LOCKED_MODEL.yaml`에 기록하고 커밋한다.
- 분석 A의 최종 모델은 잠근 설정으로 광명 308명 전체를 다시 학습해 서울에 한 번 적용한다.
- 분석 B는 잠근 설정을 광명 2022–2024로만 학습해 2025–2026에 적용한다.
- B-대조가 중요하다: 2025–2026 하락이 B-대조 하락 분포의 95% 범위를 넘을 때만 "재구성·시기 이동에 의한 하락"으로 해석한다.
- ⚠ 분석 B는 재구성 변화와 시간 변화(환자 구성, 판독·생검 관행)가 겹친다. 분석 C(같은 검사 쌍)로 재구성만의 효과를 보완한다.

---

## 5. 모델 설계

### 5.1 전처리 (`mri_staging_design.md` §6과 동일)
1. NIfTI 변환, RAS, 다중 b값 DWI는 b값별 분리.
2. ADC·고 b값을 T2 격자로 헤더 기반 재샘플, 0.5×0.5×3.0 mm.
3. 전립선 분할: TotalSegmentator `total_mr`(대안 MONAI `prostate_mri_anatomy`). 부피 계산, 15–200 mL 밖은 점검 목록.
4. 정규화: T2·고 b값은 전립선 마스크 내부 z-score(1–99 백분위 자른 뒤), ADC는 0–3000 ×10⁻⁶ mm²/s로 자른 뒤 고정 비율.
5. crop: 마스크 bounding box + 사방 10 mm(GG 판정은 전립선 내부 정보가 중심이라 병기 설계의 15–20 mm보다 좁힘), 고정 크기 예 160×160×32(Triad 조건: 각 축 16의 배수).
6. 고 b값 통일(민감도): 기관 제공 고 b값 그대로(주) 대 ADC와 b0에서 b1500 계산(민감도).

### 5.2 인코더 (모두 고정, 같은 crop 입력)

| ID | 인코더 | 사전학습 | 입력 처리 |
|---|---|---|---|
| E1 (주 후보) | Triad Swin-B (MAE) | 3D MRI 131K, 전립선 포함 | 서열별로 같은 인코더 통과, 단일 채널 |
| E2 | Triad PlainConvUNet 인코더 (MAE) | 같은 데이터 | E1과 같은 데이터로 사전학습한 CNN → 구조 효과 분리 |
| E3 | MedicalNet 3D ResNet-50 | 의료 CT·MRI 혼합 | 서열별 통과 |
| E4 | DINOv2 ViT-B/14 | 자연 영상 | 슬라이스별 특징 → 주의 기반 풀링(2.5D) |
| E5 (하한) | 3D ResNet-18 처음부터 학습 | 없음 | 부분 학습 대조군 |
| (선택) E6 | PI-CAI baseline nnU-Net 병변 확률 | PI-CAI 공개 데이터 | 환자별 최대 병변 확률을 단일 특징으로 |

- 특징: 마지막 단계 global average pooling(주), 전립선 마스크 내부 평균 풀링(보조). 서열별 768-d(인코더마다 다름)를 concat.
- 서열 조합: T2+ADC+HB(주), T2만, T2+ADC.

### 5.3 헤드와 강건성 방법

| ID | 방법 | 학습 대상 | 설명 |
|---|---|---|---|
| **R0** | 선형 탐침 | L2 로지스틱 | 고정 특징 → 로지스틱. 기준선 |
| **R1** | 재구성 모사 증강 특징 평균 | L2 로지스틱 | 학습 영상마다 증강 K=8개 뷰의 특징을 뽑아 학습에 모두 쓰고, 평가 시에는 원본만. 증강 = DLR과 제조사 차이를 흉내: 가우시안·리시안 잡음 제거(non-local means, σ 범위), unsharp masking, k-space 저역 절단(해상도 0.6–1.0배), 잡음 추가, 감마 0.8–1.2, bias field |
| **R2** | 일관성 정규화 부분 미세조정 | 인코더 마지막 단계 + 헤드 | 손실 = BCE + λ·‖z(x′) − z(x″)‖² (x′, x″ = 같은 영상의 R1 증강 두 뷰). λ ∈ {0.1, 1}, lr 1e-5–1e-4, 30 epoch, early stopping |
| **R3** | 도메인 적대 학습 | 인코더 마지막 단계 + 헤드 | 개발 데이터 안의 도메인(광명 일반 대 DLR)을 판별하는 보조 헤드에 gradient reversal. **외부(서울) 영상은 라벨 없이도 학습에 넣지 않는다** |

- 사전 지정 주 모델 후보: **E1 + R0**(가장 단순하고 재현성이 높음). R1–R3는 H2 검정용 비교. 최종 잠금은 nested CV 결과로 정하되, R2·R3가 R0보다 CV AUROC가 0.02 이상 높지 않으면 R0를 잠근다.
- R2·R3는 시간이 오래 걸리므로 E1과 E3에만 적용한다.

### 5.4 비교군
| 비교군 | 입력 | 용도 |
|---|---|---|
| 임상 | 나이, log PSA, PSA density(자동 분할 부피) | 기준 성능. PSA 결측은 fold 안 중앙값 대체 + 결측 표시 |
| 판독의 | 판독문에서 추출한 PI-RADS 최고값(가능한 환자만) | PI-RADS ≥4와 5에서 민감도·특이도 비교. 추출 규칙과 추출률을 보고 |
| 임상 + AI | 임상 + P(GG≥4) | 증분 가치 ΔAUC |

---

## 6. 평가 지표와 통계

### 6.1 판별·보정
- AUROC(bootstrap 2,000회 95% CI), 정해진 민감도 90%에서의 특이도(개발 CV에서 문턱 고정 후 외부에 그대로 적용)
- 보정: calibration intercept·slope, 보정 곡선. 외부·시간 검증에서는 재보정(intercept만, intercept+slope) 전후 모두
- DCA: 생검 GG≥4 판단 문턱 0.1–0.5

### 6.2 이동(shift) 지표
- **성능 하락:** ΔAUROC = 평가 집합 − 광명 개발 CV(OOF). 비열등 한계 −0.05(H1)
- **특징 이동:** 개발 대 평가 집합 특징의 (a) MMD(RBF, 중앙값 휴리스틱), (b) 도메인 판별 AUROC(특징으로 기관 또는 재구성을 맞히는 로지스틱의 CV AUROC, 0.5에 가까울수록 이동이 작음). 인코더별로 비교(H3)
- **예측 이동(분석 C):** 같은 검사의 일반·DLR 고 b값으로 만든 두 예측(다른 서열은 동일)의 차이: 평균 |Δp|, Bland–Altman, 판정 일치율(문턱 기준), 두 예측 각각의 AUROC와 paired DeLong

### 6.3 통계 검정
- 모델 간: paired DeLong(같은 평가 집합)
- H2: 분석 B의 하락(R1–R3 대 R0) 차이를 paired bootstrap으로
- 분석 B 해석: B-대조 하락 분포의 2.5–97.5 백분위와 비교
- 다중 비교: 주 가설 H1만 확증적, 나머지는 탐색적으로 표기

### 6.4 검정력 (참고)
- 서울 외부검증: 양성 59 / 음성 74 → AUROC 0.75에서 95% CI 반폭 약 ±0.08
- 광명 2025–2026: 양성 40 / 음성 72 → 반폭 약 ±0.09
- 따라서 비열등 판단과 방법 간 차이는 효과 크기와 CI 중심으로 보고한다.

### 6.5 하위군·민감도
- 생검→MRI 간격(<21일 대 ≥21일, MRI 선행 대 생검 선행), 전립선 부피 3분위, 라벨 출처(RP 포함 대 생검만), 장비 모델(서울 Vida 대 Skyra, DICOM 태그로 확인), GG≥3 라벨, 생검만 라벨(`y_gg4_bxonly`), 고 b값 계산값 통일, crop 여유 10 대 20 mm, 분할 실패 제외 대 포함

### 6.6 해석 가능성
- Grad-CAM 또는 attention map: 맞은·틀린 사례 각 5건(연구번호·날짜 없이)
- 분석 C 쌍에서 같은 환자의 일반·DLR 영상 attention 차이 예시 3건

---

## 7. 사전 지정 사항 (결과 보기 전에 `PREREG.md`로 커밋)
- 주 결과 y_gg4, 주 집합(광명 308 개발, 서울 133 외부), 주 모델 후보(E1+R0), 잠금 규칙(§5.3), 주 지표(외부 AUROC와 ΔAUROC, 비열등 −0.05), 분석 B·C 정의, 하위군 목록(§6.5)

---

## 8. 성공 기준과 위험 요인
- **성공 기준:** 서울 외부 AUROC ≥0.70이고 광명 CV 대비 하락이 0.05 이내. 분석 B의 하락이 B-대조 범위 안이거나, 강건성 방법이 하락을 줄임.
- **예상 성능:** 영상 단독 AUROC 0.70–0.80(선행 bpMRI 공격성 분류 연구 범위 참고, 우리 자료로는 미확인).

| 위험 | 대응 |
|---|---|
| 서울 외부검증 표본 작음(양성 59) | CI 중심 보고, 반대 방향 A′와 시간 검증 B로 보완 |
| 분석 B에서 재구성과 시기 효과가 겹침 | B-대조와 분석 C로 분리 시도, 한계로 명시 |
| DLR 플래그가 SeriesDescription 추정 | DICOM 사설 태그·ImageType 확인 후 보정, 불확실 건 목록화 |
| PSA 결측 135명 | 임상 비교군에서 결측 표시 변수 + 민감도(완전 사례) |
| 자동 분할 실패 | 실패율 보고, 중심 crop 대체를 민감도 분석으로 |
| 판독문 PI-RADS 추출률 낮음 | 추출 가능한 환자로 제한하고 추출률 보고 |

---

## 9. 코드 구조와 단계별 작업

```
mri_gg4/
├─ configs/        base.yaml (경로·seed·crop·분할), encoders/*.yaml, robust/*.yaml
├─ data/           (git 제외) extract/, nifti/, seg/, prep/, aug/, features/
├─ scripts/
│  ├─ 00_extract.py          folder + pick_*/alt_recon 시리즈만 복사, extract_qc.csv
│  ├─ 01_convert.py          NIfTI, b값 분리, RAS, 재샘플
│  ├─ 02_segment.py          TotalSegmentator total_mr, 부피, QC PNG
│  ├─ 03_preprocess.py       정규화, crop, npy 저장, preprocess_qc.csv
│  ├─ 04_augment_views.py    R1용 증강 뷰 K=8 생성(개발 집합만)
│  ├─ 05_extract_features.py 인코더 E1–E4(+E6) 고정 특징, 원본·증강 뷰
│  ├─ 06_select_cv.py        광명 2022–2024 nested CV: 인코더×풀링×서열×R0/R1 → 표
│  ├─ 07_robust_ft_cv.py     R2·R3 부분 미세조정(E1, E3) nested CV
│  ├─ 08_lock.py             LOCKED_MODEL.yaml 작성·커밋(이 시점까지 서울·광명 2025–26 성능 계산 금지)
│  ├─ 09_eval_A.py           광명 전체 재학습 → 서울 1회 평가, A′ 반대 방향
│  ├─ 10_eval_B.py           광명 2022–24 학습 → 2025–26 평가 + B-대조 50회
│  ├─ 11_eval_C.py           재구성 쌍 151건 예측 이동
│  ├─ 12_shift_metrics.py    MMD, 도메인 판별 AUROC(인코더별)
│  ├─ 13_compare.py          임상·판독의 비교, DeLong, DCA, 보정
│  └─ 14_figures.py          ROC, 보정, DCA, 이동 지표, attention
├─ src/            io_dicom.py, seg.py, preprocess.py, augment.py, encoders/, heads.py, robust.py, metrics.py, stats.py
├─ results/        tables/*.csv, figures/*.png (환자 단위 예측 git 제외)
├─ PREREG.md, LOCKED_MODEL.yaml, README.md
```

작업 순서와 확인 기준:
1. **환경:** conda env(python 3.10, torch+CUDA, monai, SimpleITK, pydicom, nibabel, TotalSegmentator, timm, scikit-learn, statsmodels, scikit-image). `requirements.lock`.
2. **코호트 재현:** §2.2 표 재현, 서버에서 못 찾은 시리즈 목록. **[중단 조건]** 서울 GG≥4가 50 미만으로 줄면 보고.
3. **00–03:** QC PNG 30건(기관·재구성별 고르게) 사람이 확인. 분할 실패율·부피 분포 보고. 서울·광명 2025–26은 QC만.
4. **DLR 플래그 확인:** DICOM 헤더(ImageType, 사설 태그, SoftwareVersions)로 `*_is_dlr`를 검증하고 불일치 목록 보고.
5. **04–05:** 특징 차원·결측 점검. 증강 뷰 예시 PNG(원본 대 증강) 10건.
6. **06–07:** 광명 2022–24 nested CV 표(인코더×풀링×서열×강건성). **이 표만 보고 주 모델을 정한다.**
7. **08:** `PREREG.md`와 `LOCKED_MODEL.yaml` 커밋 해시를 기록한 뒤에만 09–11 실행.
8. **09–14:** 표·그림 생성, TRIPOD+AI·CLAIM 2024 항목 번호를 각 표·그림에 주석으로.

---

## 10. 오프라인 서버 준비물 (로컬에서 받아 업로드)

| 받을 것 | 출처 |
|---|---|
| Triad Swin-B MAE, PlainConvUNet MAE 체크포인트 | github.com/wangshansong1/Triad, HF `ilex-hub/triad.swinb-mae.1` |
| MedicalNet ResNet-50 | github.com/Tencent/MedicalNet |
| DINOv2 ViT-B/14 | github.com/facebookresearch/dinov2 (torch hub 캐시를 통째로) |
| TotalSegmentator `total_mr` 가중치 | github.com/wasserth/TotalSegmentator (오프라인 weights 경로 설정) |
| (선택) PI-CAI baseline 가중치·코드 | github.com/DIAGNijmegen/picai_baseline |
| pip wheel | `pip download -r requirements.txt -d wheels/ --platform manylinux2014_x86_64 --python-version <서버 버전> --only-binary=:all:` |

- 각 가중치의 라이선스(학술 사용)를 확인해 `README.md`에 기록한다.
- 모델 파일과 코드만 반입하고, 환자 데이터는 기관 내부에서만 이동한다.

---

## 11. 보고 체크리스트 (요지)
- **TRIPOD+AI:** 데이터 출처·기간, 결과 정의(생검·RP 병리, max 규칙), 예측변수(영상만), 표본 크기 근거(§6.4), 결측 처리, 모델 선택과 잠금 절차, 외부·시간 검증, 보정, DCA, 공정성(장비·재구성·하위군)
- **CLAIM 2024:** 획득 파라미터 표(기관·장비·재구성별), 전처리·자동 분할·QC, reference standard, 환자 단위·기관 분리 분할, internal/external testing 용어, 실패 사례, 해석 그림

---

## 12. Claude Code에 처음 보낼 메시지 (복사용)

```
이 폴더의 mri_gg4_robustness_design.md를 끝까지 읽고, §9의 1(환경)과 2(코호트 재현)만 먼저 실행해 주세요.
external1은 파일이 매우 많으니 ls나 재귀 탐색을 하지 말고, gg4_cohort_study.csv의 folder 열로 필요한 폴더만 찾아 주세요.
§2.2 표를 재현해 보여 주고, 서버에서 찾지 못한 시리즈를 목록으로 남겨 주세요.
서울과 광명 2025–2026 데이터는 LOCKED_MODEL.yaml을 커밋하기 전까지 성능 계산에 쓰지 말아 주세요.
환자 단위 자료는 서버 밖으로 내보내지 말고, 출력은 results/ 아래 집계표로만 남겨 주세요.
```
