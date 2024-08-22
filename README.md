# SW중심대학 공동 AI 경진대회 2023

위성 이미지의 건물 영역 분할을 수행하는 AI 모델을 개발하는 대회입니다. 
- **대회 기간**: 2023.07.03 ~ 2023.07.28
- **주요 키워드**: SW중심대학, 알고리즘, 비전, 객체 분할, Dice Score

## 데이터셋

### 1. train_img
- 이미지 파일: `TRAIN_0000.png` ~ `TRAIN_7139.png`
- 해상도: 1024 x 1024

### 2. train.csv
- 이미지 파일: `TEST_00000.png` ~ `TEST_60639.png`
- 해상도: 224 x 224

### 3. test_img
- **컬럼 설명**:
  - `img_id`: 학습 위성 이미지 샘플 ID
  - `img_path`: 학습 위성 이미지 경로 (상대 경로)
  - `mask_rle`: RLE 인코딩된 이진 마스크 (0: 배경, 1: 건물)
    - 학습 이미지에는 반드시 건물이 포함되어 있습니다.
    - 추론 이미지에는 건물이 포함되지 않을 수 있습니다.
    - 촬영 해상도: 학습 이미지 0.5m/픽셀, 추론 이미지 해상도는 비공개.

### 4. test.csv
- **컬럼 설명**:
  - `img_id`: 추론 위성 이미지 샘플 ID
  - `img_path`: 추론 위성 이미지 경로 (상대 경로)

### 5. sample_submission.csv
- **컬럼 설명**:
  - `img_id`: 추론 위성 이미지 샘플 ID
  - `mask_rle`: RLE 인코딩된 예측 이진 마스크 (0: 배경, 1: 건물)
    - 예측 결과에 건물이 없는 경우 반드시 -1로 처리.

## 모델 및 방법론

### 1. 데이터 처리
#### 1-a. 이미지 분할
- 1024 x 1024 크기의 이미지를 224 x 224 크기로 분할합니다.

#### 1-b. Contour Outlier 처리
- 후처리 단계에서, 검출된 건물 영역의 크기를 기준으로 이상치를 처리합니다.

### 2. 학습 이미지 변형
```python
# Augmentation
aug_transform = Compose([
    Resize(224, 224),
    HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평으로 뒤집음
    A.VerticalFlip(p=0.5),   # 50% 확률로 이미지를 수직으로 뒤집음
    A.Rotate(limit=30),       # -30도에서 30도 사이의 각도로 이미지를 무작위로 회전
    RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),  # 컬러 변형
    Normalize(),              # 이미지를 정규화
    ToTensorV2()             # PyTorch tensor로 변환
])
3. 모델 구조
python


model = UNet(backbone_name='ResNet50', encoder_weights='imagenet').to(device)
성과
상위 25%: U-Net + ResNet50 + Contour


이 내용을 `README.md` 파일에 추가하시면 됩니다. 필요한 추가 내용이나 수정 사항이 있으면 말씀해 주세요!
