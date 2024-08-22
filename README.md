# SW중심대학 공동 AI 경진대회 2023

이 프로젝트는 **SW중심대학**에서 주최한 AI 경진대회에 참가하여 위성 이미지의 건물 영역을 분할하는 AI 모델을 개발한 내용을 담고 있습니다. 본 대회에서는 다양한 알고리즘과 딥러닝 기법을 통해 높은 정확도의 이미지 분할을 목표로 하였습니다.

## 대회 개요
- **대회 기간**: 2023.07.03 ~ 2023.07.28
- **주요 키워드**: SW중심대학, 알고리즘, 비전, 객체 분할, Dice Score

## 데이터셋

### 1. train_img
- **파일**: `TRAIN_0000.png` ~ `TRAIN_7139.png`
- **해상도**: 1024 x 1024
- **설명**: 학습에 사용되는 위성 이미지.

### 2. train.csv
- **파일**: `TEST_00000.png` ~ `TEST_60639.png`
- **해상도**: 224 x 224
- **설명**: 학습 이미지의 메타데이터.

### 3. test_img
- **컬럼 설명**:
  - `img_id`: 학습 위성 이미지 샘플 ID
  - `img_path`: 학습 위성 이미지 경로 (상대 경로)
  - `mask_rle`: RLE 인코딩된 이진 마스크 (0: 배경, 1: 건물)
    - 학습 이미지에는 반드시 건물이 포함되어 있습니다.
    - 추론 이미지에는 건물이 포함되지 않을 수 있습니다.

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
- 1024 x 1024 크기의 이미지를 224 x 224 크기로 분할하여 모델 학습에 적합한 크기로 변환합니다.

#### 1-b. Contour Outlier 처리
- 후처리 단계에서, 검출된 건물 영역의 크기를 기준으로 이상치를 처리하여 보다 정확한 예측을 도모합니다.

### 2. 학습 이미지 변형
이미지의 다양성을 높이기 위해 여러 가지 변형 기법을 적용합니다:
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
본 프로젝트에서는 U-Net 아키텍처와 ResNet50을 백본으로 사용하여 이미지 분할을 수행합니다.

python


model = UNet(backbone_name='ResNet50', encoder_weights='imagenet').to(device)
성과
상위 25%: 본 모델은 대회 참가자 중 상위 25%의 성적을 기록하였으며, 효과적인 이미지 분할을 구현하였습니다.
결론
이 프로젝트를 통해 위성 이미지 데이터의 처리 및 분석, 딥러닝 모델의 학습과 평가에 대한 깊은 이해를 얻었습니다. 향후 더 많은 데이터와 다양한 모델을 활용하여 성능을 개선할 계획입니다.

기술 스택
Python
PyTorch
OpenCV
Albumentations
scikit-learn
참고 자료
Dacon Competition Overview
