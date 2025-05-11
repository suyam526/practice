# House Prices Prediction using TensorFlow Decision Forests

### 테이블 데이터와 트리 모델
- 트리 기반 모델은 결정 규칙 기반으로 분류/예측을 하므로 명확한 경계나 규칙이 있는 데이터에 강함

- House Price 데이터
  - 특징
    1. 각 행은 한 집
    2. 열은 다양한 수치형/범주형
    3. 예측할 값은 SalePrice (수치형)

  - 트리 모델과의 상성
    - 수치형/범주형 혼합 처리에 강함
      <br/>
      ex. 방 수가 3개 이상이고, 위치가 강남이면 -> 집값이 비쌈


### train, test 데이터 무작위 split
- 무작위 비율로 test/train set 나눔
    - 각 데이터 샘플이 true인 인덱스는 test set으로, false인 인덱스는 train set으로

<br/>

```py
import numpy as np

def split_dataset(dataset, test_ratio=0.30):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))
```
<br/>

```np.random.rand(len(dataset))```
- dataset의 길이(행 개수)만큼 0 이상 1 미만 무작위 실수 생성 #매번 실행마다 다름

```< test_ratio```
- 위에서 생성된 난수 배열과 test_ratio를 비교
- ex. test_ratio = 0.3일 때
```py
array([0.42, 0.89, 0.13, 0.70, 0.02]) < 0.3
→ array([False, False, True, False, True])
```
- 즉, 0.3보다 작은 위치는 TRUE => TEST SET에 포함

```test_indices```
- Boolean 배열
- 길이는 dataset과 같음
- 각 원소는 True (-> test set으로), False (->train set으로)

### Out-of-Bag(OOB)
- 랜덤 포레스트 : 여러 개의 결정 트리를 학습시켜서 평균을 내는 앙상블 모델 !
  - 각 결정 트리를 학습시킬 때, 전체 데이터셋에서 중복을 허용하며 무작위로 샘플링(부트스트랩)함
  - 그럼 아무래도 어떤 샘플은 여러 번 선택되고, 어떤 샘플은 아예 간택받지 못할 것 ,, 

- OOB란?
  - 각 트리에 대해 학습에 사용되지 않은 데이터 샘플 = 그 트리의 OOB 데이터 / 보통 한 트리에 전체 데이터의 1/3 정도
  - 🔷 OOB 사용 이유
    1. 추가 검증 세트 없이도 성능 평가 ㄱㄴ
    - 각 샘플은 자신을 학습을 사용하지 않은 여러 트리들의 예측 결과를 이용해 평가할 수 있음
    - 심지어 모델 학습 중에 자동으로 진행
    2. 추가 validation 데이터 없어도 성능 미리 확인 ㄱㄴ
    - 데이터가 작을 때 특히 유용

  - 🔷 OOB 점수는 어떻게 계산되나?
    - 데이터 샘플 A가 있다고 가정
    1. A를 학습에 사용하지 않은 여러 개의 트리들이 존재
    2. 이 트리들에 A를 넣어 예측한 값을 평균 (또는 다수결)하여 예측값 생성
    3. 실제값과 비교하여 오차 계산
    4. 이런 방식으로 전체 데이터에 대해 평가 → OOB 점수

  - 🔷 랜포에서 OOB 활용하는 법
    - 훈련 중 트리 수가 늘어날수록 OOB RMSE는 어케 되는지 기록
    - 그래프 그리면 트리 수 vs OOB 성능 지표(RMSE) 확인 ㄱㄴ
    - 이걸로 언제 트리 수를 늘려도 의미 없는지 파악할 수 있음 → 조기 종료 기준에도 사용 가능

  - 곧 성능 평가 효율적이고, 데이터 낭비 없고, 과적합 방지 가능성이 있다 