# House Prices Prediction using TensorFlow Decision Forests
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
