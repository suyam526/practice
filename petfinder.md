### CFG 클래스
```
class CFG:
    num_workers = 4
    size = 512
    batch_size = 32
    model_name = 'tf_efficientnet_b0_ns'
    seed = 42
    target_size = 1
    target_col = 'Pawpularity'
    n_fold = 5
```

- 머신러닝 설정(Config) 저장해두는 공간
- 한 번에 여러 설정 모아서 관리하면 유지보수 쉬움
- 추후 CFG.model_name, CFG.batch_size 이런 식으로 모델 훈련에 사용됨

-------


### Utility (유틸리티 함수들)
- get_score 함수 : 모델 성능 평가 (RMSE 계산)
    ```py
    def get_score(y_true, y_pred):
    score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
    return score
    ```
    - squared=False → RMSE (제곱근 포함)
    - squared=True → MSE (제곱까지만 계산, 더 큰 오차에 민감)

- init_logger 함수 : 로그 시스템 설정
```PY
def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()
```
    - 학습 도중 중요한 정보를 화면에 출력 + 로그 파일에 저장
    - train.log 파일에 저장되며 나중에 학습 결과 확인/디버깅 시 유용

- seed_torch 함수 : 랜덤 고정
    ```py
    def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)
    ```
    - 같은 코드로 학습 여러 번 해도 결과가 조금씩 달라질 수 있으므로<br/>
    시드를 고정해서 **랜덤성 통제**하는 것<br/>
        - random.seed(): 파이썬 내장 랜덤 고정<br/>
        - np.random.seed(): 넘파이 랜덤 고정<br/>
        등등..

----------------------------

### 전체 흐름 요약

1. 이미지에서 특징 추출
    - CustomModel 클래스로 CNN 백본을 이용해 feature vector 생성 (예: 1280차원)

2. 각 fold에 대해 feature 추출
    - 사전 학습된 모델을 불러와 test 이미지에 대해 feature 추출

3. LightGBM을 이용한 메타데이터 + 이미지 특징 조합 회귀
    - run_kfold_lightgbm 함수로 5-fold 교차 검증 수행
    - 결과 시각화 및 feature 중요도 저장

+ 딥러닝 + 회귀 모델 
- CNN 기반 딥러닝 모델 (CustomModel) : 이미지에서 복잡한 패턴을 벡터로 feature 추출 -> 1280차원의 feature vector 출력
- LightGBM 회귀 모델 : 메타데이터와 cnn에서 추출된 이미지 feature를 입력 -> 트리 기반 모델, Pawpularity 점수를 회귀로 예측

-> 메타데이터 해석이 어렵지만 이미지의 복잡한 패턴을 감지할 수 있는 CNN과 이미지 같은 고차원 비정형 데이터는 못 다루지만 수치/범주형 피쳐 처리와 해석에 강한 LightGBM 합침

-------------------

### 1. 이미지에서 특징 추출
- CustomModel 모델: 이미지 -> feature vector로 변환하는 인코더 역할 

### 2. get_features 함수
```py
def get_features(test_loader, model, device):
```
- test_loader에서 배치 단위로 이미지 불러오고
- model.feature()를 통해 feature vector 추출
- 최종적으로 Nx1280 크기의 numpy 배열 반환
-> 결과는 각 fold별로 IMG_FEATURES 리스트에 저장


### 3. LightGBM 모델 (회귀 예측 모델)
```py
def run_single_lightgbm(...)
```
- train 데이터에 이미지 feature (img_0, img_1, ..., img_1279)를 열로 추가
- 주어진 fold에 따라 train/test 데이터를 분리
- lgb.train()으로 회귀 모델 학습 (rmse 기준 조기 종료)
-> run_kfold_lightgbm()으로 위 과정을 5-fold cross-validation 방식으로 반복

### 4. feature importance 시각화
LightGBM 중요도를 평균내서 시각화 -> 어떤 feature가 중요한지 보여줌