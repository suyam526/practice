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
    - 같은 코드로 학습 여러 번 해도 결과가 조금씩 달라질 수 있으므로<br/>
    시드를 고정해서 **랜덤성 통제**하는 것
        - random.seed(): 파이썬 내장 랜덤 고정
        - np.random.seed(): 넘파이 랜덤 고정<br/>
        등등..


