## 변수들에 대한 메타데이터 정리

for문 사용해 변수의 역할/데이터 유형/보존 여부/dtype/카테고리를 분류한 metadata 생성

<br/>

```py
data = [] #여기에 각 변수의 메타정보를 담을 딕셔너리 하나씩 추가

#  변수 역할 구분
if feature == 'target':
    use = 'target'
elif feature == 'id':
    use = 'id'
else:
    use = 'input'


# 변수 데이터 유형
if 'bin' in feature or feature == 'target':
    type = 'binary'  # 0/1 같은 이진값
elif 'cat' in feature or feature == 'id':
    type = 'categorical'  # 범주형 변수
elif trainset[feature].dtype == float or isinstance(trainset[feature].dtype, float):
    type = 'real'  # 실수형
elif trainset[feature].dtype == int:
    type = 'integer'  # 정수형

# 보존 여부 (preserve)
preserve = True
if feature == 'id':
    preserve = False # 분석에 필요한 변수만 보존할건데 id는 일반적으로 분석에 사용되지 않으므로 false

# dtype
dtype = trainset[feature].dtype #판다스의 원래 데이터 타입 사용

# category
if 'ind' in feature:
    category = 'individual' #사람
elif 'reg' in feature:
    category = 'registration' #등록 정보
elif 'car' in feature:
    category = 'car' #자동차 관련
elif 'calc' in feature:
    category = 'calculated' #계산된 값

# metadata 생성
feature_dictionary = {
    'varname': feature,
    'use': use,
    'type': type,
    'preserve': preserve,
    'dtype': dtype,
    'category' : category
}
data.append(feature_dictionary) # 모든 변수를 하나씩 딕셔너리로 만들어 리스트에 추가

metadata = pd.DataFrame(data, columns=['varname', 'use', 'type', 'preserve', 'dtype', 'category']) #판다스 DataFrame으로 만들고
metadata.set_index('varname', inplace=True) #varname을 인덱스로 설정
```
- varname을 인덱스로 설정한다는 것은<br/>
![prac8](./image/prac8.png)<br/>
이렇게 판다스에서는 자동으로 행의 번호를 0, 1, 2...로 붙이는데 <br/>
![prac9](./image/prac9.png)<br/>
인덱스를 varname으로 설정하면 varname 열(id, target, feature1 등..)이 0, 1, 2 대신 행의 번호로 쓰이는 것!<br/>
다시 말해, 행의 이름(label)이 'id', 'target', 'feature1' 등이 된다.

<br/>
<br/>

```py
metadata[(metadata.type == 'categorical') & (metadata.preserve)].index
# type == 'categorical' : 범주형 변수만 선택
# preserve == : True 분석에 사용할 변수만 선택
# .index : 해당 조건을 만족하는 변수 이름만 추출
```

- 이렇게 하면 나중에 원하는 변수만 쉽게 선택/필터링할 수 있음 <br/>
💡 이렇게 metadata 프레임을 만들어두면 셀프 자동화 도구를 만드는 셈! 다른 데이터셋에서도 유용하다 <br/>
특히 *변수가 30개 이상일 때*, *여러 데이터셋에 공통적으로 전처리 파이프라인을 짜야할 때*, *EDA, 모델 실험을 반복적으로 할 때* 유용함
<br/>
<br/>
<br/>

## 특정 변수만 곱하거나 제곱해서 변환하기
 실수(float).describe 결과 보고 
 
 ```py
 (pow(trainset['ps_car_12']*10,2)).head(10)

 (pow(trainset['ps_car_15'],2)).head(10)
 ```

 - 'ps_car_12'는 10을 곱한 후 제곱, 'ps_car_15'는 제곱함. 이유는?🤔
    - ps_car_12: 자연수 제곱근을 10으로 나눈 값
        - 실제 자연수형 데이터를 √ 연산 + 10으로 나눈 것.
        - 예: 차량 마력 수, 무게, 연식 등 실수형인데 너무 큰 값을 줄이기 위해 변형했을 가능성
    > ps_car_12는 √값이니까 → 제곱하면 원래 자연수 값 복원<br/> √4 / 10 = 0.2 → 제곱하면 0.04.<br/>
    그러니까 제곱하고 10을 곱하면 원래값을 간접적으로 복원 가능

    <br/>

  - ps_car_15: 
    - 이건 정수를 단순히 √ 연산만 한 값이라 카테고리적인 숫자 특성이 있거나,
    - 아니면 어떤 순서형 특성에 루트만 씌운 케이스일 수도 있음
    > √ 자연수니까 → 제곱하면 정수로 복원됨!
<br/>
<br/>

 💡 ps_car_12, ps_car_15는 원래 자연수 값에서 √ 연산을 거쳐 변형된 것이고, 분석자는 **이걸 제곱함으로써 원래 의미 있는 정수로 복원하려 한 것**


➡️ 이외에 *10으로 스케일 키우고, 제곱으로 비선형 변환(ex. 작은 차이가 더 큰 영향을 미치게)까지 해서 변수 영향력 강조하기도 함!

<br/>
<br/>
<br/>

## Smoothing 기법
