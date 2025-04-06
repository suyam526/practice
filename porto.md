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

- 이렇게 하면 나중에 원하는 변수만 쉽게 선택/필터링할 수 있음

