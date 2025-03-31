## train, test 함께 처리하는 방식
```py
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
```

```py
for dataset in combine :
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
```

- combine으로 묶은 후, train, test 데이터셋을 같은 방식으로 수정하고 싶을 때 for문을 사용해서 각각 같은 방식으로 바꾸기 (train 한번, test 한번 각각 바꿈)


## Age 결측치 채우기
```py
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median() # 중앙값 사용

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
                        
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
```

- Age 결측치 채우기 위해 ```Sex```와 ```Pclass```를 기준으로 각 그룹의 Age의 중앙값을 계산하고, 이를 결측값에 대체!
- Sex(남성, 여성)와 Pclass(1,2,3)의 조합별로 중앙값을 구하고, 그 값을 Age 컬럼의 결측치에 할당해 데이터 채움
- EX. Sex가 '남성', Pclass가 '1'인 그룹의 Age 값이 결측치일 경우 그 그룹의 다른 '남성'이면서 '1'인 사람들의 Age 중앙값을 계산 

## 두 가지 특성의 상호작용을 고려하기 위해 Age*Class라는 새로운 특성 만들기
- 같은 범주로 묶을 수 있는 Parch, SibSp을 더해서 FamilySize를 만드는 경우는 봤어도 아예 다른 특성을 곱하는건 또 처음 본다..! 왜 Age*Pclass를 만들까?
    - 이 데이터에서 Age와 Pclass는 모두 Survived에 영향을 주는 중요한 변수
        - Pclass : 1등급 객실 승객의 생존율이 가장 높고, 3등급 승객의 생존율이 가장 낮음
        - Age : 어린아이일수록 구조될 가능성이 높고, 나이가 많을수록 생존율이 낮을 가능성이 있음
    - 즉, Age와 Pclass를 따로 보는 것보다 이 둘의 곱을 새로운 변수로 만들어서 생존율과의 관계를 더 잘 파악할 수 있음!

    - 나이가 많고(Age↑), 객실 등급이 낮으면(Pclass↑) → 위험한 상황<br/>
    → 3등급(Pclass=3) 객실을 이용하는 50세 승객(Age=50)은 Age*Class = 50 * 3 = 150이 됨.<br/>
    → 높은 값일수록 생존 가능성이 낮을 수도 있음.

    - 나이가 적고(Age↓), 객실 등급이 높으면(Pclass↓) → 유리한 상황<br/>
    → 1등급(Pclass=1) 객실을 이용하는 10세 승객(Age=10)은 Age*Class = 10 * 1 = 10이 됨.<br/>
    → 낮은 값일수록 생존 가능성이 높을 수도 있음.



