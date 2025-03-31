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
```
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

