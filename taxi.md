## Heatmap of common locations from where pickup and dropoff occurs

1. Folium을 사용한 지도 시각화
- 히트맵으로 픽업 위치와 횟수를 지도에 시각화
```py
from folium.plugins import HeatMap
import folium

# 지도 객체 생성
pickup_map = folium.Map(location=[40.730610, -73.935242], zoom_start=10)

# Num_Trips를 float으로 변환
heat_data = list(zip(pickup.pickup_latitude_round3.values,
                     pickup.pickup_longitude_round3.values,
                     pickup.Num_Trips.values.astype(float)))  # <- 기존 코드 오류나서 이 부분 추가

# HeatMap 생성
hm_wide = HeatMap(heat_data,
                  min_opacity=0.2,
                  radius=5, blur=15,
                  max_zoom=1)

# 지도에 HeatMap 추가
pickup_map.add_child(hm_wide)

pickup_map
```

<br/>
<br/>

2. Matplotlib을 사용해 산점도 시각화
- 1번과는 다른 시각화 방식으로, 픽업 위치들을 평면상에서 scatter plot으로 보여줌
- 밀도보다 위치 분포 확인할 때 유용

<br/>
<br/>
<br/>

## bearing 계산
- pickup 지점에서 dropoff 지점까지의 방위각(bearing) 을 계산해서 train 데이터프레임의 bearing 열에 저장
    - bearing(방위각)이란?
        - 두 지점 간의 방향 (angle)
        - 북쪽을 기준(0도)으로 시계방향으로 돌아간 각도, 한 지점에서 다른 지점까지 가야 할 방향을 나타냄
            - 북쪽 방향: 0도
            - 동쪽 방향: 90도
            - 남쪽 방향: 180도
            - 서쪽 방향: 270도
    - 계산 이유
        - 두 지점 간의 '이동 방향' 분석
        - 예: 택시 데이터에서
            - 승객이 어느 방향으로 이동했는지 → 동서남북 방향 추정
            - 도심 ↔ 외곽 방향인지 확인
            - 특정 방향의 이동만 이상하게 오래 걸리는지 분석
            - 운전 경로가 비정상적으로 돌아간 건 아닌지 탐지
            - 출퇴근 시간대에 특정 방향으로 쏠림 현상 있는지 파악
<br/>
<br/>

```py
def calculateBearing(lat1,lng1,lat2,lng2):
    R = 6371 
    lng_delta_rad = np.radians(lng2 - lng1) #경도 차이를 라디안 단위로 변환 
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) #위도와 경도를 라디안으로 변환 (삼각함수에서 라디안 단위 사용)
    y = np.sin(lng_delta_rad) * np.cos(lat2) #방위각 공식의 분자와 분모 계산    
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad) #atan2(y, x)로 라디안 단위의 각도를 반환 → 최종적으로 도(degree) 단위로 변환하여 반환
    return np.degrees(np.arctan2(y, x)
```

- 각 row에 대해 pickup → dropoff 방향의 bearing 계산
- round3 => 소수점 3자리로 반올림된 위도/경도 (정확한 matching을 위해 자주 사용)

<br/>
<br/>

### bearing 계산 후 bearing vs trip duration (log scale)
```py
Bearing vs Trip Duration¶
plt.figure(figsize=(8,5))
plt.scatter(train['bearing'].values,y=np.log(train['trip_duration'].values))
plt.xlabel("Bearing")
plt.ylabel("Trip Duration (log scale)")
```

- bearing: 어느 방향으로 이동했는지
- trip_duration : 그 방향으로 이동할 때 얼마나 걸렸는지
    - **방향성에 따른 특성 차이를 발견**하기 위한 시각화 <br/>
    ex. 남쪽(180도)으로 가는 경로는 시간이 길다? 그럼 교통체증이 심한가? 북동쪽(45도~90도) 방향은 고속도로라서 시간이 짧은건가? 특정 방향에서만 이상하게 오래 걸리는 trip이 있다면 이상치 분석해보자!
    - 이처럼 단순히 거리뿐만 아니라 방향도 중요하기 때문에 분석 시 방향에 따른 이동 시간 패턴이 뚜렷하다면 **bearing을 모델 feature로 넣는 것이 유의미**해질 것! 
