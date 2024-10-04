import folium
import pandas as pd

# CSV 파일 경로
csv_file = r"C:\Users\WSONG\Desktop\Stations.csv"

# CSV 파일 로드
try:
    df = pd.read_csv(csv_file)
    print("CSV 파일이 성공적으로 로드되었습니다.")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {csv_file}")
    exit(1)
except pd.errors.EmptyDataError:
    print("CSV 파일이 비어 있습니다.")
    exit(1)
except pd.errors.ParserError:
    print("CSV 파일 파싱 중 오류가 발생했습니다.")
    exit(1)

# 필수 컬럼 존재 여부 확인
required_columns = {'STN_KO', 'LON', 'LAT'}
if not required_columns.issubset(df.columns):
    print(f"CSV 파일에 필요한 컬럼이 없습니다. 필요한 컬럼: {required_columns}")
    exit(1)

# 데이터 정제: 결측치 제거 및 데이터 타입 변환
initial_count = len(df)
df = df.dropna(subset=['STN_KO', 'LON', 'LAT'])
df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
df = df.dropna(subset=['LON', 'LAT'])
final_count = len(df)
print(f"데이터 정제 완료: {initial_count - final_count}개의 결측치 제거됨.")

# 지도 초기 위치 설정 (서울 중심)
map_center = [37.5665, 126.9780]  # 서울의 위도와 경도
m = folium.Map(location=map_center, zoom_start=11)

# 마커 추가
for idx, row in df.iterrows():
    location = [row['LAT'], row['LON']]
    popup_text = row['STN_KO']
    folium.Marker(
        location=location,
        popup=popup_text,
        tooltip=popup_text,
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# HTML 파일로 저장
output_file = 'map.html'
m.save(output_file)
print(f"지도가 '{output_file}' 파일로 저장되었습니다.")
print("브라우저에서 'map.html' 파일을 열어 지도를 확인하세요.")
