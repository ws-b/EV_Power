import os
import pandas as pd

# 경로 설정
gps_altitude_base_path = r"C:\Users\BSL\Desktop\BMS_GPS_ZIP\gps_altitude"
merged_base_path = r"C:\Users\BSL\Desktop\BMS_GPS_ZIP\merged"

# 차종 리스트만 사용
vehicles = ['NiroEV', 'Ioniq5', 'Ioniq6', 'KonaEV', 'EV6', 'GV60', 'Porter2EV', 'Bongo3EV']


def merge_gps_with_bms_for_all():
    # 각 차종에 대해 순회
    for vehicle in vehicles:
        vehicle_dir = os.path.join(merged_base_path, vehicle)
        if not os.path.isdir(vehicle_dir):
            print(f"Skipped: No merged directory found for {vehicle}")
            continue

        # 각 차종 디렉토리 내의 모든 bms_altitude로 시작하는 월별 파일을 순회
        for merged_file in os.listdir(vehicle_dir):
            if not merged_file.startswith('bms_altitude') or not merged_file.endswith('.csv'):
                continue

            merged_file_path = os.path.join(vehicle_dir, merged_file)
            bms_df = pd.read_csv(merged_file_path)

            # 'time' 컬럼의 포맷 확인 및 변환
            try:
                # 'time' 컬럼을 datetime으로 변환 시도 (기존 포맷이 '%Y-%m-%d %H:%M:%S'인지 확인)
                pd.to_datetime(bms_df['time'], format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                # 기존 포맷이 다를 경우 변환 수행
                bms_df['time'] = pd.to_datetime(bms_df['time'], format='%y-%m-%d %H:%M:%S').dt.strftime(
                    '%Y-%m-%d %H:%M:%S')

            # 단말기 번호와 연-월을 파일명에서 추출
            filename_parts = merged_file.split('_')
            device_number = filename_parts[2]
            year_month = filename_parts[3].split('.')[0]

            # gps_altitude 경로 설정
            gps_dir = os.path.join(gps_altitude_base_path, device_number, year_month)
            if not os.path.isdir(gps_dir):
                print(f"Skipped: No GPS data directory found for {device_number} in {year_month}")
                continue

            # gps 일별 파일을 순회하며 데이터 병합
            for gps_file in os.listdir(gps_dir):
                if not gps_file.endswith('.csv'):
                    continue

                gps_file_path = os.path.join(gps_dir, gps_file)
                gps_df = pd.read_csv(gps_file_path)

                # 'time' 컬럼을 datetime으로 변환
                gps_df['time'] = pd.to_datetime(gps_df['time'], format='%Y-%m-%d %H:%M:%S')
                bms_df['time'] = pd.to_datetime(bms_df['time'], format='%Y-%m-%d %H:%M:%S')

                # 가장 가까운 시간의 lat, lng 값을 찾아서 병합
                closest_times = gps_df['time'].searchsorted(bms_df['time'], side='left')
                closest_times = closest_times.clip(0, len(gps_df) - 1)  # 경계를 넘어가는 값을 조정

                bms_df['closest_lat'] = gps_df['lat'].iloc[closest_times].values
                bms_df['closest_lng'] = gps_df['lng'].iloc[closest_times].values

                # 'altitude' 값이 존재하는 곳에서만 업데이트
                bms_df.loc[bms_df['altitude'].notna(), 'lat'] = bms_df.loc[bms_df['altitude'].notna(), 'closest_lat']
                bms_df.loc[bms_df['altitude'].notna(), 'lng'] = bms_df.loc[bms_df['altitude'].notna(), 'closest_lng']

                # 임시 열 삭제
                bms_df.drop(columns=['closest_lat', 'closest_lng'], inplace=True)

            # 결과 저장 (덮어쓰기)
            bms_df.to_csv(merged_file_path, index=False)
            print(f"Merged file saved to: {merged_file_path}")


# 전체 차종에 대해 병합 작업 수행
merge_gps_with_bms_for_all()
