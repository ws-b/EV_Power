import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# =========================
# 설정 변수
# =========================

# 차량 종류를 지정하세요. 예: 'KonaEV', 'Ioniq5', 등
car_type = 'KonaEV'

# CSV 파일들이 위치한 폴더 경로를 지정하세요.
folder_path = '/path/to/csv_folder'  # 실제 경로로 변경하세요.

# =========================
# Vehicle 클래스 정의
# =========================
class Vehicle:
    def __init__(self, mass, load, Ca, Cb, Cc, aux, hvac, idle, eff, re_brake=1):
        self.mass = mass        # kg
        self.load = load        # kg
        self.Ca = Ca * 4.44822   # lbf를 N으로 변환 (공기 저항 계수)
        self.Cb = Cb * 4.44822 * 2.237   # lbf/mph를 N/mps로 변환 (구름 저항 계수)
        self.Cc = Cc * 4.44822 * (2.237 ** 2)  # lbf/mph²를 N/mps²로 변환 (경사 저항 계수)
        self.eff = eff          # 효율
        self.aux = aux          # 보조 전력 (W)
        self.hvac = hvac        # HVAC 전력 (W)
        self.idle = idle        # 아이들 전력 (W)
        self.re_brake = re_brake  # 회생 제동 계수

# =========================
# 차량 선택 함수
# =========================
def select_vehicle(car):
    vehicles = {
        'NiroEV': Vehicle(1928, 0, 32.717, -0.19110, 0.023073, 250, 350, 0, 0.9),
        'Ioniq5': Vehicle(2268, 0, 34.342, 0.21928, 0.022718, 250, 350, 0, 0.9),
        'Ioniq6': Vehicle(2041.168, 0, 23.958, 0.15007, 0.015929, 250, 350, 0, 0.9),
        'KonaEV': Vehicle(1814, 0, 24.859, -0.20036, 0.023656, 250, 350, 0, 0.9),
        'EV6': Vehicle(2154.564, 0, 36.158, 0.29099, 0.019825, 250, 350, 0 , 0.9),
        'GV60': Vehicle(2154.564, 0, 23.290, 0.23788, 0.019822, 250, 350, 0, 0.9)
    }
    if car in vehicles:
        return vehicles[car]
    elif car in ['Bongo3EV', 'Porter2EV']:
        print(f"{car}는 전력 소비를 계산할 수 없습니다. 다른 차량을 선택하세요.")
        return None
    else:
        print("잘못된 차량 선택입니다. 올바른 차량명을 입력하세요.")
        return None

# =========================
# 파일 처리 함수
# =========================
def process_file_power(args):
    file, folder_path, EV = args
    try:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)

        # 필수 컬럼 확인
        required_columns = ['time', 'speed', 'acceleration', 'ext_temp']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"필수 컬럼 누락: {col}")

        inertia = 0.05  # 바퀴의 회전 관성
        g = 9.18        # 중력 가속도 (m/s²)

        # 'time'을 datetime 형식으로 변환하고 정렬
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        data = data.sort_values('time').reset_index(drop=True)

        # 필요한 데이터를 numpy 배열로 변환
        v = data['speed'].to_numpy()          # 속도 (m/s)
        a = data['acceleration'].to_numpy()   # 가속도 (m/s²)
        ext_temp = data['ext_temp'].to_numpy()  # 외부 온도 (°C)

        # 전력 계산 요소
        A = EV.Ca * v / EV.eff
        B = EV.Cb * v ** 2 / EV.eff
        C = EV.Cc * v ** 3 / EV.eff

        # 가속도 기반 지수 항 계산
        exp_term = np.exp(0.0411 / np.maximum(np.abs(a), 0.001))

        # 가속도에 따른 D 항 계산
        D_positive = ((1 + inertia) * (EV.mass + EV.load) * a * v) / EV.eff
        D_negative = (((1 + inertia) * (EV.mass + EV.load) * a * v) / exp_term) * EV.eff
        D = np.where(a >= 0, D_positive, np.where(EV.re_brake == 1, D_negative, 0))

        # HVAC 및 보조 전력 계산
        Eff_hvac = 0.81  # 보조 전력 효율
        target_temp = 22  # 목표 온도 (°C)
        E_hvac = np.abs(target_temp - ext_temp) * EV.hvac * Eff_hvac
        E = np.where(v <= 0.5, EV.aux + EV.idle + E_hvac, EV.aux + E_hvac)

        # 고도 데이터 처리
        if 'altitude' in data.columns and 'NONE' in data.columns:
            # 'altitude'를 숫자형으로 변환
            data['altitude'] = pd.to_numeric(data['altitude'], errors='coerce')

            # 결측치 선형 보간
            data['altitude'] = data['altitude'].interpolate(method='linear', limit_direction='both')

            altitude = data['altitude'].to_numpy()

            # 고도 차이 계산
            altitude_diff = np.diff(altitude, prepend=altitude[0])

            # 시간 차이 계산 (초 단위)
            time_diff = data['time'].diff().dt.total_seconds().fillna(0).to_numpy()

            # 거리 차이 계산 (m)
            distance_diff = v * time_diff

            # 경사각 계산 (라디안 단위)
            with np.errstate(divide='ignore', invalid='ignore'):
                slope = np.arctan2(altitude_diff, distance_diff)
                slope = np.where(distance_diff == 0, 0, slope)  # 거리 차이가 0인 경우 경사각을 0으로 설정
            data['slope'] = slope

            # 경사 저항 항 계산
            F = EV.mass * g * np.sin(slope) * v / EV.eff
        else:
            F = np.zeros_like(v)

        # 총 물리적 전력 계산
        data['Power_phys'] = A + B + C + D + E + F

        # 원본 파일에 덮어쓰기
        data.to_csv(file_path, index=False)

        return file_path  # 처리된 파일 경로 반환

    except Exception as e:
        print(f"파일 처리 중 오류 발생 ({file}): {e}")
        return None

# =========================
# 파일 병렬 처리 함수
# =========================
def process_files_power(all_files, folder_path, EV):
    # 각 파일에 대한 인자 준비
    args = [(file, folder_path, EV) for file in all_files]

    processed_files = []
    with ProcessPoolExecutor() as executor:
        # tqdm을 사용하여 진행 상황 표시
        for result in tqdm(executor.map(process_file_power, args), total=len(args)):
            if result:
                processed_files.append(result)

    print('모든 파일이 처리되었습니다.')
    return processed_files

# =========================
# 메인 처리 함수
# =========================
def main():
    # 차량 선택
    EV = select_vehicle(car_type)
    if not EV:
        return

    # 폴더 내 모든 CSV 파일 목록 가져오기
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    all_files.sort()

    if not all_files:
        print("지정된 폴더에 CSV 파일이 없습니다.")
        return

    print(f"총 {len(all_files)}개의 파일을 '{car_type}' 차량으로 처리합니다.")

    # 파일 병렬 처리
    processed_files = process_files_power(all_files, folder_path, EV)

    print("처리가 완료되었습니다.")

# =========================
# 스크립트 실행
# =========================
if __name__ == "__main__":
    main()
