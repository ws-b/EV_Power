import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def process_files_parking_only(start_path, save_path):
    # CSV 파일 목록 가져오기
    csv_files = [os.path.join(root, file)
                 for root, _, files in os.walk(start_path)
                 for file in files if file.endswith('.csv')]
    total_files = len(csv_files)

    # Progress bar 설정
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        # ProcessPoolExecutor를 사용하여 병렬 처리
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_wrapper, file_path, save_path) for file_path in csv_files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'파일 처리 중 예외 발생: {exc}')
                finally:
                    pbar.update(1)

    print("Processing complete")


def process_wrapper(file_path, save_path):
    try:
        process_single_file(file_path, save_path)
    except Exception as e:
        print(f"파일 {file_path} 처리 중 오류: {e}")
        raise


def process_single_file(file_path, save_path):
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"파일 {file_path}을 읽는 중 오류 발생: {e}")
        return

    # CSV 파일의 구조 확인
    if not isinstance(data, pd.DataFrame):
        print(f"파일 {file_path}은(는) DataFrame 형식이 아닙니다.")
        return

    required_columns = ['chrg_cable_conn', 'speed', 'time']
    for col in required_columns:
        if col not in data.columns:
            print(f"파일 {file_path}에 필수 열 '{col}'이(가) 없습니다.")
            return

    # 파일 이름에서 device_no와 year_month 추출
    try:
        if 'altitude' in data.columns:
            parts = file_path.split(os.sep)
            file_name = parts[-1]
            name_parts = file_name.split('_')
            device_no = name_parts[2]
            year_month = name_parts[3][:7]
        else:
            parts = file_path.split(os.sep)
            file_name = parts[-1]
            name_parts = file_name.split('_')
            device_no = name_parts[1]
            year_month = name_parts[2][:7]
    except IndexError as e:
        print(f"파일 이름 {file_path}에서 device_no 또는 year_month를 추출하는 중 오류 발생: {e}")
        return

    # 이미 처리된 파일인지 확인
    parking_file_pattern = f"bms_parking_{device_no}-{year_month}-parking-"
    try:
        existing_files = [f for f in os.listdir(save_path) if f.startswith(parking_file_pattern)]
    except FileNotFoundError:
        print(f"저장 경로 {save_path}가 존재하지 않습니다. 생성합니다.")
        os.makedirs(save_path, exist_ok=True)
        existing_files = []

    if existing_files:
        print(f"Device {device_no}과 {year_month}의 주차 파일이 이미 존재합니다. 관련 파일을 건너뜁니다.")
        return

    # 시간 형식 변환
    try:
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    except Exception as e:
        print(f"파일 {file_path}에서 'time' 열을 변환하는 중 오류 발생: {e}")
        return

    if data['time'].isnull().any():
        print(f"파일 {file_path}에 'time' 열에 잘못된 형식의 데이터가 포함되어 있습니다.")
        # 필요에 따라 잘못된 행을 제거하거나 다른 처리를 할 수 있습니다.
        # 예: data = data.dropna(subset=['time'])
        return

    # 주차 상태를 나타내는 새로운 열 추가
    data['is_parking'] = False

    # 조건 1: chrg_cable_conn == 1
    data.loc[data['chrg_cable_conn'] == 1, 'is_parking'] = True

    # 조건 2: chrg_cable_conn == 0이고 speed == 0이 10분 이상 지속
    data_conn0 = data[data['chrg_cable_conn'] == 0].copy()
    data_conn0['speed_zero'] = data_conn0['speed'] == 0

    # 연속된 speed_zero 구간을 식별하기 위해 그룹화
    data_conn0['group'] = (data_conn0['speed_zero'] != data_conn0['speed_zero'].shift()).cumsum()
    speed_zero_groups = data_conn0.groupby('group')

    # 10분 이상 지속된 speed_zero 구간을 찾아 원본 데이터에 표시
    for _, group in speed_zero_groups:
        if group['speed_zero'].all():
            duration = group['time'].iloc[-1] - group['time'].iloc[0]
            if duration >= pd.Timedelta(minutes=10):
                data.loc[group.index, 'is_parking'] = True

    # 이제 is_parking을 기반으로 주차 세그먼트 식별
    parking_segments = identify_segments(data, 'is_parking')

    if not parking_segments:
        print(f"파일에서 주차 구간을 찾을 수 없습니다: {file_path}")
        return

    parking_counter = 1  # 파일당 주차 번호 시작

    for segment in parking_segments:
        try:
            parking_segment = data.loc[segment[0]:segment[1], :]
            if check_parking_conditions(parking_segment):
                # 파일 이름 생성
                filename = f"bms_parking_{device_no}-{year_month}-parking-{parking_counter}.csv"

                # 파일 저장
                os.makedirs(save_path, exist_ok=True)
                parking_segment.to_csv(os.path.join(save_path, filename), index=False)
                parking_counter += 1
        except Exception as e:
            print(f"파일 {file_path}에서 세그먼트 {segment}를 처리하는 중 오류 발생: {e}")
            continue

    print(f"Device {device_no}과 {year_month}의 주차 파일이 성공적으로 처리되었습니다.")


def identify_segments(data, condition_column):
    """
    지정된 조건 열을 기반으로 연속된 세그먼트를 식별합니다.
    condition_column: True/False로 표시된 조건 열 (문자열)
    반환: [(start_index, end_index), ...]
    """
    if not isinstance(condition_column, str):
        raise ValueError(f"condition_column은 문자열이어야 합니다. 현재 타입: {type(condition_column)}")

    condition_series = data[condition_column]

    if not isinstance(condition_series, pd.Series):
        raise TypeError(f"data[{condition_column}]은(는) Pandas Series가 아닙니다. 현재 타입: {type(condition_series)}")

    segments = []
    in_segment = False
    start_idx = None

    for idx, is_condition in condition_series.items():
        if is_condition and not in_segment:
            in_segment = True
            start_idx = idx
        elif not is_condition and in_segment:
            in_segment = False
            end_loc = data.index.get_loc(idx) - 1
            if end_loc >= 0:
                end_idx = data.index[end_loc]
                segments.append((start_idx, end_idx))

    if in_segment:
        segments.append((start_idx, data.index[-1]))

    print(f"식별된 세그먼트: {segments}")
    return segments


def check_parking_conditions(parking_segment):
    if parking_segment.empty:
        return False

    # 주차 세그먼트의 지속 시간이 최소 10분 이상인지 확인
    duration = parking_segment['time'].iloc[-1] - parking_segment['time'].iloc[0]
    if duration < pd.Timedelta(minutes=10):
        return False

    # chrg_cable_conn == 1인 경우는 무조건 주차 상태
    # chrg_cable_conn == 0인 경우 speed == 0이 10분 이상 지속된 상태
    # 이미 'is_parking'이 True로 설정된 상태이므로 추가 검증은 필요 없을 수 있음

    return True
