import os
import shutil
import glob
from datetime import datetime

# 원본 폴더 경로 설정
source_dir = "/Volumes/Data/SamsungSTF/Data/GSmbiz/bms_gps_data/"

# 차종별 단말기 번호 딕셔너리
device_ids_by_vehicle = {
    'NiroEV': ['01241228149', '01241228151', '01241228153', '01241228154', '01241228155'],
    'Bongo3EV': ['01241228162', '01241228179', '01241248642', '01241248723', '01241248829'],
    'Ionic5': [
        '01241227999', '01241228003', '01241228005', '01241228007', '01241228009', '01241228014', '01241228016',
        '01241228020', '01241228024', '01241228025', '01241228026', '01241228030', '01241228037', '01241228044',
        '01241228046', '01241228047', '01241248780', '01241248782', '01241248790', '01241248811', '01241248815',
        '01241248817', '01241248820', '01241248827', '01241364543', '01241364560', '01241364570', '01241364581',
        '01241592867', '01241592868', '01241592878', '01241592896', '01241592907', '01241597801', '01241597802',
        '01241248919'
    ],
    'Ionic6': ['01241248713', '01241592904', '01241597763', '01241597804'],
    'KonaEV': [
        '01241228102', '01241228122', '01241228123', '01241228156', '01241228197', '01241228203', '01241228204',
        '01241248726', '01241248727', '01241364621', '01241124056'
    ],
    'Porter2EV': ['01241228144', '01241228160', '01241228177', '01241228188', '01241228192'],
    'EV6': [
        '01241225206', '01241228048', '01241228049', '01241228050', '01241228051', '01241228053', '01241228054',
        '01241228055', '01241228057', '01241228059', '01241228073', '01241228075', '01241228076', '01241228082',
        '01241228084', '01241228085', '01241228086', '01241228087', '01241228090', '01241228091', '01241228092',
        '01241228094', '01241228095', '01241228097', '01241228098', '01241228099', '01241228103', '01241228104',
        '01241228106', '01241228107', '01241228114', '01241228124', '01241228132', '01241228134', '01241248679',
        '01241248818', '01241248831', '01241248833', '01241248842', '01241248843', '01241248850', '01241248860',
        '01241248876', '01241248877', '01241248882', '01241248891', '01241248892', '01241248900', '01241248903',
        '01241248908', '01241248912', '01241248913', '01241248921', '01241248924', '01241248926', '01241248927',
        '01241248929', '01241248932', '01241248933', '01241248934', '01241321943', '01241321947', '01241364554',
        '01241364575', '01241364592', '01241364627', '01241364638', '01241364714'
    ],
    'GV60': ['01241228108', '01241228130', '01241228131', '01241228136', '01241228137', '01241228138'],
}

# 각 차종의 폴더를 생성하고, 해당하는 단말기 번호의 파일을 이동
for vehicle, device_ids in device_ids_by_vehicle.items():
    for device_id in device_ids:
        # 파일 이름 패턴에 따라 파일 검색
        patterns = [f"bms_{device_id}_*.csv", f"bms_altitude_{device_id}_*.csv"]
        for pattern in patterns:
            for filename in glob.glob(os.path.join(source_dir, "**", pattern), recursive=True):
                # 파일 이름에서 날짜 형식을 검사 (년-월-일.csv 형식을 제외)
                try:
                    # 파일명에서 날짜 부분만 추출
                    date_str = os.path.basename(filename).split('_')[-1].split('.')[0]
                    # 날짜 형식 확인
                    datetime.strptime(date_str, '%Y-%m-%d')
                    # 형식에 맞으면 이 파일은 건너뜀
                    continue
                except ValueError:
                    # 날짜 형식이 아니면 파일 이동 진행
                    pass

                # 대상 폴더 경로 생성
                target_dir = f"/Volumes/Data/SamsungSTF/Processed_Data/GSmbiz/{vehicle}/"
                os.makedirs(target_dir, exist_ok=True)  # 대상 폴더가 없다면 생성

                # 대상 파일 경로 생성
                target_file_path = os.path.join(target_dir, os.path.basename(filename))

                # 파일 이동
                shutil.move(filename, target_file_path)
                print(f"Moved: {filename} -> {target_file_path}")
