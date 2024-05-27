import os
import pandas as pd
import glob
import openpyxl
from openpyxl.styles import PatternFill, Border, Side, Alignment

# 차종-단말기번호 매핑
car_models = {
    "니로EV": ['01241228149', '01241228151', '01241228153', '01241228154', '01241228155'],
    "봉고3EV": ['01241228162', '01241228179', '01241248642', '01241248723', '01241248829'],
    "아이오닉5": [
        '01241227999', '01241228003', '01241228005', '01241228007', '01241228009', '01241228014', '01241228016',
        '01241228020', '01241228024', '01241228025', '01241228026', '01241228030', '01241228037', '01241228044',
        '01241228046', '01241228047', '01241248780', '01241248782', '01241248790', '01241248811', '01241248815',
        '01241248817', '01241248820', '01241248827', '01241364543', '01241364560', '01241364570', '01241364581',
        '01241592867', '01241592868', '01241592878', '01241592896', '01241592907', '01241597801', '01241597802',
        '01241248919', '01241321944'
    ],
    "아이오닉6": ['01241248713', '01241592904', '01241597763', '01241597804'],
    "코나EV": [
        '01241228102', '01241228122', '01241228123', '01241228156', '01241228197', '01241228203', '01241228204',
        '01241248726', '01241248727', '01241364621', '01241124056'
    ],
    "포터2EV": ['01241228144', '01241228160', '01241228177', '01241228188', '01241228192', '01241228171'],
    "EV6": [
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
    "GV60": ['01241228108', '01241228130', '01241228131', '01241228136', '01241228137', '01241228138']
}

# 기본 경로 설정
base_path = "D:/SamsungSTF/Data/GSmbiz/BMS_Data"

# 단말기번호 포맷팅 함수
def format_device_number(device_number):
    return f"{device_number[:3]}-{device_number[3:7]}-{device_number[7:]}"

# 기존의 모든 단말기 번호 수집
all_device_numbers = set()
for device_numbers in car_models.values():
    all_device_numbers.update(device_numbers)

# 결과를 저장할 딕셔너리 초기화
results = []

# 디렉토리 구조 탐색 및 파일 존재 여부 확인
for device_number in all_device_numbers:
    car_model = "Unknown"
    for model, numbers in car_models.items():
        if device_number in numbers:
            car_model = model
            break

    record = {"단말기번호": format_device_number(device_number), "차종": car_model}
    collected_periods = 0  # 수집된 기간의 개수 초기화
    for year in range(2023, 2025):  # 2023년부터 2024년까지
        for month in range(1, 13):  # 1월부터 12월까지
            if year == 2024 and month > 4:
                break  # 2024년 4월까지만 포함

            year_month = f"{year}-{month:02d}"
            folder_path = os.path.join(base_path, device_number, year_month)
            bms_pattern = os.path.join(folder_path, f"bms_{device_number}_{year_month}-*.csv")
            bms_altitude_pattern = os.path.join(folder_path, f"bms_altitude_{device_number}_{year_month}-*.csv")

            # 패턴에 맞는 파일 목록 가져오기
            bms_files = glob.glob(bms_pattern)
            bms_altitude_files = glob.glob(bms_altitude_pattern)

            # 파일 존재 여부 확인
            if bms_files and not bms_altitude_files:
                status = "bms_파일만 존재"
            elif bms_altitude_files:
                status = "bms_altitude_파일 존재"
            else:
                status = "파일 없음"

            file_count = len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0
            column_name = f"{str(year)[-2:]}-{month:02d}"
            record[column_name] = f"{status} ({file_count})" if file_count > 0 else None  # 파일 개수가 0이면 비워둠
            # 파일이 존재하면 수집된 기간 증가
            if file_count > 0:
                collected_periods += 1

    record["기간"] = collected_periods
    results.append(record)


# 결과를 데이터프레임으로 변환
df = pd.DataFrame(results)

# 단말기번호를 기준으로 오름차순 정렬
df.sort_values(by="단말기번호", ascending=True, inplace=True)

# 엑셀 파일로 저장 (우선 데이터를 저장한 후 스타일을 추가)
output_path = os.path.join(os.path.dirname(base_path), "단말기_번호별_차종.xlsx")
df.to_excel(output_path, index=False)

# 엑셀 파일에 색상 및 테두리 추가
wb = openpyxl.load_workbook(output_path)
ws = wb.active

# 색상, 테두리 및 가운데 정렬 설정
green_fill = PatternFill(start_color="CCFFCC", end_color="CCFFCC", fill_type="solid")  # 연한 초록색
blue_fill = PatternFill(start_color="CCCCFF", end_color="CCCCFF", fill_type="solid")   # 연한 파란색
yellow_fill = PatternFill(start_color="FFFFCC", end_color="FFFFCC", fill_type="solid") # 연한 노란색
red_fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")    # 연한 빨간색
light_salmon_fill = PatternFill(start_color="FFA07A", end_color="FFA07A", fill_type="solid") # 연한 살몬색
light_pink_fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")   # 연한 분홍색
pink_fill = PatternFill(start_color="FFCCFF", end_color="FFCCFF", fill_type="solid")   # 연한 핑크색


border = Border(left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin'))
center_alignment = Alignment(horizontal='center', vertical='center')

# 데이터 셀 스타일 설정
for row in ws.iter_rows(min_row=2, min_col=3, max_col=ws.max_column-1, max_row=ws.max_row):
    for cell in row:
        if cell.value:
            if "bms_파일만 존재" in cell.value:
                cell.fill = green_fill
                file_count = int(cell.value.split()[-1].strip('()'))  # 파일 개수만 남김
                cell.value = file_count
            elif "bms_altitude_파일 존재" in cell.value:
                cell.fill = blue_fill
                file_count = int(cell.value.split()[-1].strip('()'))  # 파일 개수만 남김
                cell.value = file_count
        cell.alignment = center_alignment

# 단말기번호와 차종 셀 스타일 설정 및 노란색 배경 추가
for row in ws.iter_rows(min_row=2, min_col=1, max_col=2, max_row=ws.max_row):
    for cell in row:
        cell.alignment = center_alignment
        cell.fill = yellow_fill

for row in ws.iter_rows(min_row=2, min_col=ws.max_column, max_col=ws.max_column, max_row=ws.max_row):
    for cell in row:
        if cell.value == 12:
            cell.fill = light_salmon_fill
        elif cell.value < 12 and cell.value > 7:
            cell.fill = red_fill
        elif cell.value < 8 and cell.value > 3:
            cell.fill = light_pink_fill
        else:
            cell.fill = pink_fill
        cell.alignment = center_alignment

# 테두리를 추가하기 위해 모든 셀을 순회
for row in ws.iter_rows(min_row=1, min_col=1, max_col=ws.max_column, max_row=ws.max_row):
    for cell in row:
        cell.border = border

# 열 너비 조정
for col in ws.columns:
    max_length = 0
    column = col[0].column_letter  # 열 이름을 가져옵니다.
    for cell in col:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(cell.value)
        except:
            pass
    adjusted_width = (max_length + 2)
    ws.column_dimensions[column].width = adjusted_width

# 차종 열 너비 조정 (기존 너비의 1.5배로 설정)
ws.column_dimensions['B'].width = ws.column_dimensions['B'].width * 1.5

# 저장
wb.save(output_path)

print("파일 존재 여부 및 파일 개수 확인 완료 및 저장 완료.")