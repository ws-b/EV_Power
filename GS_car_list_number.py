import os
import pandas as pd
import glob
import openpyxl
from openpyxl.styles import PatternFill, Border, Side, Alignment
from collections import Counter, defaultdict
from GS_vehicle_dict import vehicle_dict

# 기본 경로 설정
base_path = "D:/SamsungSTF/Data/GSmbiz/BMS_Data"

# 단말기번호 포맷팅 함수
def format_device_number(device_number):
    return f"{device_number[:3]}-{device_number[3:7]}-{device_number[7:]}"

# 기존의 모든 단말기 번호 수집
all_device_numbers = set()
for device_numbers in vehicle_dict.values():
    all_device_numbers.update(device_numbers)

# 결과를 저장할 딕셔너리 초기화
results = []

# 월별 수집 데이터 개수를 저장할 딕셔너리 초기화
monthly_counts = defaultdict(lambda: defaultdict(int))

# 디렉토리 구조 탐색 및 파일 존재 여부 확인
for device_number in all_device_numbers:
    car_model = "Unknown"
    for model, numbers in vehicle_dict.items():
        if device_number in numbers:
            car_model = model
            break

    record = {"단말기번호": format_device_number(device_number), "차종": car_model}
    collected_periods = 0  # 수집된 기간의 개수 초기화
    for year in range(2023, 2025):  # 2023년부터 2024년까지
        for month in range(1, 13):  # 1월부터 12월까지
            if year == 2025 and month > 0:
                break  # 2024년 6월까지만 포함

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
                monthly_counts[car_model][column_name] += 1

    record["기간"] = collected_periods
    results.append(record)

# 결과를 데이터프레임으로 변환
df = pd.DataFrame(results)

# 단말기번호를 기준으로 오름차순 정렬
df.sort_values(by="단말기번호", ascending=True, inplace=True)

# 차종별 차량 개수 계산
car_model_counts = Counter(df["차종"])

# 엑셀 파일로 저장 (우선 데이터를 저장한 후 스타일을 추가)
output_path = os.path.join(os.path.dirname(base_path), "단말기_번호별_차종.xlsx")
df.to_excel(output_path, index=False)

# 엑셀 파일에 차종별 차량 개수 기록 및 색상, 테두리 추가
wb = openpyxl.load_workbook(output_path)
ws = wb.active

# 차종별 차량 개수 기록을 위한 새로운 시트 추가
summary_ws = wb.create_sheet(title="차종별_차량_개수")
summary_ws.append(["차종", "차량 개수"] + [f"{str(year)[-2:]}-{month:02d}" for year in range(2023, 2025) for month in range(1, 13) if not (year == 2024 and month > 6)])
for model, count in car_model_counts.items():
    row = [model, count] + [monthly_counts[model].get(f"{str(year)[-2:]}-{month:02d}", 0) for year in range(2023, 2025) for month in range(1, 13) if not (year == 2024 and month > 6)]
    summary_ws.append(row)

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

print("차종별 차량 개수 및 파일 존재 여부 확인 완료 및 저장 완료.")