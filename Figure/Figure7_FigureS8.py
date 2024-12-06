from PIL import Image

# 이미지 경로 설정
img_paths = [
    r"C:\Users\BSL\Desktop\Figures\Figure7_KonaEV_Composite.png",
    r"C:\Users\BSL\Desktop\Figures\Figure7_GV60_Composite.png"
]

# 이미지 불러오기
images = [Image.open(path) for path in img_paths]

# 최대 너비와 총 높이 계산
max_width = max(img.width for img in images)
total_height = sum(img.height for img in images)

# 새로운 이미지 생성 (RGB 모드, 배경은 검정색으로 설정)
combined_img = Image.new('RGB', (max_width, total_height), (0, 0, 0))

# 이미지 붙여넣기
current_height = 0
for img in images:
    combined_img.paste(img, (0, current_height))
    current_height += img.height

# 결과 저장
combined_img_path = r"C:\Users\BSL\Desktop\Figures\Supplementary\figureS8.png"
combined_img.save(combined_img_path, dpi=(300, 300))

# 결과 표시
combined_img.show()
