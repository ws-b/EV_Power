import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def figure4(img1_path, img2_path, img3_path, img4_path, save_path, figsize=(10, 10), dpi=300):
    """
    네 개의 이미지를 2x2 그리드로 결합하여 저장하는 함수.

    Parameters:
    - img1_path: 첫 번째 이미지 파일 경로
    - img2_path: 두 번째 이미지 파일 경로
    - img3_path: 세 번째 이미지 파일 경로
    - img4_path: 네 번째 이미지 파일 경로
    - save_path: 결합된 이미지를 저장할 경로
    - figsize: (가로, 세로) 크기, 기본값은 (10, 10)
    - dpi: 해상도, 기본값은 300
    """

    # 네 개의 이미지 로드
    img1 = mpimg.imread(img1_path)
    img2 = mpimg.imread(img2_path)
    img3 = mpimg.imread(img3_path)
    img4 = mpimg.imread(img4_path)

    # 2x2 서브플롯 생성
    fig, axs = plt.subplots(2, 2, figsize=(12,10))

    # 이미지 표시
    axs[0, 0].imshow(img1)
    axs[0, 1].imshow(img2)
    axs[1, 0].imshow(img3)
    axs[1, 1].imshow(img4)

    # 모든 축 숨기기
    for ax in axs.flat:
        ax.axis('off')

    # 라벨 추가
    labels = ['A', 'B', 'C', 'D']
    positions = [(0.02, 1.00), (0.02, 1.00), (0.02, 1.00), (0.02, 1.00)]
    for ax, label, pos in zip(axs.flat, labels, positions):
        ax.text(pos[0], pos[1], label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 사용 예시
img1_path = r"C:\Users\BSL\Desktop\Figures\Result\KonaEV_rmse_normalized.png"
img2_path = r"C:\Users\BSL\Desktop\Figures\Result\NiroEV_rmse_normalized.png"
img3_path = r"C:\Users\BSL\Desktop\Figures\Result\GV60_rmse_normalized.png"
img4_path = r"C:\Users\BSL\Desktop\Figures\Result\Ioniq6_rmse_normalized.png"
save_path = r'C:\Users\BSL\Desktop\Figures\Supplementary\figureS3.png'

figure4(img1_path, img2_path, img3_path, img4_path, save_path)
