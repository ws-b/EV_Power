import os
import numpy as np
import random
import matplotlib.pyplot as plt
from threading import Thread
from GS_Functions import compute_rmse
from GS_Train_XGboost import run_workflow as xgb_run_workflow
from GS_Train_Only_XGboost import run_workflow as only_run_workflow
from GS_Train_LinearR import cross_validate as lr_cross_validate
from GS_Train_Only_LR import cross_validate as only_lr_validate
import json
def run_xgb_cross_validate(sampled_vehicle_files, selected_car, adjusted_params_XGB, results_dict, size):
    """
    샘플링된 데이터를 사용해 '하이브리드 XGBoost' 모델 RMSE를 계산하고 results_dict에 저장하는 함수.
    별도 스레드에서 실행될 함수를 정의.
    """
    try:
        # run_workflow 호출(하이브리드 XGBoost)
        #   predefined_best_params 에 이미 조정(스케일링)된 파라미터를 넘김
        xgb_results, xgb_scaler, _ = xgb_run_workflow(
            sampled_vehicle_files,         # 샘플링된 파일 목록
            selected_car,
            plot=False,                    # 시각화 X
            save_dir=None,                 # 모델 저장 X
            predefined_best_params=adjusted_params_XGB
        )
        if xgb_results:
            # RMSE만 뽑아서 기록
            rmse_values = [xgb_result['rmse'] for xgb_result in xgb_results]
            results_dict[selected_car][size].append({
                'model': 'Hybrid Model(XGBoost)',
                'rmse': rmse_values
            })
    except Exception as e:
        print(f"XGBoost cross_validate error: {e}")

def run_only_xgb_validate(sampled_vehicle_files, selected_car, adjusted_params_ML, results_dict, size):
    """
    샘플링된 데이터를 사용해 'Only ML(XGBoost)' 모델 RMSE를 계산하고 results_dict에 저장하는 함수.
    별도 스레드에서 실행될 함수를 정의.
    """
    try:
        # run_workflow 호출(Only XGBoost)
        only_xgb_results, only_xgb_scaler, _ = only_run_workflow(
            sampled_vehicle_files,
            selected_car,
            plot=False,
            save_dir=None,
            predefined_best_params=adjusted_params_ML
        )
        if only_xgb_results:
            rmse_values = [only_xgb_result['rmse'] for only_xgb_result in only_xgb_results]
            results_dict[selected_car][size].append({
                'model': 'Only ML(XGBoost)',
                'rmse': rmse_values
            })
    except Exception as e:
        print(f"Only ML(XGBoost) cross_validate error: {e}")

def run_evaluate(vehicle_files, selected_car):
    """
    - vehicle_files: { '차량명': [csv파일 경로들], ... }
    - selected_car: 처리하려는 차량명

    1) 차량별 최대 데이터셋 크기에 따른 여러 개의 size를 지정(10, 20, 50, ...)
       가능한 size별로 샘플링 여러 번 -> Physics & ML 모델들의 RMSE를 구함
    2) 전체 데이터셋을 먼저 사용해 Optuna 등으로 구한 best_params를 얻어옴
    3) size별로 random sample을 뽑아 RMSE 측정
    4) 측정 결과를 results_dict에 저장 후 json으로 출력
    5) (옵션) plot_rmse_results로 시각화
    """
    # 샘플링할 파일 개수 후보
    vehicle_file_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]

    # best_params 같은 하이퍼파라미터 기록용
    l2lambda = {selected_car: []}
    # RMSE 등 모델별 결과 저장용
    results_dict = {selected_car: {}}

    # 전체 파일(트립) 수
    max_samples = len(vehicle_files[selected_car])

    # 1) 실제로 샘플링할 수 있는 size만 필터링
    filtered_vehicle_file_sizes = [size for size in vehicle_file_sizes if size <= max_samples]
    if max_samples not in filtered_vehicle_file_sizes:
        filtered_vehicle_file_sizes.append(max_samples)
    # 오름차순 정렬
    filtered_vehicle_file_sizes = sorted(filtered_vehicle_file_sizes)

    # 2) 전체 데이터를 통해 얻은 최적 하이퍼파라미터(하이브리드XGB + OnlyXGB)
    #   - plot=False, save_dir=None, predefined_best_params=None
    #     => 내부적으로 Optuna를 돌려서 best_params 획득
    results_hybrid_xgb, _ = xgb_run_workflow(vehicle_files, selected_car, False, None, None)
    # 예: results_hybrid_xgb = {'best_params': {'eta':0.1, 'reg_alpha':1, ...}, 'rmse': ...} 등등

    # Hybrid XGBoost의 best_params 저장
    l2lambda[selected_car].append({
        'model': 'Hybrid Model(XGBoost)',
        'params': results_hybrid_xgb[0]['best_params']
    })

    # Only XGBoost 모델의 best_params 저장
    results_only_xgb, _ = only_run_workflow(vehicle_files, selected_car,False, None, None)
    l2lambda[selected_car].append({
        'model': 'Only ML(XGBoost)',
        'params': results_only_xgb[0]['best_params']
    })

    # 전체 dataset 크기
    N_total = max_samples

    # 3) size별로 여러 번 샘플링
    for size in filtered_vehicle_file_sizes:
        if size not in results_dict[selected_car]:
            results_dict[selected_car][size] = []

        # 사이즈에 따라 샘플링 횟수 다르게 (small size -> 많이, large size -> 적게)
        if size < 20:
            samplings = 200
        elif 20 <= size < 50:
            samplings = 10
        elif 50 <= size <= 100:
            samplings = 5
        else:
            samplings = 1

        # samplings 번 만큼 반복
        for sampling in range(samplings):
            # random.sample로 size개 추출
            sampled_files = random.sample(vehicle_files[selected_car], size)
            sampled_vehicle_files = {selected_car: sampled_files}

            # (a) Physics 기반 모델 RMSE 계산
            rmse_physics = compute_rmse(sampled_vehicle_files, selected_car)
            if rmse_physics is not None:
                results_dict[selected_car][size].append({
                    'model': 'Physics-Based',
                    'rmse': [rmse_physics]
                })

            # (b) 파라미터 스케일링
            #     - (전체데이터크기 / 샘플사이즈)만큼 곱해줌
            #     - eta(학습률)는 고정
            adjustment_factor = N_total / size

            # 하이브리드 XGBoost용 파라미터
            adjusted_params_XGB = results_hybrid_xgb['best_params'].copy()
            for param in adjusted_params_XGB:
                if param != 'eta':
                    adjusted_params_XGB[param] *= adjustment_factor

            # Only XGBoost용 파라미터
            adjusted_params_ML = results_only_xgb['best_params'].copy()
            for param in adjusted_params_ML:
                if param != 'eta':
                    adjusted_params_ML[param] *= adjustment_factor

            # (c) 멀티스레드로 XGBoost 훈련/검증
            xgb_thread = Thread(
                target=run_xgb_cross_validate,
                args=(sampled_vehicle_files, selected_car, adjusted_params_XGB, results_dict, size)
            )
            ml_thread = Thread(
                target=run_only_xgb_validate,
                args=(sampled_vehicle_files, selected_car, adjusted_params_ML, results_dict, size)
            )
            xgb_thread.start()
            ml_thread.start()
            xgb_thread.join()
            ml_thread.join()

            # (d) Hybrid Model (Linear Regression) 실행
            try:
                hybrid_lr_results, hybrid_lr_scaler = lr_cross_validate(sampled_vehicle_files, selected_car)
                if hybrid_lr_results:
                    hybrid_lr_rmse_values = [result['rmse'] for result in hybrid_lr_results]
                    results_dict[selected_car][size].append({
                        'model': 'Hybrid Model(Linear Regression)',
                        'rmse': hybrid_lr_rmse_values
                    })
            except Exception as e:
                print(f"Linear Regression cross_validate error: {e}")

            # (e) Only LR 실행
            try:
                only_lr_results, only_lr_scaler = only_lr_validate(sampled_vehicle_files, selected_car)
                if only_lr_results:
                    only_lr_rmse_values = [result['rmse'] for result in only_lr_results]
                    results_dict[selected_car][size].append({
                        'model': 'Only ML(LR)',
                        'rmse': only_lr_rmse_values
                    })
            except Exception as e:
                print(f"Only LR cross_validate error: {e}")

    # 모든 size, 모든 샘플링에 대한 결과 출력
    print(results_dict)
    print("---------------------------\n")
    print(l2lambda)

    # 결과를 파일로 저장 (json)
    output_file_name = f"{selected_car}_results.txt"
    output_file_path = os.path.join("C:\\Users\\BSL\\Desktop", output_file_name)
    with open(output_file_path, 'w') as outfile:
        json.dump(results_dict, outfile)

    return results_dict

def plot_rmse_results(results_dict, selected_car, save_path):
    """
    1) results_dict에서 모델별 RMSE 값을 읽어 size(트립 개수)별로 정렬
    2) Physics-Based 모델 값을 기준(1.0)으로 정규화
    3) matplotlib으로 로그 스케일 플롯
    """
    results_car = results_dict[selected_car]
    # 키(사이즈)를 정수로 바꿔서 딕셔너리를 재구성
    results_car = {int(size): data for size, data in results_car.items()}
    sizes = sorted(results_car.keys())
    # 10개 미만의 size는 제외 (예시)
    sizes = [s for s in sizes if s >= 10]

    # 평균/표준편차(혹은 분산) 계산을 위한 리스트
    phys_rmse_mean = []
    phys_rmse_std = []
    xgb_rmse_mean = []
    xgb_rmse_std = []
    lr_rmse_mean = []
    lr_rmse_std = []
    only_ml_rmse_mean = []
    only_ml_rmse_std = []
    only_lr_rmse_mean = []
    only_lr_rmse_std = []

    # size별로 모델별 RMSE 추출 & 평균/표준편차
    for size in sizes:
        # 각 모델별 RMSE 리스트 뽑기
        phys_values = [
            item
            for result in results_car[size] if result['model'] == 'Physics-Based'
            for item in result['rmse']
        ]
        xgb_values = [
            item
            for result in results_car[size] if result['model'] == 'Hybrid Model(XGBoost)'
            for item in result['rmse']
        ]
        lr_values = [
            item
            for result in results_car[size] if result['model'] == 'Hybrid Model(Linear Regression)'
            for item in result['rmse']
        ]
        only_ml_values = [
            item
            for result in results_car[size] if result['model'] == 'Only ML(XGBoost)'
            for item in result['rmse']
        ]
        only_lr_values = [
            item
            for result in results_car[size] if result['model'] == 'Only ML(LR)'
            for item in result['rmse']
        ]

        # 각 모델별 평균 & 표준편차 계산
        if phys_values:
            phys_mean = np.mean(phys_values)
            phys_std  = np.std(phys_values)
        else:
            # Physics RMSE가 없으면 임의의 기본값(1.0) 사용
            phys_mean = 1.0
            phys_std  = 0.0
        phys_rmse_mean.append(phys_mean)
        phys_rmse_std.append(phys_std)

        if xgb_values:
            xgb_mean = np.mean(xgb_values)
            xgb_std  = np.std(xgb_values)
            xgb_rmse_mean.append(xgb_mean)
            xgb_rmse_std.append(xgb_std)
        else:
            xgb_rmse_mean.append(None)
            xgb_rmse_std.append(None)

        if lr_values:
            lr_mean = np.mean(lr_values)
            lr_std  = np.std(lr_values)
            lr_rmse_mean.append(lr_mean)
            lr_rmse_std.append(lr_std)
        else:
            lr_rmse_mean.append(None)
            lr_rmse_std.append(None)

        if only_ml_values:
            only_ml_mean = np.mean(only_ml_values)
            only_ml_std  = np.std(only_ml_values)
            only_ml_rmse_mean.append(only_ml_mean)
            only_ml_rmse_std.append(only_ml_std)
        else:
            only_ml_rmse_mean.append(None)
            only_ml_rmse_std.append(None)

        if only_lr_values:
            only_lr_mean = np.mean(only_lr_values)
            only_lr_std  = np.std(only_lr_values)
            only_lr_rmse_mean.append(only_lr_mean)
            only_lr_rmse_std.append(only_lr_std)
        else:
            only_lr_rmse_mean.append(None)
            only_lr_rmse_std.append(None)

    # Physics-Based RMSE를 기준(=1)으로 정규화
    #   - 실제 연구나 논문에서는 “baseline 대비 성능”을 보는 용도로 사용
    normalized_xgb_rmse_mean = [
        x / p if (x is not None and p != 0) else None
        for x, p in zip(xgb_rmse_mean, phys_rmse_mean)
    ]
    normalized_xgb_rmse_std = [
        (x / p if (x is not None and p != 0) else 0) * 0.6745
        for x, p in zip(xgb_rmse_std, phys_rmse_mean)
    ]

    normalized_lr_rmse_mean = [
        x / p if (x is not None and p != 0) else None
        for x, p in zip(lr_rmse_mean, phys_rmse_mean)
    ]
    normalized_lr_rmse_std = [
        (x / p if (x is not None and p != 0) else 0) * 0.6745
        for x, p in zip(lr_rmse_std, phys_rmse_mean)
    ]

    normalized_only_ml_rmse_mean = [
        x / p if (x is not None and p != 0) else None
        for x, p in zip(only_ml_rmse_mean, phys_rmse_mean)
    ]
    normalized_only_ml_rmse_std = [
        (x / p if (x is not None and p != 0) else 0) * 0.6745
        for x, p in zip(only_ml_rmse_std, phys_rmse_mean)
    ]

    normalized_only_lr_rmse_mean = [
        x / p if (x is not None and p != 0) else None
        for x, p in zip(only_lr_rmse_mean, phys_rmse_mean)
    ]
    normalized_only_lr_rmse_std = [
        (x / p if (x is not None and p != 0) else 0) * 0.6745
        for x, p in zip(only_lr_rmse_std, phys_rmse_mean)
    ]

    # Physics-based 모델은 무조건 1.0으로 세팅
    normalized_phys_rmse_mean = [1.0 for _ in phys_rmse_mean]
    normalized_phys_rmse_std  = [0.0 for _ in phys_rmse_mean]

    # ---------------------------
    # 그래프 그리기(여기서는 errorbar 대신 plot 사용)
    # ---------------------------
    plt.figure(figsize=(6, 5))

    # Physics-Based
    plt.plot(
        sizes,
        normalized_phys_rmse_mean,
        label='Physics-Based', linestyle='--', color='#747678ff'
    )
    # Only ML(LR)
    plt.plot(
        sizes,
        normalized_only_lr_rmse_mean,
        label='Only ML(LR)', marker='o', color='#0073c2ff'
    )
    # Only ML(XGB)
    plt.plot(
        sizes,
        normalized_only_ml_rmse_mean,
        label='Only ML(XGB)', marker='o', color='#efc000ff'
    )
    # Hybrid LR
    plt.plot(
        sizes,
        normalized_lr_rmse_mean,
        label='Hybrid Model(LR)', marker='o', color='#cd534cff'
    )
    # Hybrid XGB
    plt.plot(
        sizes,
        normalized_xgb_rmse_mean,
        label='Hybrid Model(XGB)', marker='D', color='#20854eff'
    )

    plt.xlabel('Number of Trips')
    plt.ylabel('Normalized RMSE')
    plt.title(f'RMSE vs Number of Trips for {selected_car}')
    plt.legend()
    plt.grid(False)
    plt.xscale('log')

    # x축 tick을 size 목록으로 표시
    plt.xticks(sizes, [str(size) for size in sizes], rotation=45)
    # x축 범위 살짝 조정
    plt.xlim(min(sizes) - 1, max(sizes) - 1)

    plt.tight_layout()

    # 결과 저장
    if save_path:
        plt.savefig(
            os.path.join(save_path, f"{selected_car}_rmse_normalized.png"),
            dpi=300
        )
    plt.show()