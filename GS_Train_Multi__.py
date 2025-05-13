import os
import random
import json
from threading import Thread
import numpy as np # Numpy import 추가
import matplotlib.pyplot as plt

# ------------------------------
# 고정 경로 설정 – 사용자 요청
# ------------------------------
# GS_Model_Main.py에서 base_processed_path를 사용하므로, DATA_FOLDER는 여기서 필수적이지 않을 수 있음
# RESULT_DIR은 GS_Model_Main.py에서 comparison_save_path 등으로 전달받는 것이 더 유연함
# 여기서는 기존 GS_Train_Multi__.py의 방식을 따르되, GS_Model_Main.py와 연동 시 경로 전달 방식을 고려해야 함.
DATA_FOLDER = "/home/ubuntu/SamsungSTF/Processed_Data/TripByTrip" # 필요시 GS_Model_Main에서 전달받도록 수정 가능
RESULT_DIR = "/home/ubuntu/SamsungSTF/Results_Multi"  # 결과 JSON 및 그림 저장 위치 (폴더명 명확히)

# 결과 저장 폴더가 없으면 생성
os.makedirs(RESULT_DIR, exist_ok=True)
FIGURE_SAVE_DIR = os.path.join(RESULT_DIR, "Figures") # 그림 저장용 하위 폴더
os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)


# ------------------------------
# 내부 모듈 import
# ------------------------------
from GS_vehicle_dict import vehicle_dict
from GS_Functions import get_vehicle_files, compute_rmse # get_vehicle_files는 main에서 사용

# Hybrid / Only XGBoost
from GS_Train_XGboost import run_workflow as xgb_run_workflow
from GS_Train_Only_XGboost import run_workflow as only_xgb_run_workflow

# Hybrid / Only Linear Regression
from GS_Train_LinearR import train_validate_test as lr_run_workflow
# GS_Train_Only_LR.py의 train_validate_test 함수는 vehicle_files 인자만 받으므로 selected_car 불필요
from GS_Train_Only_LR import train_validate_test as only_lr_run_workflow_direct

# Hybrid DNN / LSTM / Random-Forest
from GS_Train_DNN import run_dnn_workflow
from GS_Train_LSTM import run_lstm_workflow # GS_Train_LSTM.py가 존재하고 run_lstm_workflow 함수가 정의되어 있어야 함
from GS_Train_RF import run_rf_workflow

# ------------------------------
# 스레드에서 실행할 래퍼 함수
# ------------------------------

def _run_and_record(wrapper_fn, label, sampled_files_dict, selected_car, results_list_for_size, **kwargs_for_wrapper):
    """
    공통 래퍼 – 성공 시 RMSE 리스트를 results_list_for_size 에 기록.
    sampled_files_dict: {'selected_car': [file_paths]} 형태
    results_list_for_size: 특정 샘플 크기에 대한 결과가 추가될 리스트
    """
    try:
        # lr_run_workflow와 only_lr_run_workflow_direct는 selected_car 인자를 받지 않음
        if wrapper_fn == lr_run_workflow or wrapper_fn == only_lr_run_workflow_direct:
            # GS_Train_LinearR.py 와 GS_Train_Only_LR.py 는 vehicle_files 인자로 sampled_files_dict[selected_car] 리스트를 직접 받음
            results, _ = wrapper_fn(sampled_files_dict[selected_car], **kwargs_for_wrapper)
        else:
            results, _ = wrapper_fn(sampled_files_dict, selected_car, **kwargs_for_wrapper)
            
        if results:
            # 각 workflow의 반환값인 results 리스트에서 'rmse' 또는 'test_rmse' 키를 찾음
            rmse_values = []
            for r_item in results:
                if "rmse" in r_item and r_item["rmse"] is not None:
                    rmse_values.append(r_item["rmse"])
                elif "test_rmse" in r_item and r_item["test_rmse"] is not None: # LinearR 계열 모델용
                    rmse_values.append(r_item["test_rmse"])
            
            if rmse_values: # RMSE 값이 있는 경우에만 추가
                 results_list_for_size.append({"model": label, "rmse": rmse_values})
    except Exception as e:
        print(f"오류 발생 ({label}, 차량: {selected_car}): {e}")


# ------------------------------
# 평가 메인 루프 (기존 evaluate_single_vehicle 함수)
# ------------------------------
def run_evaluate(vehicle_files_all_cars: dict[str, list[str]], selected_car_to_eval: str):
    """
    한 차량에 대해 샘플링(데이터 수, 반복 횟수) 기반으로 여러 ML 모델 + Physics 모델을 평가.
    GS_Model_Main.py에서 호출될 함수.
    """

    print(f"=== {selected_car_to_eval} 모델 비교 평가 시작 ===")
    print(f"총 트립 수: {len(vehicle_files_all_cars[selected_car_to_eval])}")

    base_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]
    max_samples_for_car = len(vehicle_files_all_cars[selected_car_to_eval])
    
    # 실제 사용 가능한 샘플 크기 필터링 및 최대 샘플 크기 추가
    actual_sample_sizes = sorted(list(set(s for s in base_sizes if s <= max_samples_for_car) | {max_samples_for_car}))
    if not actual_sample_sizes:
        print(f"{selected_car_to_eval}: 평가할 샘플 크기가 없습니다 (트립 부족).")
        return {}

    print(f"평가할 샘플 크기: {actual_sample_sizes}")

    # -------------------------------------------------
    # 1) 전체 데이터로 하이퍼파라미터 튜닝 (한 번만 실행) - 필요한 모델만
    # -------------------------------------------------
    print(f"\n[{selected_car_to_eval} 전체 데이터 기반 하이퍼파라미터 튜닝 (필요시)]")
    
    best_params_xgb, best_params_only_xgb, best_params_dnn, best_params_lstm = None, None, None, None

    try:
        print("  - Hybrid XGBoost 파라미터 탐색...")
        full_xgb_results, _ = xgb_run_workflow(vehicle_files_all_cars, selected_car_to_eval, plot=False, save_dir=None, predefined_best_params=None)
        if full_xgb_results and "best_params" in full_xgb_results[0]:
            best_params_xgb = full_xgb_results[0]["best_params"]
            print(f"    Hybrid XGBoost 최적 파라미터: {best_params_xgb}")
        else:
            print("    Hybrid XGBoost 파라미터 탐색 실패 또는 결과 없음.")
    except Exception as e:
        print(f"    Hybrid XGBoost 파라미터 탐색 중 오류: {e}")

    try:
        print("  - Only ML(XGBoost) 파라미터 탐색...")
        # GS_Train_Only_XGboost.py의 run_workflow는 results, scaler, _ (3개) 또는 results, scaler (2개)를 반환할 수 있음.
        # 여기서는 2개만 받는 것으로 통일 (이전 검토 내용 반영)
        full_only_xgb_results, _ = only_xgb_run_workflow(vehicle_files_all_cars, selected_car_to_eval, plot=False, save_dir=None, predefined_best_params=None)
        if full_only_xgb_results and "best_params" in full_only_xgb_results[0]:
            best_params_only_xgb = full_only_xgb_results[0]["best_params"]
            print(f"    Only ML(XGBoost) 최적 파라미터: {best_params_only_xgb}")
        else:
            print("    Only ML(XGBoost) 파라미터 탐색 실패 또는 결과 없음.")
    except Exception as e:
        print(f"    Only ML(XGBoost) 파라미터 탐색 중 오류: {e}")
    
    try:
        print("  - Hybrid DNN 파라미터 탐색...")
        full_dnn_results, _ = run_dnn_workflow(vehicle_files_all_cars, selected_car_to_eval, plot=False, save_dir=None, predefined_best_params=None)
        if full_dnn_results and "best_params" in full_dnn_results[0]:
            best_params_dnn = full_dnn_results[0]["best_params"]
            print(f"    Hybrid DNN 최적 파라미터: {best_params_dnn}")
        else:
            print("    Hybrid DNN 파라미터 탐색 실패 또는 결과 없음.")
    except Exception as e:
        print(f"    Hybrid DNN 파라미터 탐색 중 오류: {e}")

    # LSTM은 sequence_length 인자가 필요
    default_sequence_length = 60
    try:
        print(f"  - Hybrid LSTM (sequence_length={default_sequence_length}) 파라미터 탐색...")
        full_lstm_results, _ = run_lstm_workflow(vehicle_files_all_cars, selected_car_to_eval, 
                                                 sequence_length=default_sequence_length, 
                                                 plot=False, save_dir=None, predefined_best_params=None)
        if full_lstm_results and "best_params" in full_lstm_results[0]:
            best_params_lstm = full_lstm_results[0]["best_params"]
            print(f"    Hybrid LSTM 최적 파라미터: {best_params_lstm}")
        else:
            print("    Hybrid LSTM 파라미터 탐색 실패 또는 결과 없음.")
    except NameError: # GS_Train_LSTM 모듈 또는 함수가 없을 경우
        print("    경고: GS_Train_LSTM 모듈 또는 run_lstm_workflow 함수를 찾을 수 없습니다. LSTM 평가는 건너<0xEB><0xA4>니다.")
    except Exception as e:
        print(f"    Hybrid LSTM 파라미터 탐색 중 오류: {e}")


    # 최종 결과를 저장할 딕셔너리 (키: 샘플 크기, 값: 해당 크기에서의 모델별 결과 리스트)
    # 예: {10: [{'model': 'Physics-Based', 'rmse': [2.5]}, {'model': 'Hybrid XGBoost', 'rmse': [1.8]} ...], 20: [...]}
    vehicle_eval_results: dict[int, list] = {size_val: [] for size_val in actual_sample_sizes}

    # ------------------------------
    # 2) 샘플 크기별 반복 평가
    # ------------------------------
    print(f"\n[{selected_car_to_eval} 샘플 크기별 모델 평가 시작]")
    for current_sample_size in actual_sample_sizes:
        # 샘플링 반복 횟수 결정
        if current_sample_size < 20:
            num_repeats = 20 # 작은 샘플은 많이 반복하여 안정적인 평균 RMSE 확보 (시간 제약 고려하여 조정)
        elif current_sample_size < 50:
            num_repeats = 10
        elif current_sample_size <= 100:
            num_repeats = 5
        else:
            num_repeats = 1 # 큰 샘플은 1회 또는 적게 (시간 제약 고려)
        
        print(f"  샘플 크기: {current_sample_size}, 반복 횟수: {num_repeats}")

        for i_repeat in range(num_repeats):
            print(f"    - 반복 {i_repeat+1}/{num_repeats}")
            # 현재 차량의 전체 파일 목록에서 current_sample_size 만큼 랜덤 샘플링
            sampled_trip_files = random.sample(vehicle_files_all_cars[selected_car_to_eval], current_sample_size)
            # 각 workflow 함수에 전달할 형태로 딕셔너리 생성
            sampled_files_for_eval = {selected_car_to_eval: sampled_trip_files}

            # 1. Physics-Based 모델 RMSE 계산
            try:
                rmse_physics = compute_rmse(sampled_files_for_eval, selected_car_to_eval)
                if rmse_physics is not None:
                    vehicle_eval_results[current_sample_size].append({"model": "Physics-Based", "rmse": [rmse_physics]})
            except Exception as e:
                print(f"    Physics-Based 모델 RMSE 계산 오류: {e}")

            # 2. ML 모델 평가 (스레드 사용)
            threads: list[Thread] = []
            
            # kwargs 공통 설정 (plot=False, save_dir=None)
            common_kwargs = dict(plot=False, save_dir=None)

            # ---- XGBoost (Hybrid) ----
            if best_params_xgb:
                # 파라미터 조정 로직 (기존 GS_Train_Multi.py 방식과 유사하게 eta 제외하고 조정)
                adj_factor = max_samples_for_car / current_sample_size if current_sample_size > 0 else 1
                current_xgb_params = {
                    k: (v if k == 'eta' else max(1e-9, v * adj_factor) if isinstance(v, (int, float)) else v) 
                    for k, v in best_params_xgb.items()
                }
                threads.append(Thread(target=_run_and_record, 
                                       args=(xgb_run_workflow, "Hybrid XGBoost", sampled_files_for_eval, selected_car_to_eval, vehicle_eval_results[current_sample_size]), 
                                       kwargs=dict(**common_kwargs, predefined_best_params=current_xgb_params)))
            else:
                print(f"    Hybrid XGBoost 평가 건너<0xEB><0xA4> (최적 파라미터 없음)")


            # ---- Only ML(XGBoost) ----
            if best_params_only_xgb:
                adj_factor = max_samples_for_car / current_sample_size if current_sample_size > 0 else 1
                current_only_xgb_params = {
                    k: (v if k == 'eta' else max(1e-9, v * adj_factor) if isinstance(v, (int, float)) else v)
                    for k, v in best_params_only_xgb.items()
                }
                threads.append(Thread(target=_run_and_record,
                                       args=(only_xgb_run_workflow, "Only ML(XGBoost)", sampled_files_for_eval, selected_car_to_eval, vehicle_eval_results[current_sample_size]),
                                       kwargs=dict(**common_kwargs, predefined_best_params=current_only_xgb_params)))
            else:
                print(f"    Only ML(XGBoost) 평가 건너<0xEB><0xA4> (최적 파라미터 없음)")

            # ---- Hybrid Linear Regression ----
            # lr_run_workflow는 selected_car 인자를 직접 받지 않고, vehicle_files (여기서는 샘플링된 파일 리스트)만 받음
            threads.append(Thread(target=_run_and_record,
                                   args=(lr_run_workflow, "Hybrid LinearR", sampled_files_for_eval, selected_car_to_eval, vehicle_eval_results[current_sample_size]),
                                   kwargs=common_kwargs)) # predefined_best_params 없음

            # ---- Only ML(Linear Regression) ----
            threads.append(Thread(target=_run_and_record,
                                   args=(only_lr_run_workflow_direct, "Only ML(LinearR)", sampled_files_for_eval, selected_car_to_eval, vehicle_eval_results[current_sample_size]),
                                   kwargs=common_kwargs)) # predefined_best_params 없음
            
            # ---- Hybrid DNN ----
            if best_params_dnn:
                # DNN은 일반적으로 파라미터 스케일링을 하지 않음. 전체 데이터로 찾은 파라미터 사용.
                threads.append(Thread(target=_run_and_record,
                                       args=(run_dnn_workflow, "Hybrid DNN", sampled_files_for_eval, selected_car_to_eval, vehicle_eval_results[current_sample_size]),
                                       kwargs=dict(**common_kwargs, predefined_best_params=best_params_dnn)))
            else:
                print(f"    Hybrid DNN 평가 건너<0xEB><0xA4> (최적 파라미터 없음)")

            # ---- Hybrid LSTM ----
            if 'run_lstm_workflow' in globals() and best_params_lstm: # LSTM 함수 존재하고 파라미터 있을 때만
                threads.append(Thread(target=_run_and_record,
                                       args=(run_lstm_workflow, "Hybrid LSTM", sampled_files_for_eval, selected_car_to_eval, vehicle_eval_results[current_sample_size]),
                                       kwargs=dict(**common_kwargs, predefined_best_params=best_params_lstm, sequence_length=default_sequence_length)))
            elif 'run_lstm_workflow' in globals() and not best_params_lstm:
                 print(f"    Hybrid LSTM 평가 건너<0xEB><0xA4> (최적 파라미터 없음)")


            # ---- Hybrid Random Forest ----
            # RF는 일반적으로 파라미터 스케일링보다는 고정된 좋은 값 사용 또는 샘플별 튜닝 필요. 여기서는 튜닝 없이 기본값으로 실행.
            threads.append(Thread(target=_run_and_record,
                                   args=(run_rf_workflow, "Hybrid RandomForest", sampled_files_for_eval, selected_car_to_eval, vehicle_eval_results[current_sample_size]),
                                   kwargs=common_kwargs))


            # 모든 스레드 시작 및 종료 대기
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        
        print(f"  샘플 크기 {current_sample_size} 완료.\n")

    # 결과 JSON 파일로 저장
    json_output_filename = f"ML_Eval_Results_{selected_car_to_eval}.json"
    json_output_path = os.path.join(RESULT_DIR, json_output_filename)
    try:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(vehicle_eval_results, f, ensure_ascii=False, indent=4)
        print(f"결과가 {json_output_path} 에 저장되었습니다.")
    except Exception as e:
        print(f"결과를 JSON 파일로 저장하는 중 오류 발생: {e}")

    return vehicle_eval_results


# ------------------------------
# 결과 시각화 함수
# ------------------------------
def plot_rmse_results(results_dict_for_car: dict, car_name: str, save_dir: str):
    """
    단일 차량에 대한 샘플 크기별 모델 RMSE 결과를 시각화합니다.
    results_dict_for_car: run_evaluate가 반환한 특정 차량의 결과 딕셔너리.
                          {10: [{'model': 'M1', 'rmse': [val1, val2...]}, ...], 20: ...}
    car_name: 그래프 제목 및 파일명에 사용될 차량 이름.
    save_dir: 그래프 이미지를 저장할 디렉토리 (FIGURE_SAVE_DIR 사용).
    """
    if not results_dict_for_car:
        print(f"{car_name}에 대한 시각화할 결과 데이터가 없습니다.")
        return

    # 결과에서 등장하는 모든 모델 이름 수집
    model_names_present = sorted(list(set(
        item["model"] for size_data in results_dict_for_car.values() for item in size_data
    )))

    # 샘플 크기 (x축 값)
    sample_sizes = sorted(results_dict_for_car.keys())

    # 모델별 색상 및 마커 정의 (필요시 확장)
    model_styles = {
        "Physics-Based": {"color": "#747678", "marker": None, "linestyle": '--'},
        "Only ML(LinearR)": {"color": "#0073c2", "marker": "o", "linestyle": '-'},
        "Only ML(XGBoost)": {"color": "#efc000", "marker": "s", "linestyle": '-'},
        "Hybrid LinearR": {"color": "#cd534c", "marker": "^", "linestyle": '-'},
        "Hybrid XGBoost": {"color": "#20854e", "marker": "D", "linestyle": '-'},
        "Hybrid DNN": {"color": "#8a2be2", "marker": "p", "linestyle": '-'},
        "Hybrid LSTM": {"color": "#ff7f0e", "marker": "*", "linestyle": '-'},
        "Hybrid RandomForest": {"color": "#1f77b4", "marker": "X", "linestyle": '-'} # 기존 Only ML(LR)과 색상 중복될 수 있어 변경
    }
    # 기본 스타일 (위 딕셔너리에 없는 모델용)
    default_style = {"marker": "x", "linestyle": ':'}


    plt.figure(figsize=(12, 7)) # 그래프 크기 조정

    # Physics-Based 모델의 평균 RMSE 값을 기준으로 다른 모델들의 RMSE를 정규화
    # 각 샘플 크기별 Physics-Based 모델의 평균 RMSE를 먼저 계산
    physics_avg_rmse_per_size = {}
    for size_val in sample_sizes:
        physics_rmses = [
            rmse_val for item in results_dict_for_car.get(size_val, [])
            if item["model"] == "Physics-Based" and item.get("rmse")
            for rmse_val in item["rmse"] # item["rmse"]는 리스트일 수 있음
        ]
        if physics_rmses:
            physics_avg_rmse_per_size[size_val] = np.mean(physics_rmses)
        else:
            physics_avg_rmse_per_size[size_val] = 1.0 # Physics RMSE가 없으면 정규화 기준을 1로 (오류 방지)


    for model_name in model_names_present:
        avg_normalized_rmses = []
        std_normalized_rmses = [] # 정규화된 RMSE의 표준편차 (옵션)
        
        plot_sample_sizes = [] # 실제 데이터가 있는 샘플 크기만 플롯

        for size_val in sample_sizes:
            # 현재 모델, 현재 샘플 크기의 모든 RMSE 값 (정규화 전)
            current_model_rmses_at_size = [
                rmse_val for item in results_dict_for_car.get(size_val, [])
                if item["model"] == model_name and item.get("rmse")
                for rmse_val in item["rmse"]
            ]

            if current_model_rmses_at_size:
                # Physics-Based 모델의 평균 RMSE로 정규화
                baseline_rmse = physics_avg_rmse_per_size.get(size_val, 1.0)
                if baseline_rmse == 0: baseline_rmse = 1.0 # 0으로 나누기 방지

                normalized_rmses_at_size = [val / baseline_rmse for val in current_model_rmses_at_size]
                
                avg_normalized_rmses.append(np.mean(normalized_rmses_at_size))
                if len(normalized_rmses_at_size) > 1: # 반복 횟수가 2 이상일 때만 표준편차 의미 있음
                    std_normalized_rmses.append(np.std(normalized_rmses_at_size))
                else:
                    std_normalized_rmses.append(0) # 표준편차 0으로 처리
                plot_sample_sizes.append(size_val)
        
        if not avg_normalized_rmses: # 해당 모델에 대한 정규화된 RMSE 데이터가 없으면 건너뛰기
            continue

        style = model_styles.get(model_name, default_style.copy())
        
        plt.plot(plot_sample_sizes, avg_normalized_rmses, 
                 label=model_name, 
                 marker=style.get("marker"), 
                 linestyle=style.get("linestyle"), 
                 color=style.get("color"))
        
        # 표준편차 음영 처리 (선택적)
        if any(s > 0 for s in std_normalized_rmses) and len(plot_sample_sizes) == len(std_normalized_rmses):
             avg_np = np.array(avg_normalized_rmses)
             std_np = np.array(std_normalized_rmses)
             # 오차범위가 너무 커서 그래프가 어지러워 보일 수 있으므로, 신중히 사용
             # 예: plt.fill_between(plot_sample_sizes, avg_np - std_np, avg_np + std_np, color=style.get("color"), alpha=0.15)
             pass # 여기서는 fill_between 생략


    plt.xlabel("샘플 크기 (트립 수)")
    plt.ylabel("정규화된 RMSE (Physics-Based 모델 대비)")
    plt.title(f"{car_name}: 샘플 크기에 따른 모델별 정규화된 RMSE")
    
    if len(sample_sizes) > 1 and (max(sample_sizes) / min(sample_sizes) > 50): # 샘플 크기 범위가 넓으면 로그 스케일
        plt.xscale('log')
        # 로그 스케일일 때 x축 눈금을 명시적으로 표시 (가독성 향상)
        # 실제 sample_sizes 값을 눈금으로 사용
        plt.xticks(sample_sizes, [str(s) for s in sample_sizes], rotation=45, ha="right")
    else:
        plt.xticks(rotation=45, ha="right")


    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # 범례 위치 조정
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # 범례가 잘리지 않도록 레이아웃 조정 (오른쪽 여백 확보)

    # 그림 파일로 저장
    output_plot_filename = f"RMSE_Comparison_Normalized_{car_name}.png"
    output_plot_path = os.path.join(save_dir, output_plot_filename) # save_dir은 FIGURE_SAVE_DIR 사용
    try:
        plt.savefig(output_plot_path, dpi=300)
        print(f"RMSE 비교 그래프가 {output_plot_path} 에 저장되었습니다.")
    except Exception as e:
        print(f"RMSE 비교 그래프를 저장하는 중 오류 발생: {e}")
    # plt.show() # 로컬 환경에서 직접 보려면 활성화
    plt.close() # Figure 객체 메모리 해제


# ------------------------------
# entry point (GS_Model_Main.py에서 직접 run_evaluate와 plot_rmse_results를 호출)
# ------------------------------
if __name__ == "__main__":
    # 이 부분은 GS_Model_Main.py에서 제어하므로, 여기서는 테스트 또는 단독 실행용으로 남겨둘 수 있음
    print("GS_Train_Multi__.py 단독 실행 모드 (테스트용)")

    # 테스트를 위한 임시 vehicle_dict 및 파일 목록 (실제 경로로 대체 필요)
    # 예시: 실제로는 GS_vehicle_dict와 GS_Functions.get_vehicle_files를 통해 로드
    vehicle_dict_sample = {'EV6': ['file1.csv', 'file2.csv', ...], 'Ioniq5': [...]} 
    
    # DATA_FOLDER는 스크립트 상단에 정의된 것을 사용
    selected_cars_for_test, vehicle_files_for_test = get_vehicle_files(
        {i + 1: name for i, name in enumerate(vehicle_dict.keys())}, 
        DATA_FOLDER, 
        vehicle_dict
    )

    if not selected_cars_for_test:
        print("테스트할 차량이 선택되지 않았습니다.")
    else:
        all_evaluation_results = {}
        for car_name_test in selected_cars_for_test:
            print(f"\n>>> {car_name_test} 차량 평가 시작 (테스트 모드) <<<")
            # vehicle_files_for_test 에는 모든 선택된 차량의 파일 목록이 들어있음
            # run_evaluate는 이 전체 딕셔너리와 평가할 특정 차량 이름을 받음
            results_for_single_car = run_evaluate(vehicle_files_for_test, car_name_test)
            if results_for_single_car:
                 all_evaluation_results[car_name_test] = results_for_single_car
                 # FIGURE_SAVE_DIR은 스크립트 상단에 정의된 경로 사용
                 plot_rmse_results(results_for_single_car, car_name_test, FIGURE_SAVE_DIR)
            else:
                print(f"{car_name_test} 차량에 대한 평가 결과가 없습니다.")

        # 모든 차량에 대한 종합 결과도 별도 파일로 저장 (선택적)
        if all_evaluation_results:
            all_results_json_path = os.path.join(RESULT_DIR, "ML_Eval_Results_All_Selected.json")
            try:
                with open(all_results_json_path, "w", encoding="utf-8") as f_all:
                    json.dump(all_evaluation_results, f_all, ensure_ascii=False, indent=4)
                print(f"\n모든 선택된 차량의 종합 평가 결과가 {all_results_json_path} 에 저장되었습니다.")
            except Exception as e:
                print(f"모든 차량 종합 결과를 JSON 파일로 저장하는 중 오류 발생: {e}")
        
        print("\nGS_Train_Multi__.py 테스트 실행 완료.")