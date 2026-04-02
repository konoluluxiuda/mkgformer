import os
import re
import subprocess
import json
from datetime import datetime

def run_experiment_k():
    k_list = [5, 10, 15, 20, 30]
    graph_file = "preprocess_paper_graph.py"
    
    with open(graph_file, "r") as f:
        graph_code = f.read()

    results = {}
    for k in k_list:
        print(f"\n{'='*40}\nRunning K = {k}\n{'='*40}")

        # 1. 仅替换 preprocess_paper_graph.py 中的 TOP_K_HERB (保持 TOP_K_DISEASE 不变)
        new_code = re.sub(r'TOP_K_HERB\s*=\s*\d+', f'TOP_K_HERB = {k}', graph_code)
        # 确保 DISEASE 是固定的最优值 15 (如果原本就是 15 则不动)
        new_code = re.sub(r'TOP_K_DISEASE\s*=\s*\d+', f'TOP_K_DISEASE = 15', new_code)
        with open(graph_file, "w") as f:
            f.write(new_code)
            
        # 2. 运行构图脚本
        print(f"  -> Building graph with K_HERB={k}...")
        subprocess.run(["python", graph_file], check=True)
        
        # 3. 运行训练脚本并捕获输出
        print("  -> Training model...")
        result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
        
        # 4. 用正则解析最终输出的 F1@10 和 Recall@10
        match = re.search(r'R@10=([0-9\.]+)\s+F1@10=([0-9\.]+)', result.stdout)
        if match:
            recall, f1 = match.groups()
            results[k] = {'Recall@10': float(recall), 'F1@10': float(f1)}
            print(f"  => Result for K={k}: F1@10={f1}, Recall@10={recall}")
        else:
            print(f"  => Failed to parse result for K={k}. Check train.py output.")
            
    # 恢复原文件
    with open(graph_file, "w") as f:
        f.write(graph_code)
        
    return results

def run_experiment_lambda():
    l_list = [0.01, 0.05, 0.1, 0.2, 0.5]
    train_file = "train.py"
    
    with open(train_file, "r") as f:
        train_code = f.read()

    results = {}
    for l in l_list:
        print(f"\n{'='*40}\nRunning Lambda = {l}\n{'='*40}")
        
        # 1. 替换 train.py 中的 CROSS_MODAL_WEIGHT 和 PROP_CHEM_WEIGHT 
        new_code = re.sub(r'CROSS_MODAL_WEIGHT\s*=\s*[0-9\.]+', f'CROSS_MODAL_WEIGHT = {l}', train_code)
        new_code = re.sub(r'PROP_CHEM_WEIGHT\s*=\s*[0-9\.]+', f'PROP_CHEM_WEIGHT = {l}', new_code)
        with open(train_file, "w") as f:
            f.write(new_code)
            
        # 2. 运行训练脚本并捕获输出
        print(f"  -> Training model with Lambda={l}...")
        result = subprocess.run(["python", train_file], capture_output=True, text=True)
        
        # 3. 解析结果
        match = re.search(r'R@10=([0-9\.]+)\s+F1@10=([0-9\.]+)', result.stdout)
        if match:
            recall, f1 = match.groups()
            # 将 l 转成字符串作为 JSON 键
            str_l = str(l)
            results[str_l] = {'Recall@10': float(recall), 'F1@10': float(f1)}
            print(f"  => Result for Lambda={l}: F1@10={f1}, Recall@10={recall}")
        else:
            print(f"  => Failed to parse result for Lambda={l}. Check train.py output.")
            
    # 恢复原文件
    with open(train_file, "w") as f:
        f.write(train_code)
        
    return results

if __name__ == "__main__":
    print("=== Phase 1: K Sensitivity ===")
    k_res = run_experiment_k()
    
    # 在开始 Phase 2 之前加一层保险，重新构图一次复原为基准配置数据
    print("\n[Restoring Base Graph for Phase 2...]")
    subprocess.run(["python", "preprocess_paper_graph.py"], check=True)
    
    print("\n=== Phase 2: Lambda Sensitivity ===")
    l_res = run_experiment_lambda()
    
    # -- 结果汇总与保存 --
    final_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "K_Sensitivity": k_res,
        "Lambda_Sensitivity": l_res
    }
    
    # 打印到控制台
    print("\n\n" + "="*40)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*40)
    print(json.dumps(final_results, indent=4))
    
    # 写入到文件
    output_file = "sensitivity_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\n✅ Results have been securely saved to: {os.path.abspath(output_file)}")