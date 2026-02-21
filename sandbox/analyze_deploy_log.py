import re
import sys
from collections import defaultdict

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def max_val(lst):
    return max(lst) if lst else 0.0

def diffs(lst):
    return [lst[i+1] - lst[i] for i in range(len(lst)-1)]

def corrcoef(x, y):
    if len(x) < 2: return float('nan')
    mx = mean(x)
    my = mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = sum((xi - mx)**2 for xi in x)
    den_y = sum((yi - my)**2 for yi in y)
    if den_x == 0 or den_y == 0: return float('nan')
    return num / ((den_x * den_y) ** 0.5)

def std_dev(lst):
    if len(lst) < 2: return 0.0
    m = mean(lst)
    return (sum((x - m)**2 for x in lst) / len(lst)) ** 0.5

def main(log_path):
    deploy_re = re.compile(r">>> DEPLOY epsilon-sim:.*?model=(\w+).*?profile=(\w+)")
    line_re = re.compile(r"inst_seed=(\d+).*?eps=(\d+).*?exact\(E=([0-9.]+).*?t=([0-9.]+)s\).*?guided\(E=([0-9.]+).*?t=([0-9.]+)s\)")

    current_model = None
    data = defaultdict(lambda: defaultdict(list))
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                m = deploy_re.search(line)
                if m:
                    current_model = m.group(1)
                    continue
                if current_model:
                    m2 = line_re.search(line)
                    if m2:
                        seed = int(m2.group(1))
                        eps = int(m2.group(2))
                        exact_e = float(m2.group(3))
                        exact_t = float(m2.group(4))
                        guided_e = float(m2.group(5))
                        guided_t = float(m2.group(6))
                        data[current_model][seed].append({
                            'eps': eps,
                            'exact': exact_e,
                            'exact_t': exact_t,
                            'guided': guided_e,
                            'guided_t': guided_t
                        })
    except FileNotFoundError:
        print(f"Error: Could not find log file at {log_path}")
        sys.exit(1)

    if not data:
        print(f"No valid deploy metrics found in {log_path}.")
        sys.exit(0)
                        
    for model, seeds_data in data.items():
        print(f"============================================================")
        print(f"MODEL: {model.upper()}")
        print(f"============================================================")
        
        all_gaps = []
        all_exact_times = []
        all_guided_times = []
        mono_exact_count = 0
        mono_guided_count = 0
        mono_total = 0
        correlations = []
        
        for seed, records in seeds_data.items():
            records.sort(key=lambda x: x['eps'], reverse=True)
            
            eps_list = [r['eps'] for r in records]
            exact_list = [r['exact'] for r in records]
            guided_list = [r['guided'] for r in records]
            
            exact_times = [r['exact_t'] for r in records]
            guided_times = [r['guided_t'] for r in records]
            all_exact_times.extend(exact_times)
            all_guided_times.extend(guided_times)
            
            total_e_time = sum(exact_times)
            total_g_time = sum(guided_times)
            seed_speedup = (total_e_time / total_g_time) if total_g_time > 0 else 0
            
            gaps = [ (g - e)/e * 100.0 if e > 0 else 0 for e, g in zip(exact_list, guided_list) ]
            avg_gap = mean(gaps)
            max_g = max_val(gaps)
            all_gaps.extend(gaps)
            
            exact_d = diffs(exact_list)
            guided_d = diffs(guided_list)
            
            if len(exact_d) > 0:
                mono_e = sum(1.0 for d in exact_d if d >= 0) / len(exact_d)
                mono_g = sum(1.0 for d in guided_d if d >= 0) / len(guided_d)
                
                mono_exact_count += sum(1 for d in exact_d if d >= 0)
                mono_guided_count += sum(1 for d in guided_d if d >= 0)
                mono_total += len(exact_d)
            else:
                mono_e = 1.0
                mono_g = 1.0
                
            if len(exact_list) > 1 and std_dev(exact_list) > 0 and std_dev(guided_list) > 0:
                corr = corrcoef(exact_list, guided_list)
                correlations.append(corr)
            else:
                corr = float('nan')
                
            print(f"  Seed {seed:6d} | Steps: {len(records):2d} | Avg Gap: {avg_gap:5.2f}% (Max: {max_g:5.2f}%) | "
                  f"Exact Mono: {mono_e*100:5.1f}% | Guided Mono: {mono_g*100:5.1f}% | Seed Speedup: {seed_speedup:5.1f}x")
            
        print(f"\n------------------------------------------------------------")
        print(f"OVERALL SUMMARY FOR '{model.upper()}'")
        print(f"------------------------------------------------------------")
        print(f"► Total Instances              : {len(seeds_data)}")
        print(f"► Total Search Steps evaluated : {len(all_gaps)}")
        print(f"► Mean Optimality Gap          : {mean(all_gaps):.4f}%")
        print(f"► Max Optimality Gap           : {max_val(all_gaps):.4f}%")
        
        sum_e_time = sum(all_exact_times)
        sum_g_time = sum(all_guided_times)
        overall_speedup = (sum_e_time / sum_g_time) if sum_g_time > 0 else 0
        
        print(f"\n[Performance & Latency]")
        print(f"► Total Exact Solve Time       : {sum_e_time:.1f}s")
        print(f"► Total Guided Solve Time      : {sum_g_time:.1f}s")
        print(f"► OVERALL AGGREGATE SPEEDUP    : {overall_speedup:.2f}x")
        
        if mono_total > 0:
            print(f"\n[Monotonicity (Does tighter horizon = higher energy?)]")
            print(f"► Exact Energy strictly obeyed law  : {mono_exact_count / mono_total * 100:.2f}% of steps")
            print(f"► Guided Energy properly correlated : {mono_guided_count / mono_total * 100:.2f}% of steps")
        
        if correlations:
            import math
            valid_corrs = [c for c in correlations if not math.isnan(c)]
            if valid_corrs:
                print(f"► Mean Correlation (Pearson)        : {mean(valid_corrs):.4f} (1.0 is perfect tracking)")
        print("\n\n")

if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'deploy_medium.log'
    main(log_path)
