import sys
import re
import numpy as np

def analyze_log(log_path):
    print(f"Analyzing log: {log_path}\n")
    runs = []
    current = None
    eps_data = {}

    pending_eval_price_mode = None
    pending_eval_price_params = None

    def _is_nonempty_run(r):
        return (
            (r.get('model_path') is not None)
            or (r.get('eval_rows') and len(r.get('eval_rows')) > 0)
            or (r.get('eval_summary_overall') is not None)
            or (r.get('eval_summary_beams') and len(r.get('eval_summary_beams')) > 0)
            or (r.get('train_r2') is not None)
        )
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.rstrip("\n")

                if "[pool] === TRAINING PHASE ===" in line:
                    if current is not None and _is_nonempty_run(current):
                        runs.append(current)
                    current = {
                        'run_id': len(runs) + 1,
                        'model_type': None,
                        'model_path': None,
                        'poly_l2': None,
                        'l2': None,
                        'train_r2': None,
                        'test_r2': None,
                        'train_mae': None,
                        'test_mae': None,
                        'eval_price_mode': None,
                        'eval_price_params': None,
                        'eval_rows': [],
                        'eval_timeouts': 0,
                        'eval_summary_overall': None,
                        'eval_summary_beams': {},
                    }

                if current is None:
                    current = {
                        'run_id': 1,
                        'model_type': None,
                        'model_path': None,
                        'poly_l2': None,
                        'l2': None,
                        'train_r2': None,
                        'test_r2': None,
                        'train_mae': None,
                        'test_mae': None,
                        'eval_price_mode': None,
                        'eval_price_params': None,
                        'eval_rows': [],
                        'eval_timeouts': 0,
                        'eval_summary_overall': None,
                        'eval_summary_beams': {},
                    }

                m = re.search(r'^\[pool\] Model type:\s*(\S+)', line)
                if m:
                    current['model_type'] = m.group(1)

                m = re.search(r'R2_train=([\d.\-]+)\s+R2_test=([\d.\-]+)', line)
                if m:
                    current['train_r2'] = float(m.group(1))
                    current['test_r2'] = float(m.group(2))

                m = re.search(r'MAE_train=([\d.\-]+)\s+MAE_test=([\d.\-]+)', line)
                if m:
                    current['train_mae'] = float(m.group(1))
                    current['test_mae'] = float(m.group(2))

                m = re.search(r'^\[pool\] Model saved to (.+?)\s*\(type=(\S+)\)', line)
                if m:
                    current['model_path'] = m.group(1)
                    if current['model_type'] is None:
                        current['model_type'] = m.group(2)
                    m_l2 = re.search(r'poly_l2=([0-9.eE\-]+)', current['model_path'])
                    if m_l2:
                        try:
                            current['poly_l2'] = float(m_l2.group(1))
                        except ValueError:
                            current['poly_l2'] = m_l2.group(1)

                m = re.search(r'^\[pool\] Eval prices:\s*(\S+)\s*\((.*)$', line)
                if m:
                    pending_eval_price_mode = m.group(1)
                    pending_eval_price_params = m.group(2)
                    if ")" in pending_eval_price_params:
                        pending_eval_price_params = pending_eval_price_params.split(")", 1)[0]
                        current['eval_price_mode'] = pending_eval_price_mode
                        current['eval_price_params'] = pending_eval_price_params
                        pending_eval_price_mode = None
                        pending_eval_price_params = None
                elif pending_eval_price_params is not None:
                    pending_eval_price_params = pending_eval_price_params + line.strip()
                    if ")" in pending_eval_price_params:
                        pending_eval_price_params = pending_eval_price_params.split(")", 1)[0]
                        current['eval_price_mode'] = pending_eval_price_mode
                        current['eval_price_params'] = pending_eval_price_params
                        pending_eval_price_mode = None
                        pending_eval_price_params = None

                if "exact DP timed out" in line:
                    current['eval_timeouts'] += 1

                # Parse evaluation lines
                # example: seed=500 beam=2 exact=736.0 L=736.0 Z=865.0 P=736.0 gapL=0.00% gapZ=17.53% gapP=0.00% t_exact=25.978s tL=0.293s
                if line.startswith("seed=") and "beam=" in line:
                    m = re.search(
                        r'^seed=(\d+)\s+beam=(\d+)\s+exact=([\d.]+)\s+L=([\d.]+)\s+Z=([\d.]+)\s+P=([\d.]+)\s+'
                        r'gapL=([\d.]+)%\s+gapZ=([\d.]+)%\s+gapP=([\d.]+)%\s+'
                        r't_exact=([\d.]+)s\s+tL=([\d.]+)s',
                        line
                    )
                    if m:
                        current['eval_rows'].append({
                            'seed': int(m.group(1)),
                            'beam': int(m.group(2)),
                            'exact': float(m.group(3)),
                            'L': float(m.group(4)),
                            'Z': float(m.group(5)),
                            'P': float(m.group(6)),
                            'gapL': float(m.group(7)),
                            'gapZ': float(m.group(8)),
                            'gapP': float(m.group(9)),
                            't_exact': float(m.group(10)),
                            'tL': float(m.group(11)),
                        })

                m = re.search(r'^\[pool\] Overall:\s+gapL=\s*([\d.\-]+)%\s+gapZ=\s*([\d.\-]+)%\s+gapP=\s*([\d.\-]+)%', line)
                if m:
                    current['eval_summary_overall'] = {
                        'gapL': float(m.group(1)),
                        'gapZ': float(m.group(2)),
                        'gapP': float(m.group(3)),
                    }

                m = re.search(
                    r'^beam=\s*(\d+)\s+n=\s*(\d+)\s+gapL=\s*([\d.\-]+)%/\s*([\d.\-]+)%\s+'
                    r'gapZ=\s*([\d.\-]+)%/\s*([\d.\-]+)%\s+gapP=\s*([\d.\-]+)%/\s*([\d.\-]+)%\s+speedL=([\d.\-]+)x',
                    line
                )
                if m:
                    b = int(m.group(1))
                    current['eval_summary_beams'][b] = {
                        'n': int(m.group(2)),
                        'gapL_mean': float(m.group(3)),
                        'gapL_median': float(m.group(4)),
                        'gapZ_mean': float(m.group(5)),
                        'gapZ_median': float(m.group(6)),
                        'gapP_mean': float(m.group(7)),
                        'gapP_median': float(m.group(8)),
                        'speedL': float(m.group(9)),
                    }
                
                # Parse epsilon lines
                # example: inst_seed=185082 eps_iter=1 eps=340 min_eps=292 ... exact(E=5431.00,mk=340,t=1540.78s) guided(E=5431.00,mk=340,t=39.16s) ...
                if line.startswith("inst_seed=") and "eps_iter=" in line:
                    m_seed = re.search(r'inst_seed=(\d+)', line)
                    m_exact = re.search(r'exact\(E=([\d.]+),mk=[\d.]+,t=([\d.]+)s\)', line)
                    m_guided = re.search(r'guided\(E=([\d.]+),mk=[\d.]+,t=([\d.]+)s\)', line)
                    m_price = re.search(r'price\(E=([\d.]+),mk=[\d.]+,t=([\d.]+)s\)', line)
                    
                    if m_seed and m_exact and m_guided:
                        seed = int(m_seed.group(1))
                        if seed not in eps_data:
                            eps_data[seed] = []
                        eps_data[seed].append({
                            'eps': int(re.search(r'\seps=(\d+)', line).group(1)) if re.search(r'\seps=(\d+)', line) else None,
                            'exact_E': float(m_exact.group(1)),
                            'guided_E': float(m_guided.group(1)),
                            'time_exact': float(m_exact.group(2)),
                            'time_guided': float(m_guided.group(2)),
                            'price_E': float(m_price.group(1)) if m_price else None,
                            'time_price': float(m_price.group(2)) if m_price else None,
                        })
    except FileNotFoundError:
        print(f"File not found: {log_path}")
        return

    if current is not None and _is_nonempty_run(current):
        runs.append(current)
                    
    print("=== EVALUATION PHASE ===")
    any_eval = False
    best_run = None
    best_score = None
    for r in runs:
        if not r['eval_rows'] and r['eval_summary_overall'] is None and not r['eval_summary_beams']:
            continue
        any_eval = True
        hdr = f"Run #{r['run_id']}"
        if r.get('model_type') is not None:
            hdr += f" model={r['model_type']}"
        if r.get('poly_l2') is not None:
            hdr += f" poly_l2={r['poly_l2']}"
        if r.get('eval_price_mode') is not None:
            hdr += f" eval_price_mode={r['eval_price_mode']}"
        print(hdr)
        if r.get('eval_price_params'):
            print(f"  eval_price_params: {r['eval_price_params']}")
        if r.get('model_path'):
            print(f"  model_path: {r['model_path']}")
        if r.get('train_r2') is not None and r.get('test_r2') is not None:
            print(f"  fit: R2_train={r['train_r2']:.4f} R2_test={r['test_r2']:.4f}  MAE_train={r['train_mae']:.4f} MAE_test={r['test_mae']:.4f}")
        if r.get('eval_timeouts'):
            print(f"  exact_DP_timeouts: {r['eval_timeouts']}")

        if r['eval_summary_overall'] is not None:
            o = r['eval_summary_overall']
            print(f"  overall: gapL={o['gapL']:.2f}% gapZ={o['gapZ']:.2f}% gapP={o['gapP']:.2f}%")
        for beam in sorted(r['eval_summary_beams'].keys()):
            s = r['eval_summary_beams'][beam]
            print(
                f"  beam={beam} n={s['n']} "
                f"gapL={s['gapL_mean']:.2f}%/{s['gapL_median']:.2f}% "
                f"gapP={s['gapP_mean']:.2f}%/{s['gapP_median']:.2f}% "
                f"gapZ={s['gapZ_mean']:.2f}%/{s['gapZ_median']:.2f}% "
                f"speedL={s['speedL']:.2f}x"
            )

        if r['eval_rows']:
            gapsL = np.array([row['gapL'] for row in r['eval_rows']], dtype=np.float64)
            gapsP = np.array([row['gapP'] for row in r['eval_rows']], dtype=np.float64)
            t_exact = np.array([row['t_exact'] for row in r['eval_rows']], dtype=np.float64)
            tL = np.array([row['tL'] for row in r['eval_rows']], dtype=np.float64)
            mean_speed = np.mean(t_exact) / max(np.mean(tL), 1e-12)
            print(f"  parsed_rows: {len(r['eval_rows'])}  mean_gapL={np.mean(gapsL):.2f}%  mean_gapP={np.mean(gapsP):.2f}%  mean_speed={mean_speed:.2f}x")

        if r['eval_summary_overall'] is not None:
            score_gap = float(r['eval_summary_overall']['gapL'])
        elif r['eval_rows']:
            score_gap = float(np.mean([row['gapL'] for row in r['eval_rows']]))
        else:
            score_gap = None

        if score_gap is not None:
            score_speed = None
            if r['eval_rows']:
                t_exact = np.array([row['t_exact'] for row in r['eval_rows']], dtype=np.float64)
                tL = np.array([row['tL'] for row in r['eval_rows']], dtype=np.float64)
                score_speed = float(np.mean(t_exact) / max(np.mean(tL), 1e-12))
            if score_speed is None:
                score_speed = 0.0
            score = (score_gap, -score_speed)
            if best_score is None or score < best_score:
                best_score = score
                best_run = r

    if not any_eval:
        print("No evaluation data found.")
    elif best_run is not None:
        suffix = ""
        if best_run.get('poly_l2') is not None:
            suffix += f" poly_l2={best_run['poly_l2']}"
        if best_run.get('model_path'):
            suffix += f" model_path={best_run['model_path']}"
        if best_run.get('eval_summary_overall') is not None:
            o = best_run['eval_summary_overall']
            suffix += f" overall_gapL={o['gapL']:.2f}% overall_gapP={o['gapP']:.2f}%"
        print(f"\nBest run by lowest gapL:{suffix}")
        
    print("\n=== EPSILON CONSTRAINT PHASE ===")
    if eps_data:
        all_exact_times = []
        all_guided_times = []
        all_price_times = []
        all_gaps = []
        all_gaps_price = []
        correlations = []
        trend_matches = []
        
        for seed, data in eps_data.items():
            exact_costs = [d['exact_E'] for d in data]
            guided_costs = [d['guided_E'] for d in data]
            price_costs = [d['price_E'] for d in data if d.get('price_E') is not None]
            exact_times = [d['time_exact'] for d in data]
            guided_times = [d['time_guided'] for d in data]
            price_times = [d['time_price'] for d in data if d.get('time_price') is not None]
            
            all_exact_times.extend(exact_times)
            all_guided_times.extend(guided_times)
            all_price_times.extend(price_times)
            
            gaps = [ (g - e)/e * 100 if e > 0 else 0 for e, g in zip(exact_costs, guided_costs) ]
            all_gaps.extend(gaps)

            if len(price_costs) == len(exact_costs) and len(price_costs) > 0:
                gaps_p = [ (p - e)/e * 100 if e > 0 else 0 for e, p in zip(exact_costs, price_costs) ]
                all_gaps_price.extend(gaps_p)
            
            # Correlation
            if len(exact_costs) > 1:
                # Need variance to compute correlation
                if np.std(exact_costs) > 0 and np.std(guided_costs) > 0:
                    corr = np.corrcoef(exact_costs, guided_costs)[0, 1]
                    correlations.append(corr)
                
                # Check trend direction match
                # if exact is decreasing, predicted should be decreasing
                matches = 0
                total_transitions = 0
                for i in range(1, len(exact_costs)):
                    diff_e = exact_costs[i] - exact_costs[i-1]
                    diff_g = guided_costs[i] - guided_costs[i-1]
                    
                    if diff_e != 0:
                        total_transitions += 1
                        if np.sign(diff_e) == np.sign(diff_g):
                            matches += 1
                if total_transitions > 0:
                    trend_matches.append(matches / total_transitions)

        print(f"Total unique seeds parsed: {len(eps_data)}")
        print(f"Total epsilon steps parsed: {len(all_exact_times)}")
        print(f"Average Gap (Guided vs Exact): {np.mean(all_gaps):.2f}%")
        print(f"Max Gap (Guided vs Exact): {np.max(all_gaps):.2f}%")
        if all_gaps_price:
            print(f"Average Gap (Price vs Exact): {np.mean(all_gaps_price):.2f}%")
            print(f"Max Gap (Price vs Exact): {np.max(all_gaps_price):.2f}%")
        print(f"Average Exact Time: {np.mean(all_exact_times):.2f}s")
        print(f"Average Guided Time: {np.mean(all_guided_times):.2f}s")
        if np.mean(all_guided_times) > 0:
            print(f"Average Speedup: {np.mean(all_exact_times) / np.mean(all_guided_times):.2f}x")
        if all_price_times:
            print(f"Average Price Time: {np.mean(all_price_times):.2f}s")

        if correlations:
            print(
                f"\nAverage Pearson Correlation (Exact Cost vs Guided Cost): {np.mean(correlations):.4f}"
            )
            print(f"Min Pearson Correlation recorded: {np.min(correlations):.4f}")
        if trend_matches:
            print(
                f"Average Trend Match Frequency (same direction of change): {np.mean(trend_matches)*100:.1f}%"
            )

    else:
        print("No epsilon data found.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_log(sys.argv[1])
    else:
        print("Usage: python analyze_log.py <path_to_log>")
