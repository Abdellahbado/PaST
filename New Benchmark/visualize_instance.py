#!/usr/bin/env python3
"""
Visualization Script for Parallel Machine Scheduling Benchmark Instances
=========================================================================

This script provides clear, informative visualizations of benchmark instances
for parallel machine scheduling with energy consideration (TOU pricing).

Usage:
    python visualize_instance.py                    # Interactive mode
    python visualize_instance.py --id 1             # Visualize instance 1
    python visualize_instance.py --id 1 5 10        # Visualize multiple instances
    python visualize_instance.py --range 1 10       # Visualize instances 1-10
    python visualize_instance.py --all              # Visualize all instances
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_instances(json_path: str = "instances_90.json") -> List[Dict[str, Any]]:
    """Load instances from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_instance_by_id(instances: List[Dict], instance_id: int) -> Optional[Dict]:
    """Get instance by its ID."""
    for inst in instances:
        if inst["instance_id"] == instance_id:
            return inst
    return None


def get_expected_ranges(instance: Dict) -> Dict[str, tuple]:
    """
    Get expected parameter ranges based on paper and scale.
    Used to verify if instance conforms to configuration.
    """
    paper = instance["paper"]
    scale = instance["scale"]
    
    if paper == "Wang2018":
        return {
            "p_range": (1, 4),
            "e_range": (1, 3),
            "ck_range": (1, 4),
            "Tk_choices": {2, 3, 5}
        }
    elif paper == "Anghinolfi2021":
        return {
            "p_range": (1, 12),
            "e_range": (1, 6),
            "ck_range": (1, 8),
            "Tk_choices": {2, 3, 5}
        }
    return {}


def verify_instance(instance: Dict) -> Dict[str, bool]:
    """Verify if instance parameters are within expected ranges."""
    expected = get_expected_ranges(instance)
    if not expected:
        return {}
    
    p_min, p_max = expected["p_range"]
    e_min, e_max = expected["e_range"]
    ck_min, ck_max = expected["ck_range"]
    Tk_choices = expected["Tk_choices"]
    
    return {
        "p_valid": all(p_min <= p <= p_max for p in instance["p"]),
        "e_valid": all(e_min <= e <= e_max for e in instance["e"]),
        "ck_valid": all(ck_min <= c <= ck_max for c in instance["ck"]),
        "Tk_valid": all(tk in Tk_choices for tk in instance["Tk"]),
        "Tk_sum_valid": sum(instance["Tk"]) == instance["T"],
        "ct_length_valid": len(instance["ct"]) == instance["T"],
        "p_length_valid": len(instance["p"]) == instance["n"],
        "e_length_valid": len(instance["e"]) == instance["m"],
    }


def create_color_palette():
    """Create a cohesive color palette for the visualization."""
    return {
        "primary": "#2563eb",       # Blue
        "secondary": "#7c3aed",     # Purple
        "accent": "#059669",        # Green
        "warning": "#d97706",       # Orange
        "danger": "#dc2626",        # Red
        "bg_light": "#f8fafc",      # Light background
        "bg_dark": "#1e293b",       # Dark background
        "text": "#334155",          # Text color
        "grid": "#e2e8f0",          # Grid color
        "price_gradient": ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"],
    }


def visualize_instance(instance: Dict, save_path: Optional[str] = None, show: bool = True):
    """
    Create a comprehensive visualization of a single instance.
    
    The visualization includes:
    1. Instance metadata and configuration summary
    2. Machine energy rates visualization
    3. Job processing times visualization
    4. Electricity price profile over time (intervals and per-period)
    5. Verification status against expected configuration
    """
    colors = create_color_palette()
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12), facecolor=colors["bg_light"])
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.8, 1, 1.2], 
                  hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel 1: Instance Metadata (Top Left)
    # =========================================================================
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.set_facecolor(colors["bg_light"])
    ax_info.axis("off")
    
    # Verification
    verification = verify_instance(instance)
    expected = get_expected_ranges(instance)
    all_valid = all(verification.values()) if verification else True
    status_color = colors["accent"] if all_valid else colors["danger"]
    status_text = "✓ VALID" if all_valid else "✗ ISSUES DETECTED"
    
    info_text = (
        f"Instance #{instance['instance_id']}\n"
        f"─────────────────────\n"
        f"Paper: {instance['paper']}\n"
        f"Scale: {instance['scale'].upper()}\n"
        f"─────────────────────\n"
        f"Machines (m): {instance['m']}\n"
        f"Jobs (n): {instance['n']}\n"
        f"Time Horizon (T): {instance['T']}\n"
        f"Price Intervals (K): {len(instance['Tk'])}\n"
        f"─────────────────────\n"
        f"Status: {status_text}"
    )
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                          edgecolor=status_color, linewidth=2))
    
    # =========================================================================
    # Panel 2: Expected Ranges (Top Center)
    # =========================================================================
    ax_ranges = fig.add_subplot(gs[0, 1])
    ax_ranges.set_facecolor(colors["bg_light"])
    ax_ranges.axis("off")
    
    if expected:
        ranges_text = (
            f"Expected Configuration\n"
            f"─────────────────────────\n"
            f"Processing times p: [{expected['p_range'][0]}, {expected['p_range'][1]}]\n"
            f"Energy rates e:     [{expected['e_range'][0]}, {expected['e_range'][1]}]\n"
            f"Interval prices ck: [{expected['ck_range'][0]}, {expected['ck_range'][1]}]\n"
            f"Interval lengths:   {sorted(expected['Tk_choices'])}\n"
            f"─────────────────────────\n"
            f"Actual Values\n"
            f"─────────────────────────\n"
            f"p: [{min(instance['p'])}, {max(instance['p'])}]\n"
            f"e: [{min(instance['e'])}, {max(instance['e'])}]\n"
            f"ck: [{min(instance['ck'])}, {max(instance['ck'])}]\n"
            f"Tk: {sorted(set(instance['Tk']))}"
        )
    else:
        ranges_text = "Configuration info not available"
    
    ax_ranges.text(0.05, 0.95, ranges_text, transform=ax_ranges.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor=colors["grid"], linewidth=1))
    
    # =========================================================================
    # Panel 3: Verification Details (Top Right)
    # =========================================================================
    ax_verify = fig.add_subplot(gs[0, 2])
    ax_verify.set_facecolor(colors["bg_light"])
    ax_verify.axis("off")
    
    verify_lines = ["Verification Checks", "─────────────────────────"]
    for key, valid in verification.items():
        symbol = "✓" if valid else "✗"
        color_indicator = "" 
        label = key.replace("_", " ").replace("valid", "").strip().title()
        verify_lines.append(f"{symbol} {label}")
    
    verify_text = "\n".join(verify_lines)
    ax_verify.text(0.05, 0.95, verify_text, transform=ax_verify.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor=colors["grid"], linewidth=1))
    
    # =========================================================================
    # Panel 4: Machine Energy Rates (Middle Left)
    # =========================================================================
    ax_machines = fig.add_subplot(gs[1, 0])
    ax_machines.set_facecolor('white')
    
    machines = list(range(1, instance['m'] + 1))
    energy_rates = instance['e']
    
    # Color bars by energy rate (gradient from green to red)
    e_min, e_max = min(energy_rates), max(energy_rates)
    if e_max > e_min:
        normalized = [(e - e_min) / (e_max - e_min) for e in energy_rates]
    else:
        normalized = [0.5] * len(energy_rates)
    
    bar_colors = [plt.cm.RdYlGn_r(n) for n in normalized]
    
    bars = ax_machines.bar(machines, energy_rates, color=bar_colors, 
                           edgecolor='white', linewidth=1.5)
    
    ax_machines.set_xlabel("Machine Index", fontsize=10, fontweight='bold')
    ax_machines.set_ylabel("Energy Rate (e)", fontsize=10, fontweight='bold')
    ax_machines.set_title("Machine Energy Consumption Rates", 
                          fontsize=12, fontweight='bold', pad=10)
    ax_machines.set_xticks(machines)
    ax_machines.set_ylim(0, max(energy_rates) * 1.2)
    ax_machines.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, energy_rates):
        ax_machines.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # =========================================================================
    # Panel 5: Job Processing Times (Middle Center and Right)
    # =========================================================================
    ax_jobs = fig.add_subplot(gs[1, 1:])
    ax_jobs.set_facecolor('white')
    
    jobs = list(range(1, instance['n'] + 1))
    proc_times = instance['p']
    
    # Color bars by processing time
    p_min, p_max = min(proc_times), max(proc_times)
    if p_max > p_min:
        normalized_p = [(p - p_min) / (p_max - p_min) for p in proc_times]
    else:
        normalized_p = [0.5] * len(proc_times)
    
    bar_colors_p = [plt.cm.Blues(0.4 + 0.5 * n) for n in normalized_p]
    
    bars_p = ax_jobs.bar(jobs, proc_times, color=bar_colors_p,
                         edgecolor='white', linewidth=0.5)
    
    ax_jobs.set_xlabel("Job Index", fontsize=10, fontweight='bold')
    ax_jobs.set_ylabel("Processing Time (p)", fontsize=10, fontweight='bold')
    ax_jobs.set_title("Job Processing Times", fontsize=12, fontweight='bold', pad=10)
    
    # Smart x-tick labeling for large number of jobs
    if instance['n'] <= 30:
        ax_jobs.set_xticks(jobs)
    else:
        step = max(1, instance['n'] // 20)
        ax_jobs.set_xticks(jobs[::step])
    
    ax_jobs.set_ylim(0, max(proc_times) * 1.2)
    ax_jobs.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistics annotation
    stats_text = f"n={instance['n']}  |  min={p_min}  |  max={p_max}  |  avg={np.mean(proc_times):.1f}  |  total={sum(proc_times)}"
    ax_jobs.text(0.5, 1.02, stats_text, transform=ax_jobs.transAxes,
                 ha='center', fontsize=9, style='italic', color=colors["text"])
    
    # =========================================================================
    # Panel 6: Electricity Price Profile (Bottom - Full Width)
    # =========================================================================
    ax_price = fig.add_subplot(gs[2, :])
    ax_price.set_facecolor('white')
    
    # Plot per-period prices as a step function
    ct = instance['ct']
    T = instance['T']
    time_points = list(range(T))
    
    # Create step plot for per-period prices
    ax_price.step(time_points, ct, where='post', color=colors["primary"], 
                  linewidth=2, label='Per-period price (ct)')
    ax_price.fill_between(time_points, ct, step='post', alpha=0.3, color=colors["primary"])
    
    # Overlay interval boundaries with different colors
    Tk = instance['Tk']
    ck = instance['ck']
    
    # Get unique prices for color mapping
    unique_prices = sorted(set(ck))
    price_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(unique_prices)))
    price_color_map = dict(zip(unique_prices, price_colors))
    
    # Draw interval rectangles
    t_start = 0
    for k, (duration, price) in enumerate(zip(Tk, ck)):
        t_end = t_start + duration
        rect = mpatches.Rectangle((t_start, 0), duration, price,
                                   linewidth=1.5, edgecolor='black',
                                   facecolor=price_color_map[price], alpha=0.4)
        ax_price.add_patch(rect)
        
        # Add interval boundary markers (vertical dashed lines)
        ax_price.axvline(x=t_start, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        
        # Add interval label and time range for small/medium instances
        if len(Tk) <= 30:
            # Interval name at top
            ax_price.text(t_start + duration/2, price + 0.15, f"k{k+1}",
                         ha='center', fontsize=7, fontweight='bold')
            # Time range annotation at bottom (inside the rectangle)
            time_label = f"[{t_end - t_start})"
            ax_price.text(t_start + duration/2, 0.1, time_label,
                         ha='center', va='bottom', fontsize=6, 
                         fontfamily='monospace', color='#1e293b',
                         bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                                  edgecolor='none', alpha=0.7))
        elif len(Tk) <= 60:
            # For medium-large instances, show time range at boundaries
            if k % 3 == 0:  # Show every 3rd interval to avoid clutter
                time_label = f"[{t_end - t_start})"
                ax_price.text(t_start + duration/2, 0.1, time_label,
                             ha='center', va='bottom', fontsize=5, 
                             fontfamily='monospace', color='#1e293b', rotation=45)
        
        t_start = t_end
    
    # Add final boundary marker
    ax_price.axvline(x=T, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    
    ax_price.set_xlabel("Time Period (t)", fontsize=11, fontweight='bold')
    ax_price.set_ylabel("Electricity Price", fontsize=11, fontweight='bold')
    ax_price.set_title("Electricity Price Profile (Time-of-Use Pricing)", 
                       fontsize=12, fontweight='bold', pad=10)
    ax_price.set_xlim(0, T)
    ax_price.set_ylim(0, max(ct) * 1.3)
    
    # Set x-axis ticks at interval boundaries (where each period starts/ends)
    interval_boundaries = [0]
    cumsum = 0
    for tk in Tk:
        cumsum += tk
        interval_boundaries.append(cumsum)
    
    # For small instances, show all boundaries; for larger ones, show subset
    if len(interval_boundaries) <= 20:
        ax_price.set_xticks(interval_boundaries)
    else:
        # Show every nth boundary to avoid clutter, always include 0 and T
        step = max(1, len(interval_boundaries) // 15)
        sparse_boundaries = [interval_boundaries[i] for i in range(0, len(interval_boundaries), step)]
        if T not in sparse_boundaries:
            sparse_boundaries.append(T)
        ax_price.set_xticks(sorted(set(sparse_boundaries)))
    
    ax_price.tick_params(axis='x', rotation=45 if len(interval_boundaries) > 15 else 0)
    ax_price.grid(axis='both', alpha=0.3, linestyle='--')
    
    # Add legend with price levels
    legend_patches = [mpatches.Patch(facecolor=price_color_map[p], 
                                     edgecolor='black', alpha=0.6,
                                     label=f'Price level {p}')
                     for p in unique_prices]
    ax_price.legend(handles=legend_patches, loc='upper right', 
                   fontsize=9, framealpha=0.9)
    
    # Add summary annotation
    price_stats = (f"K={len(Tk)} intervals  |  "
                   f"Price range: [{min(ck)}, {max(ck)}]  |  "
                   f"Interval lengths: {sorted(set(Tk))}")
    ax_price.text(0.5, 1.02, price_stats, transform=ax_price.transAxes,
                  ha='center', fontsize=9, style='italic', color=colors["text"])
    
    # =========================================================================
    # Final adjustments
    # =========================================================================
    plt.suptitle(f"Benchmark Instance Visualization", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor=colors["bg_light"])
        print(f"Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_comparison(instances: List[Dict], save_path: Optional[str] = None, show: bool = True):
    """
    Create a comparison visualization for multiple instances.
    Shows key metrics side by side for easy comparison.
    """
    if len(instances) > 6:
        print(f"Warning: Showing only first 6 instances for comparison clarity.")
        instances = instances[:6]
    
    colors = create_color_palette()
    n_inst = len(instances)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=colors["bg_light"])
    
    ids = [inst["instance_id"] for inst in instances]
    labels = [f"#{inst['instance_id']}\n({inst['scale']})" for inst in instances]
    
    # Panel 1: Number of machines and jobs
    ax1 = axes[0, 0]
    x = np.arange(n_inst)
    width = 0.35
    
    machines = [inst['m'] for inst in instances]
    jobs = [inst['n'] for inst in instances]
    
    bars1 = ax1.bar(x - width/2, machines, width, label='Machines (m)', 
                    color=colors["primary"], edgecolor='white')
    bars2 = ax1.bar(x + width/2, jobs, width, label='Jobs (n)',
                    color=colors["secondary"], edgecolor='white')
    
    ax1.set_xlabel('Instance')
    ax1.set_ylabel('Count')
    ax1.set_title('Problem Size: Machines vs Jobs', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Time horizon and number of intervals
    ax2 = axes[0, 1]
    
    horizons = [inst['T'] for inst in instances]
    intervals = [len(inst['Tk']) for inst in instances]
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.bar(x - width/2, horizons, width, label='Time Horizon (T)',
                    color=colors["accent"], edgecolor='white')
    line2 = ax2_twin.bar(x + width/2, intervals, width, label='Intervals (K)',
                         color=colors["warning"], edgecolor='white')
    
    ax2.set_xlabel('Instance')
    ax2.set_ylabel('Time Horizon (T)', color=colors["accent"])
    ax2_twin.set_ylabel('Number of Intervals (K)', color=colors["warning"])
    ax2.set_title('Temporal Structure', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Panel 3: Processing time statistics
    ax3 = axes[1, 0]
    
    p_mins = [min(inst['p']) for inst in instances]
    p_maxs = [max(inst['p']) for inst in instances]
    p_avgs = [np.mean(inst['p']) for inst in instances]
    
    ax3.bar(x - width, p_mins, width, label='Min p', color='#22c55e', edgecolor='white')
    ax3.bar(x, p_avgs, width, label='Avg p', color='#3b82f6', edgecolor='white')
    ax3.bar(x + width, p_maxs, width, label='Max p', color='#ef4444', edgecolor='white')
    
    ax3.set_xlabel('Instance')
    ax3.set_ylabel('Processing Time')
    ax3.set_title('Processing Time Distribution', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Price statistics
    ax4 = axes[1, 1]
    
    ck_mins = [min(inst['ck']) for inst in instances]
    ck_maxs = [max(inst['ck']) for inst in instances]
    ck_avgs = [np.mean(inst['ck']) for inst in instances]
    
    ax4.bar(x - width, ck_mins, width, label='Min price', color='#22c55e', edgecolor='white')
    ax4.bar(x, ck_avgs, width, label='Avg price', color='#3b82f6', edgecolor='white')
    ax4.bar(x + width, ck_maxs, width, label='Max price', color='#ef4444', edgecolor='white')
    
    ax4.set_xlabel('Instance')
    ax4.set_ylabel('Price')
    ax4.set_title('Electricity Price Distribution', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Instance Comparison', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor=colors["bg_light"])
        print(f"Saved comparison to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def list_instances(instances: List[Dict]):
    """Print a summary table of all instances."""
    print("\n" + "="*80)
    print("AVAILABLE INSTANCES")
    print("="*80)
    print(f"{'ID':>4} | {'Paper':<15} | {'Scale':<6} | {'m':>3} | {'n':>4} | {'T':>4} | {'K':>3}")
    print("-"*80)
    
    current_paper = None
    for inst in instances:
        if inst['paper'] != current_paper:
            if current_paper is not None:
                print("-"*80)
            current_paper = inst['paper']
        
        print(f"{inst['instance_id']:>4} | {inst['paper']:<15} | {inst['scale']:<6} | "
              f"{inst['m']:>3} | {inst['n']:>4} | {inst['T']:>4} | {len(inst['Tk']):>3}")
    
    print("="*80)
    print(f"Total: {len(instances)} instances")
    print()


def interactive_mode(instances: List[Dict]):
    """Run interactive visualization mode."""
    print("\n" + "="*60)
    print("PARALLEL MACHINE SCHEDULING INSTANCE VISUALIZER")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("  [number]     - Visualize instance by ID (e.g., '1' or '1,5,10')")
        print("  [range]      - Visualize range (e.g., '1-5')")
        print("  'list'       - List all available instances")
        print("  'compare'    - Compare multiple instances (enter IDs when prompted)")
        print("  'quit'       - Exit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'quit' or choice == 'q':
            print("Goodbye!")
            break
        
        elif choice == 'list':
            list_instances(instances)
        
        elif choice == 'compare':
            ids_input = input("Enter instance IDs to compare (comma-separated, max 6): ")
            try:
                ids = [int(x.strip()) for x in ids_input.split(',')]
                selected = [get_instance_by_id(instances, i) for i in ids]
                selected = [s for s in selected if s is not None]
                if selected:
                    visualize_comparison(selected)
                else:
                    print("No valid instances found.")
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers.")
        
        elif '-' in choice:
            try:
                start, end = map(int, choice.split('-'))
                for i in range(start, end + 1):
                    inst = get_instance_by_id(instances, i)
                    if inst:
                        visualize_instance(inst)
            except ValueError:
                print("Invalid range format. Use 'start-end' (e.g., '1-5')")
        
        elif ',' in choice:
            try:
                ids = [int(x.strip()) for x in choice.split(',')]
                for i in ids:
                    inst = get_instance_by_id(instances, i)
                    if inst:
                        visualize_instance(inst)
                    else:
                        print(f"Instance {i} not found.")
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers.")
        
        else:
            try:
                inst_id = int(choice)
                inst = get_instance_by_id(instances, inst_id)
                if inst:
                    visualize_instance(inst)
                else:
                    print(f"Instance {inst_id} not found.")
            except ValueError:
                print("Invalid choice. Please try again.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize parallel machine scheduling benchmark instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_instance.py                    # Interactive mode
  python visualize_instance.py --id 1             # Visualize instance 1
  python visualize_instance.py --id 1 5 10        # Visualize instances 1, 5, and 10
  python visualize_instance.py --range 1 10       # Visualize instances 1 through 10
  python visualize_instance.py --compare 1 2 3    # Compare instances 1, 2, and 3
  python visualize_instance.py --list             # List all instances
  python visualize_instance.py --save             # Save visualizations to files
        """
    )
    
    parser.add_argument("--json", type=str, default="instances_90.json",
                       help="Path to instances JSON file")
    parser.add_argument("--id", type=int, nargs="+",
                       help="Instance ID(s) to visualize")
    parser.add_argument("--range", type=int, nargs=2, metavar=("START", "END"),
                       help="Range of instance IDs to visualize")
    parser.add_argument("--compare", type=int, nargs="+",
                       help="Instance IDs to compare side by side")
    parser.add_argument("--all", action="store_true",
                       help="Visualize all instances")
    parser.add_argument("--list", action="store_true",
                       help="List all available instances")
    parser.add_argument("--save", action="store_true",
                       help="Save visualizations to files instead of displaying")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                       help="Directory to save visualizations (default: visualizations)")
    
    args = parser.parse_args()
    
    # Load instances
    try:
        instances = load_instances(args.json)
        print(f"Loaded {len(instances)} instances from {args.json}")
    except FileNotFoundError:
        print(f"Error: Could not find {args.json}")
        print("Make sure the instances file exists in the current directory.")
        return
    
    # Create output directory if saving
    if args.save:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Handle different modes
    if args.list:
        list_instances(instances)
    
    elif args.compare:
        selected = [get_instance_by_id(instances, i) for i in args.compare]
        selected = [s for s in selected if s is not None]
        if selected:
            save_path = f"{args.output_dir}/comparison.png" if args.save else None
            visualize_comparison(selected, save_path=save_path, show=not args.save)
    
    elif args.id:
        for inst_id in args.id:
            inst = get_instance_by_id(instances, inst_id)
            if inst:
                save_path = f"{args.output_dir}/instance_{inst_id:03d}.png" if args.save else None
                visualize_instance(inst, save_path=save_path, show=not args.save)
            else:
                print(f"Instance {inst_id} not found.")
    
    elif args.range:
        start, end = args.range
        for inst_id in range(start, end + 1):
            inst = get_instance_by_id(instances, inst_id)
            if inst:
                save_path = f"{args.output_dir}/instance_{inst_id:03d}.png" if args.save else None
                visualize_instance(inst, save_path=save_path, show=not args.save)
    
    elif args.all:
        for inst in instances:
            save_path = f"{args.output_dir}/instance_{inst['instance_id']:03d}.png" if args.save else None
            visualize_instance(inst, save_path=save_path, show=not args.save)
    
    else:
        # Default to interactive mode
        interactive_mode(instances)


if __name__ == "__main__":
    main()
