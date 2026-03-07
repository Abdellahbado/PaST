#pragma once

#include "dp_solver.hpp"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dp
{

    struct MachineStateConfig
    {
        std::vector<std::string> states;
        int off_idx = -1;
        int proc_idx = -1;
        std::vector<std::vector<int>> t_trans;
        std::vector<std::vector<double>> p_trans;

        static MachineStateConfig paper_nosby();
    };

    struct SPACESResult
    {
        std::vector<double> c_star;
        std::vector<double> c_start;
        std::vector<double> c_end;
        double p_proc = 0.0;
        int early = 0;
        int late = 0;
        int h = 0;
        int max_gap = -1;
        bool banded = false;

        double gap_cost(int t_end, int t_start) const noexcept;
    };

    struct StatefulParent
    {
        int prev_t_end = -1;
        int64_t prev_state = 0;
        int length = 0;
        int t_start = 0;
    };

    MachineStateConfig make_paper_nosby_config();
    std::vector<double> build_proc_prefix(const std::vector<double> &prices, double p_proc);
    SPACESResult compute_spaces(
        const std::vector<double> &prices,
        const MachineStateConfig &config,
        int max_gap = -1);

    DPResult solve_sparse_dp_stateful(
        const std::vector<int> &lengths,
        const std::vector<int> &totals,
        const std::vector<double> &prefix_proc,
        int T,
        const SPACESResult &spaces,
        const DPParams &params = {});

} // namespace dp