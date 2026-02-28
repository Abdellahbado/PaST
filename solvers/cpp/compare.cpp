/**
 * compare.cpp — DP vs BnB comparison.
 *
 * Usage:
 *   ./compare <size> <n_instances> <seed>
 *
 *   size:        small | mls | vls
 *   n_instances: how many random instances to run
 *   seed:        base random seed (each instance gets seed+i)
 *
 * Size presets (matching Wang2018/Anghinolfi benchmarks):
 *   small  n_jobs=8   T∈{50,80}     pj~U[1,4]  ck~U[1,4]  time_limit=30s
 *   mls    n_jobs=15  T∈{100,300}   pj~U[1,4]  ck~U[1,4]  time_limit=60s
 *   vls    n_jobs=25  T∈{350,500}   pj~U[1,12] ck~U[1,8]  time_limit=120s
 *
 * Optional overrides:
 *   --n-jobs N        override jobs per instance
 *   --T T             fix horizon (0 = pick from preset, default)
 *   --time-limit SEC  override time limit
 *   --no-schedule     skip schedule tracking (faster, no Gantt output)
 *   --dp-only / --bnb-only
 *   --out-dir DIR     output directory (default: results/<size>_s<seed>)
 *
 * Outputs:
 *   <out-dir>/instance_XXXX.json   — per-instance prices + schedules
 *   <out-dir>/results.csv          — summary table
 *
 * Visualize:
 *   python3 visualize_schedules.py <out-dir>
 */

#include "dp_solver.hpp"
#include "bnb_solver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  Size presets
// ─────────────────────────────────────────────────────────────────────────────
struct SizePreset {
    std::string      name;
    int              default_n_jobs;
    std::vector<int> T_choices;
    int              p_min, p_max;    // job processing times U[p_min,p_max]
    int              ck_min, ck_max;  // interval prices U[ck_min,ck_max]
    std::vector<int> Tk_choices;      // interval duration choices
    double           time_limit;      // seconds per solver
};

static const SizePreset PRESETS[] = {
    // name   n_jobs  T_choices        p       ck       Tk      limit
    {"small",    8,  {50, 80},        1,  4,  1,  4,  {2,3,5},  30.0},
    {"mls",     15,  {100, 300},      1,  4,  1,  4,  {2,3,5},  60.0},
    {"vls",     25,  {350, 500},      1, 12,  1,  8,  {2,3,5}, 120.0},
};

static const SizePreset& get_preset(const std::string& name) {
    for (auto& p : PRESETS) if (p.name == name) return p;
    std::cerr << "Unknown size: '" << name
              << "'.  Choose: small | mls | vls\n";
    std::exit(1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Piecewise-constant price generation (mirrors generate_data.py)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<double> make_prices(
    int T, const std::vector<int>& Tk_choices,
    int ck_min, int ck_max, std::mt19937_64& rng)
{
    std::uniform_int_distribution<int> ck_dist(ck_min, ck_max);
    std::vector<double> prices;
    int remaining = T;
    while (remaining > 0) {
        std::vector<int> ok;
        for (int x : Tk_choices) if (x <= remaining) ok.push_back(x);
        if (ok.empty()) { prices.clear(); remaining = T; continue; }
        int dur = ok[std::uniform_int_distribution<int>(0,(int)ok.size()-1)(rng)];
        double price = ck_dist(rng);
        for (int i = 0; i < dur; ++i) prices.push_back(price);
        remaining -= dur;
    }
    return prices;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Instance
// ─────────────────────────────────────────────────────────────────────────────
struct Instance {
    int    id, n_jobs, T;
    uint64_t seed;
    std::string size_name;
    std::vector<int>    proc_times;
    std::vector<double> prices, prefix;
};

static Instance gen(int id, int n_jobs, const SizePreset& pr,
                    int T_override, uint64_t seed) {
    std::mt19937_64 rng(seed);
    int T = T_override > 0 ? T_override
          : pr.T_choices[std::uniform_int_distribution<int>(
                0, (int)pr.T_choices.size()-1)(rng)];
    auto prices = make_prices(T, pr.Tk_choices, pr.ck_min, pr.ck_max, rng);
    std::vector<int> pts(n_jobs);
    std::uniform_int_distribution<int> pd(pr.p_min, pr.p_max);
    for (int& p : pts) p = pd(rng);
    std::vector<double> pfx(T+1); pfx[0]=0;
    for (int t=0;t<T;++t) pfx[t+1]=pfx[t]+prices[t];
    return {id, n_jobs, T, seed, pr.name, pts, prices, pfx};
}

// ─────────────────────────────────────────────────────────────────────────────
//  Result
// ─────────────────────────────────────────────────────────────────────────────
struct Result {
    Instance inst;
    double dp_cost=dp::kInf, dp_ms=0; bool dp_tle=false, dp_run=false;
    std::vector<dp::Segment> dp_segs;
    double bnb_cost=bnb::kInf, bnb_ms=0; bool bnb_tle=false, bnb_run=false;
    int bnb_nodes=0;
    std::vector<int> bnb_seq, bnb_starts;
    bool match=false;
};

static double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-t0).count();
}

static Result run(int id, int n_jobs, const SizePreset& pr, int T_override,
                  uint64_t seed, double tlimit, bool do_dp, bool do_bnb,
                  bool save_sched)
{
    Result r; r.inst = gen(id, n_jobs, pr, T_override, seed);
    auto& d = r.inst;

    // DP
    if (do_dp) {
        std::map<int,int> cnt; for (int p:d.proc_times) cnt[p]++;
        std::vector<int> lens,tots;
        for (auto& [p,c]:cnt){lens.push_back(p);tots.push_back(c);}
        dp::DPParams dp_p;
        dp_p.time_limit=tlimit; dp_p.track_schedule=save_sched;
        dp_p.early_tie_break=true;
        auto t0=std::chrono::steady_clock::now();
        auto res=dp::solve_sparse_dp(lens,tots,d.prefix,d.T,dp_p);
        r.dp_ms=elapsed_ms(t0); r.dp_tle=res.timed_out; r.dp_run=true;
        r.dp_cost=res.feasible?res.cost:dp::kInf;
        if (save_sched) r.dp_segs=res.segments;
    }

    // BnB
    if (do_bnb) {
        bnb::Instance bi; bi.n_jobs=n_jobs;
        bi.processing_times=d.proc_times; bi.T=d.T; bi.energy_costs=d.prices;
        bnb::BnBParams bp; bp.time_limit=tlimit;
        auto t0=std::chrono::steady_clock::now();
        auto res=bnb::solve_bnb(bi,bp);
        r.bnb_ms=elapsed_ms(t0); r.bnb_tle=res.timed_out; r.bnb_run=true;
        r.bnb_cost=res.cost; r.bnb_nodes=res.nodes;
        if (save_sched){r.bnb_seq=res.sequence;r.bnb_starts=res.starts;}
    }

    if (r.dp_run&&r.bnb_run&&r.dp_cost<dp::kInf&&r.bnb_cost<bnb::kInf
        &&!r.dp_tle&&!r.bnb_tle)
        r.match=std::fabs(r.dp_cost-r.bnb_cost)<=1e-4;
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
//  JSON / CSV output
// ─────────────────────────────────────────────────────────────────────────────
static std::string dbl(double v, int p=4){
    if(v>=1e299)return "\"inf\"";
    std::ostringstream s;s<<std::fixed<<std::setprecision(p)<<v;return s.str();
}

static void write_json(const std::string& path, const Result& r){
    std::ofstream f(path); if(!f){std::cerr<<"Cannot write "<<path<<"\n";return;}
    auto& d=r.inst;
    f<<"{\n  \"instance_id\":"<<d.id<<",\n  \"size\":\""<<d.size_name
      <<"\",\n  \"n_jobs\":"<<d.n_jobs<<",\n  \"T\":"<<d.T
      <<",\n  \"seed\":"<<d.seed<<",\n";
    f<<"  \"proc_times\":[";
    for(int i=0;i<(int)d.proc_times.size();++i)f<<(i?",":"")<<d.proc_times[i];
    f<<"],\n  \"prices\":[";
    for(int i=0;i<(int)d.prices.size();++i)f<<(i?",":"")<<dbl(d.prices[i]);
    f<<"],\n";
    // DP
    f<<"  \"dp\":{\"cost\":"<<dbl(r.dp_cost,6)<<",\"time_ms\":"
      <<dbl(r.dp_ms,2)<<",\"timed_out\":"<<(r.dp_tle?"true":"false")
      <<",\"segments\":[";
    for(int i=0;i<(int)r.dp_segs.size();++i){
        auto&s=r.dp_segs[i]; f<<(i?",":"")<<"["<<s.start<<","<<s.length<<"]";
    }
    f<<"]},\n";
    // BnB
    f<<"  \"bnb\":{\"cost\":"<<dbl(r.bnb_cost,6)<<",\"time_ms\":"
      <<dbl(r.bnb_ms,2)<<",\"timed_out\":"<<(r.bnb_tle?"true":"false")
      <<",\"nodes\":"<<r.bnb_nodes<<",\"sequence\":[";
    for(int i=0;i<(int)r.bnb_seq.size();++i)f<<(i?",":"")<<r.bnb_seq[i];
    f<<"],\"starts\":[";
    for(int i=0;i<(int)r.bnb_starts.size();++i)f<<(i?",":"")<<r.bnb_starts[i];
    f<<"]},\n";
    f<<"  \"cost_match\":"<<(r.match?"true":"false")<<"\n}\n";
}

static void write_csv(const std::string& path, const std::vector<Result>& rv){
    std::ofstream f(path); if(!f)return;
    f<<"id,size,n_jobs,T,seed,dp_cost,dp_ms,dp_tle,bnb_cost,bnb_ms,bnb_nodes,bnb_tle,match\n";
    for(auto&r:rv){
        auto b=[](bool v){return v?"1":"0";};
        f<<r.inst.id<<","<<r.inst.size_name<<","<<r.inst.n_jobs<<","
          <<r.inst.T<<","<<r.inst.seed<<","
          <<dbl(r.dp_cost,4)<<","<<std::fixed<<std::setprecision(2)<<r.dp_ms<<","
          <<b(r.dp_tle)<<","
          <<dbl(r.bnb_cost,4)<<","<<r.bnb_ms<<","
          <<r.bnb_nodes<<","<<b(r.bnb_tle)<<","<<b(r.match)<<"\n";
    }
}

static void mkdirs(const std::string& path){
    std::system(("mkdir -p \""+path+"\"").c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Arg helpers
// ─────────────────────────────────────────────────────────────────────────────
static bool has(int argc,char**argv,const char*f){
    for(int i=1;i<argc;++i)if(!std::strcmp(argv[i],f))return true;return false;}
static std::string garg(int argc,char**argv,const char*f,const char*d){
    for(int i=1;i+1<argc;++i)if(!std::strcmp(argv[i],f))return argv[i+1];return d;}
static int    gi(int argc,char**argv,const char*f,int    d){return std::stoi(garg(argc,argv,f,std::to_string(d).c_str()));}
static double gd(int argc,char**argv,const char*f,double d){return std::stod(garg(argc,argv,f,std::to_string(d).c_str()));}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv){

    // ── Help ──────────────────────────────────────────────────────────────
    if (argc < 4 || has(argc,argv,"--help") || has(argc,argv,"-h")) {
        std::cout <<
R"(Usage:
  compare <size> <n_instances> <seed>   [OPTIONS]

  size         : small | mls | vls
  n_instances  : number of random instances to run
  seed         : base random seed

Size defaults:
  small  → n_jobs=8   T∈{50,80}    pj~U[1,4]   ck~U[1,4]   limit=30s
  mls    → n_jobs=15  T∈{100,300}  pj~U[1,4]   ck~U[1,4]   limit=60s
  vls    → n_jobs=25  T∈{350,500}  pj~U[1,12]  ck~U[1,8]   limit=120s

Options:
  --n-jobs N         override jobs per instance
  --T T              fix horizon (0 = pick from preset)
  --time-limit SEC   override per-solver time limit
  --threads N        worker threads (default: all CPU cores)
  --dp-only          run DP only
  --bnb-only         run BnB only
  --no-schedule      skip schedule tracking (no JSON/Gantt output)
  --out-dir DIR      output directory

Outputs (default on):
  <out-dir>/instance_XXXX.json   prices + schedules per instance
  <out-dir>/results.csv          summary table

Visualize:
  python3 visualize_schedules.py <out-dir>

Examples:
  compare small 10 42
  compare mls   5  0  --n-jobs 20
  compare vls   3  1  --time-limit 300
)";
        return 0;
    }

    // ── Parse positional args ─────────────────────────────────────────────
    std::string size_name = argv[1];
    int    n_inst    = std::stoi(argv[2]);
    uint64_t seed    = (uint64_t)std::stoul(argv[3]);

    const SizePreset& pr = get_preset(size_name);

    // Optional overrides
    int    n_jobs    = gi(argc,argv,"--n-jobs",    pr.default_n_jobs);
    int    T_ovr     = gi(argc,argv,"--T",         0);
    double tlimit    = gd(argc,argv,"--time-limit", pr.time_limit);
    int    n_threads = gi(argc,argv,"--threads",
                         (int)std::thread::hardware_concurrency());
    if (n_threads < 1) n_threads = 1;
    bool   dp_only   = has(argc,argv,"--dp-only");
    bool   bnb_only  = has(argc,argv,"--bnb-only");
    bool   no_sched  = has(argc,argv,"--no-schedule");
    bool   save_sch  = !no_sched;

    // Default output directory
    std::ostringstream def_dir;
    def_dir << "results/" << size_name << "_s" << seed;
    std::string out_dir = garg(argc,argv,"--out-dir", def_dir.str().c_str());

    bool do_dp  = !bnb_only;
    bool do_bnb = !dp_only;

    if (save_sch) mkdirs(out_dir);

    // ── Header ────────────────────────────────────────────────────────────
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << "  DP vs BnB  [" << size_name << "]"
              << "  n_jobs=" << n_jobs
              << "  instances=" << n_inst
              << "  seed=" << seed
              << "  limit=" << tlimit << "s"
              << "  threads=" << n_threads << "\n";
    std::cout << "════════════════════════════════════════════════\n\n";

    // ── Parallel execution ────────────────────────────────────────────────
    // Pre-allocate: each thread writes to its own index (no races).
    std::vector<Result> all(n_inst);
    std::atomic<int> next_id{0};
    std::mutex print_mtx;  // guard stdout during live progress dots

    auto worker = [&]() {
        while (true) {
            int i = next_id.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_inst) break;
            uint64_t iseed = seed + (uint64_t)i;
            all[i] = run(i, n_jobs, pr, T_ovr, iseed, tlimit,
                         do_dp, do_bnb, save_sch);
            // Live dot so user sees progress
            if (n_threads > 1) {
                std::lock_guard<std::mutex> lk(print_mtx);
                std::cout << "." << std::flush;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) threads.emplace_back(worker);
    for (auto& t : threads) t.join();
    if (n_threads > 1) std::cout << "\n\n";

    // ── Print results in order ────────────────────────────────────────────
    int matches=0, compared=0;
    double dp_total=0, bnb_total=0;

    for (int i = 0; i < n_inst; ++i) {
        const auto& r = all[i];

        std::cout << "  #" << std::setw(3) << i
                  << "  T=" << std::setw(4) << r.inst.T;
        if (do_dp) {
            std::cout << "  DP=" << std::setw(9) << std::fixed << std::setprecision(4)
                      << r.dp_cost << " (" << std::setw(6) << std::setprecision(1)
                      << r.dp_ms << "ms)";
            if (r.dp_tle) std::cout << "[TLE]";
        }
        if (do_bnb) {
            std::cout << "  BnB=" << std::setw(9) << std::fixed << std::setprecision(4)
                      << r.bnb_cost << " (" << std::setw(6) << std::setprecision(1)
                      << r.bnb_ms << "ms  n" << r.bnb_nodes << ")";
            if (r.bnb_tle) std::cout << "[TLE]";
        }
        if (do_dp && do_bnb) {
            if (r.dp_tle || r.bnb_tle) std::cout << "  -";
            else std::cout << "  " << (r.match ? "✓" : "✗!!!");
        }
        std::cout << "\n";

        if (save_sch) {
            std::ostringstream jp;
            jp << out_dir << "/instance_" << std::setw(4) << std::setfill('0')
               << i << ".json";
            write_json(jp.str(), r);
        }

        if (do_dp)  dp_total  += r.dp_ms;
        if (do_bnb) bnb_total += r.bnb_ms;
        if (!r.dp_tle && !r.bnb_tle && r.dp_run && r.bnb_run)
            { ++compared; if (r.match) ++matches; }
    }

    // ── Summary ───────────────────────────────────────────────────────────
    std::cout << "\n────────────────── SUMMARY ──────────────────\n";
    if (do_dp)  std::cout << "  DP  avg : " << std::fixed << std::setprecision(1)
                          << dp_total/n_inst  << " ms\n";
    if (do_bnb) std::cout << "  BnB avg : " << std::fixed << std::setprecision(1)
                          << bnb_total/n_inst << " ms\n";
    if (do_dp && do_bnb && compared>0){
        std::cout << "  Match   : " << matches << "/" << compared;
        std::cout << (matches==compared ? "  ✓ ALL MATCH" : "  ✗ MISMATCH") << "\n";
    }
    if (save_sch) {
        write_csv(out_dir+"/results.csv", all);
        std::cout << "  Results : " << out_dir << "/\n";
        std::cout << "  Gantt   : python3 visualize_schedules.py " << out_dir << "\n";
    }
    std::cout << "─────────────────────────────────────────────\n";

    for (auto& r:all)
        if (r.dp_run&&r.bnb_run&&!r.dp_tle&&!r.bnb_tle&&!r.match){
            std::cerr<<"MISMATCH #"<<r.inst.id
                     <<" dp="<<r.dp_cost<<" bnb="<<r.bnb_cost<<"\n";
            return 1;
        }
    return 0;
}
