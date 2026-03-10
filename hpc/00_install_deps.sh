#!/usr/bin/env bash
###############################################################################
# 00_install_deps.sh — Install all dependencies for HPC benchmarking
#
# Run this FIRST on the HPC node. Adapt module names to your cluster.
#
# Usage:  bash hpc/00_install_deps.sh
###############################################################################
set -euo pipefail

echo "============================================================"
echo " PaST HPC Benchmark — Dependency Installation"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# --------------------------------------------------------------------------
# 1. Detect environment
# --------------------------------------------------------------------------
echo ""
echo "--- System Information ---"
uname -a
echo "CPU: $(lscpu 2>/dev/null | grep 'Model name' | head -1 || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
echo "Cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"
echo "RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB\n", $1/1073741824}' || echo 'unknown')"
echo ""

# --------------------------------------------------------------------------
# 2. HPC module loads (ADAPT THESE TO YOUR CLUSTER)
# --------------------------------------------------------------------------
# Uncomment and modify for your HPC scheduler:
#
# --- SLURM-based clusters (most common) ---
# module purge
# module load gcc/12.2.0         # or gcc/13.x
# module load cmake/3.26.3
# module load python/3.11
# module load dotnet/8.0         # for paper's C# solver
#
# --- PBS/Torque clusters ---
# module load gcc cmake python dotnet-sdk
#
# --- If no module system (bare metal / VM) ---
# Dependencies will be checked below and install instructions provided.

echo "--- Checking dependencies ---"
echo ""

MISSING=()

# --------------------------------------------------------------------------
# 3. C++ toolchain (for our solver)
# --------------------------------------------------------------------------
echo -n "C++ compiler (g++ or clang++): "
# Also check conda-prefixed compiler (gxx_linux-64 installs as x86_64-conda-linux-gnu-g++)
_CONDA_GXX="${CONDA_PREFIX:-}/bin/x86_64-conda-linux-gnu-g++"
if command -v g++ &>/dev/null; then
    echo "$(g++ --version | head -1)"
elif command -v clang++ &>/dev/null; then
    echo "$(clang++ --version | head -1)"
elif [[ -x "$_CONDA_GXX" ]]; then
    echo "$(${_CONDA_GXX} --version | head -1)  [conda, creating symlink g++]"
    ln -sf "$_CONDA_GXX" "${CONDA_PREFIX}/bin/g++"
    ln -sf "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" "${CONDA_PREFIX}/bin/gcc" 2>/dev/null || true
else
    echo "NOT FOUND"
    MISSING+=("C++ compiler (g++ >= 10 or clang++ >= 13)")
fi

echo -n "CMake: "
if command -v cmake &>/dev/null; then
    echo "$(cmake --version | head -1)"
else
    echo "NOT FOUND"
    MISSING+=("cmake >= 3.16")
fi

echo -n "Make: "
if command -v make &>/dev/null; then
    echo "$(make --version | head -1)"
else
    echo "NOT FOUND"
    MISSING+=("make")
fi

# --------------------------------------------------------------------------
# 4. Python (for orchestration & analysis)
# --------------------------------------------------------------------------
echo -n "Python 3: "
if command -v python3 &>/dev/null; then
    echo "$(python3 --version)"
else
    echo "NOT FOUND"
    MISSING+=("python3 >= 3.8")
fi

# --------------------------------------------------------------------------
# 5. .NET 8 SDK (for paper's C# solver)
# --------------------------------------------------------------------------
echo -n ".NET SDK: "
if command -v dotnet &>/dev/null; then
    echo "$(dotnet --version)"
else
    echo "NOT FOUND"
    MISSING+=("dotnet SDK 8.0")
fi

# --------------------------------------------------------------------------
# 6. Utilities
# --------------------------------------------------------------------------
echo -n "GNU time (for memory tracking): "
if command -v /usr/bin/time &>/dev/null; then
    echo "available"
elif command -v gtime &>/dev/null; then
    echo "available (gtime)"
else
    echo "NOT FOUND (optional, install via: apt install time / brew install gnu-time)"
fi

echo ""

# --------------------------------------------------------------------------
# 7. Report missing deps
# --------------------------------------------------------------------------
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "============================================================"
    echo " MISSING DEPENDENCIES — Install these before proceeding:"
    echo "============================================================"
    for dep in "${MISSING[@]}"; do
        echo "  - $dep"
    done
    echo ""
    echo "Installation commands (pick your OS):"
    echo ""
    echo "  # Ubuntu/Debian:"
    echo "  sudo apt update && sudo apt install -y build-essential cmake python3 python3-pip"
    echo "  # .NET 8:"
    echo "  wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh"
    echo "  chmod +x dotnet-install.sh && ./dotnet-install.sh --channel 8.0"
    echo "  export PATH=\$HOME/.dotnet:\$PATH"
    echo ""
    echo "  # CentOS/RHEL/Rocky:"
    echo "  sudo yum groupinstall -y 'Development Tools'"
    echo "  sudo yum install -y cmake3 python3"
    echo "  # .NET 8: same wget method as above"
    echo ""
    echo "  # macOS:"
    echo "  xcode-select --install"
    echo "  brew install cmake python3"
    echo "  brew install --cask dotnet-sdk"
    echo ""
    echo "  # HPC with module system:"
    echo "  module load gcc cmake python dotnet-sdk"
    echo ""
    exit 1
fi

echo "============================================================"
echo " All dependencies found. Ready to build."
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# 8. Install Python packages (minimal — just for analysis)
# --------------------------------------------------------------------------
echo "--- Installing Python packages for analysis ---"
pip3 install --user pandas matplotlib 2>/dev/null || python3 -m pip install --user pandas matplotlib 2>/dev/null || echo "Warning: Could not install pandas/matplotlib (optional for analysis)"

echo ""
echo "Done. Next step: bash hpc/01_build_our_solver.sh"
