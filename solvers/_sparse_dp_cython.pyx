# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython-accelerated sparse DP with **native C hash maps**.

Memory-efficient replacement for Python dict-based DP layers.
Each DP state entry uses ~46 bytes effective (32-byte slot at 70% load)
versus ~260 bytes with Python dicts, yielding ~5-6x memory reduction.

The parent map uses ~17 bytes/entry (12-byte slot at 70% load) versus
~106 bytes with Python dicts, yielding ~6x memory reduction.

Return interface identical to solve_sparse_dp_python():
    (best_cost, best_finish_time, parent_dict, timed_out, best_partial)
"""

import time as time_module
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport fabs

cnp.import_array()

cdef double _EPS = 1e-12
cdef double _C_INF = 1e300
cdef long long _EMPTY = -1   # sentinel for empty hash-map slots


# ===================================================================
#  StateVal — 24 bytes per value (cost + pen + rw + jd)
# ===================================================================
cdef struct StateVal:
    double cost       # 8
    long long pen     # 8  (int64 for large tie-break sums)
    int rw            # 4  (remaining work)
    int jd            # 4  (jobs done)


# ===================================================================
#  CStateMap — open-addressing int64 → StateVal
# ===================================================================
cdef struct CStateMap:
    long long* keys       # capacity slots; _EMPTY = empty
    StateVal* vals        # parallel values
    Py_ssize_t capacity   # always power of 2
    Py_ssize_t mask       # capacity − 1
    Py_ssize_t size       # number of live entries


cdef inline Py_ssize_t _hash64(long long key, Py_ssize_t mask) noexcept nogil:
    """Multiplicative fibonacci hash for int64 keys."""
    cdef unsigned long long h = <unsigned long long>key
    h = h * <unsigned long long>14695981039346656037ULL
    h ^= (h >> 32)
    h ^= (h >> 16)
    return <Py_ssize_t>(h & <unsigned long long>mask)


cdef CStateMap* smap_create(Py_ssize_t initial_cap) noexcept:
    cdef Py_ssize_t cap = 16
    while cap < initial_cap:
        cap <<= 1
    cdef CStateMap* m = <CStateMap*>malloc(sizeof(CStateMap))
    if m == NULL:
        return NULL
    m.keys = <long long*>malloc(cap * sizeof(long long))
    m.vals = <StateVal*>malloc(cap * sizeof(StateVal))
    if m.keys == NULL or m.vals == NULL:
        free(m.keys); free(m.vals); free(m)
        return NULL
    memset(m.keys, 0xFF, cap * sizeof(long long))  # fill −1
    m.capacity = cap
    m.mask = cap - 1
    m.size = 0
    return m


cdef void smap_destroy(CStateMap* m) noexcept nogil:
    if m != NULL:
        free(m.keys)
        free(m.vals)
        free(m)


cdef void _smap_grow(CStateMap* m) except *:
    """Double capacity and rehash.  Raises MemoryError on allocation failure."""
    cdef Py_ssize_t old_cap = m.capacity
    cdef long long* old_keys = m.keys
    cdef StateVal*  old_vals = m.vals

    cdef Py_ssize_t new_cap = old_cap << 1
    cdef Py_ssize_t new_mask = new_cap - 1

    cdef long long* new_keys = <long long*>malloc(new_cap * sizeof(long long))
    cdef StateVal*  new_vals = <StateVal*>malloc(new_cap * sizeof(StateVal))
    if new_keys == NULL or new_vals == NULL:
        # Allocation failed — free any partial alloc, keep old arrays.
        if new_keys != NULL:
            free(new_keys)
        if new_vals != NULL:
            free(new_vals)
        raise MemoryError(
            f"CStateMap: cannot grow to capacity {new_cap} "
            f"({new_cap * (sizeof(long long) + sizeof(StateVal)) // 1048576} MB)"
        )
    m.keys = new_keys
    m.vals = new_vals
    memset(m.keys, 0xFF, new_cap * sizeof(long long))
    m.capacity = new_cap
    m.mask = new_mask
    m.size = 0

    cdef Py_ssize_t i, idx
    for i in range(old_cap):
        if old_keys[i] != _EMPTY:
            idx = _hash64(old_keys[i], new_mask)
            while m.keys[idx] != _EMPTY:
                idx = (idx + 1) & new_mask
            m.keys[idx] = old_keys[i]
            m.vals[idx] = old_vals[i]
            m.size += 1
    free(old_keys)
    free(old_vals)


cdef inline Py_ssize_t smap_lookup(CStateMap* m, long long key) noexcept nogil:
    """Return slot index or −1 if not found."""
    cdef Py_ssize_t idx = _hash64(key, m.mask)
    while True:
        if m.keys[idx] == key:
            return idx
        if m.keys[idx] == _EMPTY:
            return -1
        idx = (idx + 1) & m.mask


cdef inline Py_ssize_t smap_put(CStateMap* m, long long key, StateVal val) except -2:
    """Insert new key with value.  Grows if needed.  Returns slot."""
    if m.size * 10 > m.capacity * 7:   # load > 70%
        _smap_grow(m)  # may raise MemoryError
    cdef Py_ssize_t idx = _hash64(key, m.mask)
    while True:
        if m.keys[idx] == _EMPTY:
            m.keys[idx] = key
            m.vals[idx] = val
            m.size += 1
            return idx
        if m.keys[idx] == key:
            return idx        # already exists — caller decides
        idx = (idx + 1) & m.mask


# ===================================================================
#  CParentMap — open-addressing  int64 → int  (job length)
# ===================================================================
cdef struct CParentMap:
    long long* keys
    int* vals
    Py_ssize_t capacity
    Py_ssize_t mask
    Py_ssize_t size


cdef CParentMap* pmap_create(Py_ssize_t initial_cap) noexcept:
    cdef Py_ssize_t cap = 16
    while cap < initial_cap:
        cap <<= 1
    cdef CParentMap* m = <CParentMap*>malloc(sizeof(CParentMap))
    if m == NULL:
        return NULL
    m.keys = <long long*>malloc(cap * sizeof(long long))
    m.vals = <int*>malloc(cap * sizeof(int))
    if m.keys == NULL or m.vals == NULL:
        free(m.keys); free(m.vals); free(m)
        return NULL
    memset(m.keys, 0xFF, cap * sizeof(long long))
    m.capacity = cap
    m.mask = cap - 1
    m.size = 0
    return m


cdef void pmap_destroy(CParentMap* m) noexcept nogil:
    if m != NULL:
        free(m.keys)
        free(m.vals)
        free(m)


cdef void _pmap_grow(CParentMap* m) except *:
    cdef Py_ssize_t old_cap = m.capacity
    cdef long long* old_keys = m.keys
    cdef int*       old_vals = m.vals
    cdef Py_ssize_t new_cap = old_cap << 1
    cdef Py_ssize_t new_mask = new_cap - 1

    cdef long long* new_keys = <long long*>malloc(new_cap * sizeof(long long))
    cdef int*       new_vals = <int*>malloc(new_cap * sizeof(int))
    if new_keys == NULL or new_vals == NULL:
        if new_keys != NULL:
            free(new_keys)
        if new_vals != NULL:
            free(new_vals)
        raise MemoryError(
            f"CParentMap: cannot grow to capacity {new_cap} "
            f"({new_cap * (sizeof(long long) + sizeof(int)) // 1048576} MB)"
        )
    m.keys = new_keys
    m.vals = new_vals
    memset(m.keys, 0xFF, new_cap * sizeof(long long))
    m.capacity = new_cap
    m.mask = new_mask
    m.size = 0

    cdef Py_ssize_t i, idx
    for i in range(old_cap):
        if old_keys[i] != _EMPTY:
            idx = _hash64(old_keys[i], new_mask)
            while m.keys[idx] != _EMPTY:
                idx = (idx + 1) & new_mask
            m.keys[idx] = old_keys[i]
            m.vals[idx] = old_vals[i]
            m.size += 1
    free(old_keys)
    free(old_vals)


cdef inline void pmap_set(CParentMap* m, long long key, int val) except *:
    """Insert or overwrite."""
    if m.size * 10 > m.capacity * 7:
        _pmap_grow(m)  # may raise MemoryError
    cdef Py_ssize_t idx = _hash64(key, m.mask)
    while True:
        if m.keys[idx] == _EMPTY:
            m.keys[idx] = key
            m.vals[idx] = val
            m.size += 1
            return
        if m.keys[idx] == key:
            m.vals[idx] = val
            return
        idx = (idx + 1) & m.mask


cdef dict pmap_to_pydict(CParentMap* m):
    """Convert native parent map → Python dict for the return value."""
    cdef dict out = {}
    cdef Py_ssize_t i
    for i in range(m.capacity):
        if m.keys[i] != _EMPTY:
            out[m.keys[i]] = m.vals[i]
    return out


# ===================================================================
#  Main DP function — native storage, same interface as Python version
# ===================================================================
def solve_sparse_dp_cython(
    cnp.ndarray[cnp.int64_t, ndim=1] lengths,
    cnp.ndarray[cnp.int64_t, ndim=1] totals,
    cnp.ndarray[cnp.float64_t, ndim=1] prefix,
    int T,
    cnp.ndarray[cnp.int64_t, ndim=1] radices,
    cnp.ndarray[cnp.int64_t, ndim=1] mult,
    int K,
    long long final_state,
    double time_limit = -1.0,
    str tie_break = "early",
    bint track_schedule = True,
    long long max_states = 0,
    double known_upper_bound = -1.0,
):
    """
    Cython sparse DP with C hash maps.  Drop-in for solve_sparse_dp_python.

    Returns
    -------
    (best_cost, best_finish_time, parent_dict, timed_out, best_partial)
    """
    cdef double start_time = time_module.perf_counter()

    # -- Local typed copies (K ≤ 12) -----------------------------------
    cdef int[12] c_len, c_tot, c_rad
    cdef long long[12] c_inc
    cdef int i, max_job_len = 0, total_rw = 0

    for i in range(K):
        c_len[i]  = <int>lengths[i]
        c_tot[i]  = <int>totals[i]
        c_rad[i]  = <int>radices[i]
        c_inc[i]  = mult[i]
        if c_len[i] > max_job_len:
            max_job_len = c_len[i]
        total_rw += c_tot[i] * c_len[i]

    cdef long long state_bound = final_state + 1

    # -- Block-based admissible LB (kept as NumPy — small, read-only) ---
    cdef int _LB_BLOCK = 20
    prices_arr = np.diff(prefix)
    cdef dict _lb_py = {}
    cdef int b
    for b in range(0, T + 1, _LB_BLOCK):
        if b < T:
            sp = np.sort(prices_arr[b:])
            cs = np.empty(len(sp) + 1, dtype=np.float64)
            cs[0] = 0.0
            cs[1:] = np.cumsum(sp)
            _lb_py[b] = cs
        else:
            _lb_py[b] = np.zeros(1, dtype=np.float64)

    # -- Allocate native structures -------------------------------------
    cdef CStateMap** layers = <CStateMap**>malloc((T + 1) * sizeof(CStateMap*))
    if layers == NULL:
        raise MemoryError("DP layers allocation failed")
    cdef int tt
    for tt in range(T + 1):
        layers[tt] = smap_create(16)
        if layers[tt] == NULL:
            for i in range(tt):
                smap_destroy(layers[i])
            free(layers)
            raise MemoryError("State map allocation failed")

    cdef CParentMap* c_parent = NULL
    if track_schedule:
        c_parent = pmap_create(4096)
        if c_parent == NULL:
            for tt in range(T + 1):
                smap_destroy(layers[tt])
            free(layers)
            raise MemoryError("Parent map allocation failed")

    # Seed
    cdef StateVal sv0
    sv0.cost = 0.0;  sv0.pen = 0;  sv0.rw = total_rw;  sv0.jd = 0
    smap_put(layers[0], 0, sv0)

    # -- Tracking variables ---------------------------------------------
    cdef double best_final_cost = _C_INF
    if known_upper_bound > 0:
        best_final_cost = known_upper_bound + 1e-8
    cdef long long best_final_pen = <long long>(1LL << 62)
    cdef int best_final_time = -1
    cdef bint timed_out = False

    cdef int bp_jobs = 0
    cdef double bp_cost = _C_INF
    cdef int bp_time = 0
    cdef long long bp_state = 0

    cdef double* pprefix = <double*>prefix.data
    cdef bint early = (tie_break == "early")

    # -- Loop variables -------------------------------------------------
    cdef CStateMap* layer
    cdef CStateMap* nlayer          # idle target = layers[tt+1]
    cdef CStateMap* tlayer          # job target  = layers[end]
    cdef Py_ssize_t pos, idx
    cdef long long state, ns, x
    cdef StateVal sv, nsv
    cdef int remaining, freed_t, end, L, ui, nrw, njd, rw, jd
    cdef double c0, cc, lb_val
    cdef long long p0, cp
    cdef bint better

    cdef int _lb_b, _lb_len
    cdef cnp.ndarray[cnp.float64_t, ndim=1] _lb_arr

    # Return value variables (declared outside try so cdef is valid)
    cdef double ret_cost
    cdef dict py_parent

    # ==================================================================
    #  MAIN DP LOOP
    # ==================================================================
    try:
        for tt in range(T + 1):

            # --- timeout ------------------------------------------------
            if time_limit > 0.0 and (time_module.perf_counter() - start_time) > time_limit:
                timed_out = True
                break

            layer = layers[tt]
            if layer.size == 0:
                continue

            # --- memory guardrail ---------------------------------------
            if max_states > 0 and layer.size > max_states:
                timed_out = True
                break

            # --- best-partial update ------------------------------------
            for pos in range(layer.capacity):
                if layer.keys[pos] == _EMPTY:
                    continue
                state = layer.keys[pos]
                sv = layer.vals[pos]
                if sv.jd > bp_jobs or (sv.jd == bp_jobs and sv.cost < bp_cost):
                    bp_jobs  = sv.jd
                    bp_cost  = sv.cost
                    bp_time  = tt
                    bp_state = state

            # --- check final state --------------------------------------
            idx = smap_lookup(layer, final_state)
            if idx >= 0:
                sv = layer.vals[idx]
                better = sv.cost < best_final_cost
                if early and not better and fabs(sv.cost - best_final_cost) <= _EPS:
                    better = (sv.pen < best_final_pen or
                              (sv.pen == best_final_pen and tt < best_final_time))
                if better:
                    best_final_cost = sv.cost
                    best_final_pen  = sv.pen
                    best_final_time = tt

            if tt == T:
                continue

            # --- transitions (merged idle + jobs, single pass) ----------
            nlayer    = layers[tt + 1]
            remaining = T - tt
            _lb_b   = (tt // _LB_BLOCK) * _LB_BLOCK
            _lb_arr = <cnp.ndarray[cnp.float64_t, ndim=1]>_lb_py[_lb_b]
            _lb_len = <int>len(_lb_arr)

            for pos in range(layer.capacity):
                if layer.keys[pos] == _EMPTY:
                    continue
                state = layer.keys[pos]
                sv    = layer.vals[pos]
                c0 = sv.cost;  p0 = sv.pen;  rw = sv.rw;  jd = sv.jd

                # feasibility
                if rw > remaining:
                    continue

                # LB pruning (inlined)
                if rw < _lb_len:
                    lb_val = _lb_arr[rw]
                    if c0 + lb_val > best_final_cost:
                        continue
                else:
                    continue   # infeasible

                # -- idle ------------------------------------------------
                idx = smap_lookup(nlayer, state)
                if idx < 0:
                    nsv.cost = c0;  nsv.pen = p0;  nsv.rw = rw;  nsv.jd = jd
                    smap_put(nlayer, state, nsv)
                    if track_schedule:
                        pmap_set(c_parent, (tt + 1) * state_bound + state, 0)
                else:
                    better = c0 < nlayer.vals[idx].cost
                    if early and not better and fabs(c0 - nlayer.vals[idx].cost) <= _EPS:
                        better = p0 < nlayer.vals[idx].pen
                    if better:
                        nlayer.vals[idx].cost = c0
                        nlayer.vals[idx].pen  = p0
                        nlayer.vals[idx].rw   = rw
                        nlayer.vals[idx].jd   = jd
                        if track_schedule:
                            pmap_set(c_parent, (tt + 1) * state_bound + state, 0)

                # -- jobs (inline decode) --------------------------------
                x = state
                for i in range(K):
                    ui = <int>(x % c_rad[i])
                    x  = x // c_rad[i]

                    if ui >= c_tot[i]:
                        continue
                    L   = c_len[i]
                    end = tt + L
                    if end > T:
                        continue

                    ns  = state + c_inc[i]
                    nrw = rw - L
                    njd = jd + 1
                    cc  = c0 + (pprefix[end] - pprefix[tt])
                    cp  = p0 + tt if early else p0

                    tlayer = layers[end]
                    idx = smap_lookup(tlayer, ns)
                    if idx < 0:
                        nsv.cost = cc;  nsv.pen = cp
                        nsv.rw = nrw;   nsv.jd  = njd
                        smap_put(tlayer, ns, nsv)
                        if track_schedule:
                            pmap_set(c_parent, end * state_bound + ns, L)
                    else:
                        better = cc < tlayer.vals[idx].cost
                        if early and not better and fabs(cc - tlayer.vals[idx].cost) <= _EPS:
                            better = cp < tlayer.vals[idx].pen
                        if better:
                            tlayer.vals[idx].cost = cc
                            tlayer.vals[idx].pen  = cp
                            tlayer.vals[idx].rw   = nrw
                            tlayer.vals[idx].jd   = njd
                            if track_schedule:
                                pmap_set(c_parent, end * state_bound + ns, L)

            # -- free old layer (destroy to actually release memory) -----
            freed_t = tt - max_job_len
            if freed_t >= 0 and layers[freed_t] != NULL:
                smap_destroy(layers[freed_t])
                layers[freed_t] = NULL

        # ==============================================================
        #  Build return values
        # ==============================================================
        ret_cost = best_final_cost
        if ret_cost >= _C_INF:
            ret_cost = float("inf")

        py_parent = {}
        if track_schedule and c_parent != NULL:
            py_parent = pmap_to_pydict(c_parent)

        best_partial = None
        if timed_out and best_final_time < 0 and bp_jobs > 0:
            best_partial = (bp_time, bp_state, bp_cost)

        return ret_cost, best_final_time, py_parent, timed_out, best_partial

    finally:
        # -- always release native memory --------------------------------
        for tt in range(T + 1):
            if layers[tt] != NULL:
                smap_destroy(layers[tt])
        free(layers)
        if c_parent != NULL:
            pmap_destroy(c_parent)
