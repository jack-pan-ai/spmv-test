import os
import time
from typing import Tuple

import torch
from torch.utils.cpp_extension import load
import torch.nn as nn
from typing import Optional

try:
    import numpy as np  # Prefer fast CSV loading when available
except Exception:
    np = None

def build_extension() -> object:
    src = "/home/panq/dev/FlexSpmv/EASIER/trash/merged_binding.cu"
    include_root = "/home/panq/dev/FlexSpmv/EASIER/trash"
    name = "merged_spmv_ext"

    ext = load(
        name=name,
        sources=[src],
        extra_include_paths=[include_root],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return ext


@torch.inference_mode()
def gen_random_csr(
    num_rows: int, num_cols: int, nnz_per_row: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    nnz_per_row = max(1, int(nnz_per_row))
    total_nnz = num_rows * nnz_per_row

    # CSR row_ptr: length num_rows + 1
    row_ptr = torch.arange(0, total_nnz + 1, nnz_per_row, dtype=torch.int32)
    if row_ptr.numel() != num_rows + 1:
        row_ptr = torch.empty(num_rows + 1, dtype=torch.int32)
        row_ptr[0] = 0
        step = max(1, total_nnz // num_rows)
        for r in range(1, num_rows + 1):
            row_ptr[r] = min(total_nnz, row_ptr[r - 1] + step)
        row_ptr[-1] = total_nnz

    # Column indices, values, and dense vector
    col_idx = torch.randint(0, num_cols, (total_nnz,), dtype=torch.int32, device=device).contiguous()
    sx = torch.randn(total_nnz, device=device, dtype=dtype).contiguous()
    x = torch.randn(num_cols, device=device, dtype=dtype).contiguous()
    row_ptr = row_ptr.to(device).contiguous()

    return sx, x, col_idx, row_ptr, num_rows, num_cols


@torch.inference_mode()
def load_res_data(
    res_dir: str, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    edge_path = os.path.join(res_dir, "edge.csv")
    x_idx_path = os.path.join(res_dir, "x_indices.csv")
    row_ptr_path = os.path.join(res_dir, "row_end_offsets.csv")
    vec_path = os.path.join(res_dir, "vector.csv")

    if np is not None:
        sx_np = np.loadtxt(edge_path, dtype=np.float64)
        x_np = np.loadtxt(vec_path, dtype=np.float64)
        col_np = np.loadtxt(x_idx_path, dtype=np.int32)
        row_ptr_np = np.loadtxt(row_ptr_path, dtype=np.int32)

        sx = torch.from_numpy(sx_np).to(dtype=dtype, device=device).contiguous()
        x = torch.from_numpy(x_np).to(dtype=dtype, device=device).contiguous()
        col_idx = torch.from_numpy(col_np.astype(np.int32)).to(device=device).contiguous()
        row_ptr = torch.from_numpy(row_ptr_np.astype(np.int32)).to(device=device).contiguous()
    else:
        # Fallback pure-Python loader
        def load_float_lines(p):
            with open(p, "r") as f:
                return [float(line.strip()) for line in f if line.strip()]
        def load_int_lines(p):
            with open(p, "r") as f:
                return [int(line.strip()) for line in f if line.strip()]
        sx = torch.tensor(load_float_lines(edge_path), dtype=dtype, device=device).contiguous()
        x = torch.tensor(load_float_lines(vec_path), dtype=dtype, device=device).contiguous()
        col_idx = torch.tensor(load_int_lines(x_idx_path), dtype=torch.int32, device=device).contiguous()
        row_ptr = torch.tensor(load_int_lines(row_ptr_path), dtype=torch.int32, device=device).contiguous()

    # Infer sizes and basic sanity checks
    num_rows = int(row_ptr.numel() - 1)
    num_cols = int(x.numel())
    nnz = int(sx.numel())
    assert nnz == int(col_idx.numel()), "edge.csv and x_indices.csv must have same length"
    assert int(row_ptr[-1].item()) == nnz, "row_end_offsets[-1] must equal number of nonzeros"

    return sx, x, col_idx, row_ptr, num_rows, num_cols


@torch.inference_mode()
def csr_spmv_reference(
    sx: torch.Tensor, col_idx: torch.Tensor, row_ptr: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    num_rows = row_ptr.numel() - 1
    out = torch.zeros(num_rows, dtype=x.dtype, device="cpu")
    sx_cpu = sx.cpu()
    col_cpu = col_idx.cpu()
    x_cpu = x.cpu()
    row_ptr_cpu = row_ptr.cpu()
    for r in range(num_rows):
        start = int(row_ptr_cpu[r].item())
        end = int(row_ptr_cpu[r + 1].item())
        if end > start:
            cols = col_cpu[start:end].long()
            vals = sx_cpu[start:end]
            out[r] = (vals * x_cpu[cols]).sum()
    return out.to(x.device)


class SpmvCudaWrapper(nn.Module):
    def __init__(self, ext_module, selector_idx: torch.Tensor, row_end_offsets: torch.Tensor, d_temp_storage: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        t = selector_idx.contiguous()
        if t.dtype != torch.int32:
            t = t.to(torch.int32)
        self.register_buffer("selector_idx", t, persistent=False)
        if row_end_offsets is not None:
            reo = row_end_offsets.contiguous()
            if reo.dtype != torch.int32:
                reo = reo.to(torch.int32)
        else:
            reo = None
        self.register_buffer("row_end_offsets", reo, persistent=False)
        self.register_buffer("d_temp_storage", d_temp_storage, persistent=False)
        self._ext = ext_module

    def __call__(self, sx: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sx = sx if sx.is_contiguous() else sx.contiguous()
        x = x if x.is_contiguous() else x.contiguous()
        y = y if y.is_contiguous() else y.contiguous()
        device = sx.device

        ro = self.row_end_offsets
        if ro is not None and ro.device != device:
            ro = ro.to(device, non_blocking=True)
            self.row_end_offsets = ro
            self.register_buffer("row_end_offsets", ro, persistent=False)

        gi = self.selector_idx
        if gi.device != device:
            gi = gi.to(device, non_blocking=True)
            self.selector_idx = gi
            self.register_buffer("selector_idx", gi, persistent=False)

        # Ensure a Tensor (not None) is passed to the extension
        d_temp_storage = self.d_temp_storage
        if d_temp_storage is None:
            d_temp_storage = torch.empty(0, dtype=torch.uint8, device=device)
            self.d_temp_storage = d_temp_storage
            self.register_buffer("d_temp_storage", d_temp_storage, persistent=False)
        else:
            if d_temp_storage.dtype != torch.uint8:
                d_temp_storage = d_temp_storage.to(dtype=torch.uint8)
                self.d_temp_storage = d_temp_storage
                self.register_buffer("d_temp_storage", d_temp_storage, persistent=False)
            if d_temp_storage.device != device:
                d_temp_storage = d_temp_storage.to(device, non_blocking=True)
                self.d_temp_storage = d_temp_storage
                self.register_buffer("d_temp_storage", d_temp_storage, persistent=False)

        num_cols = int(x.shape[0])
        num_rows = int(ro.numel() - 1) if ro is not None else num_cols

        outs = self._ext.merged_spmv_launch(sx, x, gi, ro, num_rows, num_cols, d_temp_storage)
        # if isinstance(outs, tuple):
        #     return outs[0]
        y.copy_(outs[0])


def benchmark(
    ext: object,
    num_rows: int = 4096,
    num_cols: int = 4096,
    nnz_per_row: int = 16,
    dtype: torch.dtype = torch.float32,
    warmup: int = 100,
    iters: int = 5,
    use_res: bool = True,
    res_dir: str = "/home/panq/dev/FlexSpmv/EASIER/res",
) -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    if use_res:
        sx, x, col_idx, row_ptr, nrows, ncols = load_res_data(res_dir, device=device, dtype=dtype)
    else:
        sx, x, col_idx, row_ptr, nrows, ncols = gen_random_csr(
            num_rows=num_rows, num_cols=num_cols, nnz_per_row=nnz_per_row, device=device, dtype=dtype
        )
    d_temp_storage = None
    wrapper = SpmvCudaWrapper(ext, col_idx, row_ptr, d_temp_storage)

    # Warmup
    y = torch.empty(nrows, dtype=dtype, device=device)
    for _ in range(warmup):
        wrapper(sx, x, y)
    torch.cuda.synchronize()

    # Benchmark
    num_profile_iters = 100
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_profile_iters):
        wrapper(sx, x, y)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"Average time over {num_profile_iters} iterations: {1000*(t1-t0)/num_profile_iters:.3f} ms")
    # for i in range(iters):
        # # start.record()
        # y = wrapper(sx, x)
        # # end.record()
        # # torch.cuda.synchronize()
        # # t_ms = start.elapsed_time(end)
        # # times_ms.append(t_ms)
        # # print(f"iter {i + 1}: {t_ms:.3f} ms")

    # print(f"[merged_spmv] dtype={dtype}, rows={nrows}, cols={ncols}, nnz={sx.numel()}")

    # Correctness check
    # ref = csr_spmv_reference(sx, col_idx, row_ptr, x)
    # max_abs_err = (y - ref).abs().max().item()
    # print(f"  max_abs_err: {max_abs_err:.6e}")


if __name__ == "__main__":
    ext = build_extension()
    rows = int(os.getenv("EASIER_BENCH_ROWS", "4096"))
    cols = int(os.getenv("EASIER_BENCH_COLS", "4096"))
    nnzpr = int(os.getenv("EASIER_BENCH_NNZ_PER_ROW", "16"))
    dtype_str = os.getenv("EASIER_BENCH_DTYPE", "fp64").lower()
    dtype = torch.float32 if dtype_str in ("fp32", "float", "float32") else torch.float64
    use_res = os.getenv("EASIER_BENCH_USE_RES", "1") not in ("0", "false", "False")
    res_dir = os.getenv("EASIER_BENCH_RES_DIR", "/home/panq/dev/FlexSpmv/EASIER/res")
    benchmark(ext, rows, cols, nnzpr, dtype=dtype, use_res=use_res, res_dir=res_dir)


# python -u /home/panq/dev/FlexSpmv/EASIER/trash/bench_merged_spmv.py
# nsys profile   -o /home/panq/dev/FlexSpmv/EASIER/trash/spmv_nsys_source   -t cuda,nvtx,osrt   -s none --force-overwrite true  python /home/panq/dev/FlexSpmv/EASIER/trash/bench_merged_spmv.py

