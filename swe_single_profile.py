# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import torch
import time
from tqdm import tqdm
import numpy as np

from torch.profiler import profile, record_function, ProfilerActivity

import easier as esr

class ShallowWaterEquation(esr.Module):
    def __init__(self, mesh_path: str, sw_path: str, dt=0.005, device='cpu') -> None:
        super().__init__()

        self.dt = dt
        # src (torch.LongTensor): src cell indices, with shape `(ne,)`
        self.src = esr.hdf5(mesh_path, 'src', dtype=torch.long)
        # dst (torch.LongTensor): dst cell indices, with shape `(ne,)`
        self.dst = esr.hdf5(mesh_path, 'dst', dtype=torch.long)
        self.ne = self.src.shape[0]

        # cells (torch.LongTensor): three point indices for each triangle
        #   cells, with shape `(nc, 3)`, `nc` means number of cells
        self.cells = esr.hdf5(mesh_path, 'cells', dtype=torch.long)
        self.nc = self.cells.shape[0]

        # points (torch.DoubleTensor): point coordinates on a plane,
        #   with shape `(np, 2)`, `np` means number of points
        self.points = esr.hdf5(mesh_path, 'points', dtype=torch.long)
        self.np = self.points.shape[0]

        # bcells (torch.LongTensor): boundary cell indices, with shape `(nbc,)`,
        #   `nbc` means number of boundary cell
        self.bcells = esr.hdf5(mesh_path, 'bcells', dtype=torch.long)
        self.nbc = self.bcells.shape[0]

        # bpoints (torch.LongTensor): boundary points indices in each boundary
        #   cell, with shape `(nbc, 2)`, `nbc` means number of boundary cell
        self.bpoints = esr.hdf5(mesh_path, 'bpoints', dtype=torch.long)

        self.scatter = esr.Reducer(self.dst, self.nc)
        self.gather_src = esr.Selector(self.src)
        self.gather_dst = esr.Selector(self.dst)
        self.scatter_b = esr.Reducer(self.bcells, self.nc)
        self.gather_b = esr.Selector(self.bcells)

        self.x = esr.Tensor(
            esr.hdf5(sw_path, 'x', dtype=torch.double), mode='partition'
        )
        self.y = esr.Tensor(
            esr.hdf5(sw_path, 'y', dtype=torch.double), mode='partition'
        )
        self.area = esr.Tensor(
            esr.hdf5(sw_path, 'area', dtype=torch.double), mode='partition'
        )
        self.sx = esr.Tensor(
            esr.hdf5(sw_path, 'sx', dtype=torch.double), mode='partition'
        )
        self.sy = esr.Tensor(
            esr.hdf5(sw_path, 'sy', dtype=torch.double), mode='partition'
        )
        self.bsx = esr.Tensor(
            esr.hdf5(sw_path, 'bsx', dtype=torch.double), mode='partition'
        )
        self.bsy = esr.Tensor(
            esr.hdf5(sw_path, 'bsy', dtype=torch.double), mode='partition'
        )
        self.h = esr.Tensor(
            esr.hdf5(sw_path, 'h', dtype=torch.double), mode='partition'
        )
        self.alpha = esr.Tensor(
            esr.hdf5(sw_path, 'alpha', dtype=torch.double), mode='partition'
        )

        self.uh = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double), mode='partition'
        )
        self.vh = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double), mode='partition'
        )

        self.to(device)

    def face_reconstruct(self, phi):
        return (1 - self.alpha) * self.gather_src(phi) + \
            self.alpha * self.gather_dst(phi)

    def delta(self, h, uh, vh):
        h_f = self.face_reconstruct(h)
        uh_f = self.face_reconstruct(uh)
        vh_f = self.face_reconstruct(vh)

        u_f = uh_f / h_f
        v_f = vh_f / h_f

        h_f_square = 0.5 * h_f * h_f
        uh_f_times_sx = uh_f * self.sx
        vh_f_times_sy = vh_f * self.sy
        gather_b_h_square = 0.5 * self.gather_b(h)**2

        delta_h = - (
            self.scatter(uh_f_times_sx + vh_f_times_sy)
        ) / self.area

        delta_uh = - (
            self.scatter((u_f * uh_f + h_f_square) * self.sx +
                         u_f * vh_f_times_sy) +
            self.scatter_b(gather_b_h_square * self.bsx)
        ) / self.area

        delta_vh = - (
            self.scatter(v_f * uh_f_times_sx +
                         (v_f * vh_f + h_f_square) * self.sy) +
            self.scatter_b(gather_b_h_square * self.bsy)
        ) / self.area

        return delta_h, delta_uh, delta_vh

    def forward(self):
        # Minimal SpMV: y = A * x where A is defined by (src, dst).
        # Only a single Reducer call performs the reduction.
        vec = self.gather_src(self.x)
        edge = self.sx
        self.y.copy_(self.scatter(vec * edge))

    def export_res(self, res_dir: str = None):
        """
        Save vector, edge, x_indices, and row_end_offsets into ./res for external SpMV benchmarks.
        """
        # Only rank-0 writes files if running distributed
        try:
            from easier.core.runtime.dist_env import get_default_dist_env
            if get_default_dist_env().rank != 0:
                return
        except Exception:
            pass

        # Resolve default res directory relative to repo root
        if res_dir is None:
            # .../EASIER/tutorial/shallow_water_equation/swe_single_profile.py -> .../EASIER/res
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            res_dir = os.path.join(repo_root, "res")
        os.makedirs(res_dir, exist_ok=True)

        # Collect full tensors (ensure data is ready post-compile)
        # vector: dense input x
        vector = self.x.collect().cpu().numpy()
        # per-edge weights
        edge_vals = self.sx.collect().cpu().numpy()
        # source/destination indices for edges (after compile / rewriting)
        src_idx = self.gather_src.idx.detach().cpu().numpy()
        dst_idx = self.scatter.idx.detach().cpu().numpy()

        # Build CSR by sorting edges by row (dst)
        num_rows = int(self.nc)
        # stable sort to keep deterministic order within a row
        perm = np.argsort(dst_idx, kind="stable")
        col_idx = src_idx[perm].astype(np.int64, copy=False)
        edge_sorted = edge_vals[perm]

        # row_end_offsets (CSR row_ptr), length = num_rows + 1
        counts = np.bincount(dst_idx, minlength=num_rows).astype(np.int64, copy=False)
        row_ptr = np.empty(num_rows + 1, dtype=np.int64)
        row_ptr[0] = 0
        np.cumsum(counts, out=row_ptr[1:])

        # Save CSVs
        vec_path = os.path.join(res_dir, "vector.csv")
        edge_path = os.path.join(res_dir, "edge.csv")
        xidx_path = os.path.join(res_dir, "x_indices.csv")
        reo_path = os.path.join(res_dir, "row_end_offsets.csv")

        # Use double precision text for floats; int32 for indices/row_ptr to match loaders
        np.savetxt(vec_path, vector, fmt="%.18e")
        np.savetxt(edge_path, edge_sorted, fmt="%.18e")
        np.savetxt(xidx_path, col_idx.astype(np.int32, copy=False), fmt="%d")
        np.savetxt(reo_path, row_ptr.astype(np.int32, copy=False), fmt="%d")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument(
        "--backend", type=str, choices=["none", "torch", "cpu", "cuda"],
        default='torch'
    )
    parser.add_argument(
        "--comm_backend", type=str, choices=["gloo", "nccl"],
        default='gloo'
    )
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--output", type=str)
    parser.add_argument("--profile_dir", type=str, default="logs")
    # TensorBoard trace handler options
    parser.add_argument("--tb_worker_name", type=str, default=None)
    parser.add_argument("--tb_use_gzip", action="store_true")
    # Profiler detail toggles
    parser.add_argument("--no_record_shapes", action="store_true")
    parser.add_argument("--no_profile_memory", action="store_true")
    parser.add_argument("--no_stack", action="store_true")
    parser.add_argument("--with_modules", action="store_true")
    # Extra exports/marking
    parser.add_argument("--export_chrome", action="store_true")
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--profile_iters", type=int, default=10000)
    parser.add_argument("mesh", type=str)
    parser.add_argument("shallow_water", type=str)
    args = parser.parse_args()
    
    #  ncu --set full -f  --target-processes all  --launch-skip 80  --launch-count 200  -o ./trash/ncu_swe_profile_gen torchrun tutorial/shallow_water_equation/swe_single_profile.py --backend cuda --device cuda --comm_backend nccl ~/.easier/triangular_1000.hdf5 ~/.easier/SW_1000.hdf5
    
    #  nsys profile   -o /home/panq/dev/FlexSpmv/EASIER/trash/spmv_nsys_easier   --trace=cuda,nvtx,osrt --cuda-memory-usage=true --cuda-um-cpu-page-faults=true  --sample=process-tree --python-sampling=true --python-sampling-frequency=1000 --delay=60 --force-overwrite true  torchrun ./tutorial/shallow_water_equation/swe_single_profile.py         --backend cuda --device cuda --comm_backend nccl         ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5
    
    # nsys profile   -o /home/panq/dev/FlexSpmv/EASIER/trash/spmv_nsys_easier_2000   --trace=cuda,nvtx,osrt,mpi --cuda-memory-usage=true --cuda-um-cpu-page-faults=true   --delay=60 --force-overwrite true  torchrun ./tutorial/shallow_water_equation/swe_single_profile.py         --backend cuda --device cuda --comm_backend nccl         ~/.easier/triangular_1000.hdf5 ~/.easier/SW_1000.hdf5


    esr.init(args.comm_backend)

    eqn = ShallowWaterEquation(
        args.mesh, args.shallow_water, args.dt, args.device
    )
    [eqn] = esr.compile([eqn], args.backend)
    eqn()
    # Save data for external SpMV benchmarks
    # eqn.export_res()
    eqn.jit_engine.graph.print_tabular()

    for _ in tqdm(range(args.warmup_iters)):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push("warmup/step")
        eqn()
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
        # print(f"eqn.y.shape: {eqn.y.shape}, eqn.x.shape: {eqn.x.shape}, eqn.sx.shape: {eqn.sx.shape}")
    torch.cuda.synchronize()

    # # Optional: short sleep to stabilize GPU frequency
    # time.sleep(0.1)

    # --- Profiling / monitored phase ---
    import time

    num_profile_iters = 100
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    for i in range(num_profile_iters):
        eqn()
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"Average time over {num_profile_iters} iterations: {1000*(t1-t0)/num_profile_iters:.3f} ms")
