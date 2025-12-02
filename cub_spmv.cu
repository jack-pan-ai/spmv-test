// Copyright (c) 2025
// CUB-based SpMV using segmented reduction with CSR end-offsets
// Exposes: cub_spmv(row_end_offsets, x_indices, x, optional edge) -> y

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>

template<typename index_t, typename scalar_t>
__global__ void compute_values_kernel(
    const scalar_t* __restrict__ x,            // [rows]
    const index_t* __restrict__ x_indices,     // [nnz]
    const scalar_t* __restrict__ edge,         // [nnz] or nullptr
    scalar_t* __restrict__ vals,               // [nnz]
    int64_t nnz,
    bool use_edge)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < nnz) {
        scalar_t xv = x[static_cast<int64_t>(x_indices[i])];
        vals[i] = use_edge ? edge[i] * xv : xv;
    }
}

template<typename index_t>
__global__ void build_offsets_kernel(
    const index_t* __restrict__ row_end_offsets, // [rows]
    index_t* __restrict__ offsets,               // [rows + 1]
    int64_t rows)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i == 0) {
        offsets[0] = static_cast<index_t>(0);
    }
    if (i < rows) {
        offsets[i + 1] = row_end_offsets[i];
    }
}

template<typename index_t, typename scalar_t>
static inline void cub_spmv_impl(
    const torch::Tensor& row_end_offsets, // 1D, int32/int64, CUDA
    const torch::Tensor& x_indices,       // 1D, int32/int64, CUDA
    const torch::Tensor& x,               // 1D, float/double, CUDA
    const c10::optional<torch::Tensor>& edge_opt, // 1D same dtype as x, CUDA
    torch::Tensor& y)                     // 1D, same dtype/device as x
{
    const int64_t rows = row_end_offsets.size(0);
    const int64_t nnz = x_indices.size(0);

    auto vals = torch::empty({nnz}, x.options());
    auto offsets = torch::empty({rows + 1}, row_end_offsets.options());

    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 256;
    const int blocks_vals = static_cast<int>((nnz + threads - 1) / threads);
    const int blocks_rows = static_cast<int>((rows + threads - 1) / threads);

    const bool use_edge = edge_opt.has_value() && edge_opt.value().defined();

    compute_values_kernel<index_t, scalar_t><<<blocks_vals, threads, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        x_indices.data_ptr<index_t>(),
        use_edge ? edge_opt.value().data_ptr<scalar_t>() : nullptr,
        vals.data_ptr<scalar_t>(),
        nnz,
        use_edge
    );

    build_offsets_kernel<index_t><<<blocks_rows, threads, 0, stream>>>(
        row_end_offsets.data_ptr<index_t>(),
        offsets.data_ptr<index_t>(),
        rows
    );

    // Size query
    size_t temp_bytes = 0;
    void* temp_storage = nullptr;
    cub::DeviceSegmentedReduce::Sum(
        temp_storage, temp_bytes,
        vals.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        rows,
        offsets.data_ptr<index_t>(),
        offsets.data_ptr<index_t>() + 1,
        stream
    );

    auto d_temp = torch::empty(
        {static_cast<long>(temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(y.device())
    );
    temp_storage = d_temp.data_ptr();

    // Actual compute
    cub::DeviceSegmentedReduce::Sum(
        temp_storage, temp_bytes,
        vals.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        rows,
        offsets.data_ptr<index_t>(),
        offsets.data_ptr<index_t>() + 1,
        stream
    );
}

torch::Tensor cub_spmv_forward(
    torch::Tensor row_end_offsets,
    torch::Tensor x_indices,
    torch::Tensor x,
    c10::optional<torch::Tensor> edge)
{
    TORCH_CHECK(row_end_offsets.is_cuda() && x_indices.is_cuda() && x.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(row_end_offsets.dim() == 1 && x_indices.dim() == 1 && x.dim() == 1, "Inputs must be 1D");
    if (edge.has_value() && edge.value().defined()) {
        TORCH_CHECK(edge.value().is_cuda() && edge.value().dim() == 1, "edge must be 1D CUDA if provided");
        TORCH_CHECK(edge.value().size(0) == x_indices.size(0), "edge length must equal nnz");
        TORCH_CHECK(edge.value().scalar_type() == x.scalar_type(), "edge dtype must match x dtype");
    }

    const int64_t rows = row_end_offsets.size(0);
    auto y = torch::empty({rows}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cub_spmv_forward", [&]{
        if (row_end_offsets.scalar_type() == torch::kInt32 && x_indices.scalar_type() == torch::kInt32) {
            cub_spmv_impl<int, scalar_t>(row_end_offsets, x_indices, x, edge, y);
        } else if (row_end_offsets.scalar_type() == torch::kInt64 && x_indices.scalar_type() == torch::kInt64) {
            cub_spmv_impl<long long, scalar_t>(row_end_offsets, x_indices, x, edge, y);
        } else {
            TORCH_CHECK(false, "Index dtypes must match and be int32 or int64");
        }
    });
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cub_spmv", &cub_spmv_forward, "CUB-based SpMV (CSR end-offsets)");
}


