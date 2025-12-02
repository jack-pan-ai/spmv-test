// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "include/merged_spmv.cuh"
#include "include/merged_utils.cuh"

namespace {

template <typename ValueT, typename OffsetT>
static std::tuple<torch::Tensor> merged_spmv_launch_typed(
    torch::Tensor sx,
    torch::Tensor x,
    torch::Tensor gather_src_idx,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols,
    torch::Tensor temp_storage
) {
  TORCH_CHECK(sx.is_cuda(), "sx must be a CUDA tensor");
  TORCH_CHECK(sx.is_contiguous(), "sx must be contiguous");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(gather_src_idx.is_cuda(), "gather_src_idx must be a CUDA tensor");
  TORCH_CHECK(gather_src_idx.is_contiguous(), "gather_src_idx must be contiguous");
  TORCH_CHECK(row_end_offsets.is_cuda(), "row_end_offsets must be a CUDA tensor");
  TORCH_CHECK(row_end_offsets.is_contiguous(), "row_end_offsets must be contiguous");

  TORCH_CHECK(sx.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "sx dtype mismatch");
  TORCH_CHECK(x.scalar_type() ==             c10::CppTypeToScalarType<ValueT>::value, "x dtype mismatch");
  TORCH_CHECK(gather_src_idx.scalar_type() ==             c10::CppTypeToScalarType<OffsetT>::value, "gather_src_idx dtype mismatch");
  TORCH_CHECK(row_end_offsets.scalar_type() ==             c10::CppTypeToScalarType<OffsetT>::value, "row_end_offsets dtype mismatch");


  const int64_t ne = gather_src_idx.numel();
  TORCH_CHECK(ne > 0, "selector index must be non-empty");
  // TORCH_CHECK(row_end_offsets.numel() == num_rows + 1, "row_end_offsets must have length num_rows + 1");


  auto options_val = torch::TensorOptions().dtype(sx.scalar_type()).device(sx.device());
  torch::Tensor out_0_scatter = torch::zeros({num_rows}, options_val);


  FlexParams<ValueT, OffsetT> params;
  params.sx_ptr =         reinterpret_cast<ValueT*>(sx.data_ptr());
  params.x_ptr =         reinterpret_cast<ValueT*>(x.data_ptr());

  params.gather_src_ptr =         reinterpret_cast<OffsetT*>((gather_src_idx).data_ptr());

  params.output_y_scatter_ptr =                 reinterpret_cast<ValueT*>(out_0_scatter.data_ptr());

  params.d_row_end_offsets = reinterpret_cast<OffsetT*>(row_end_offsets.data_ptr());
  params.num_rows = static_cast<int>(num_rows);
  params.num_cols = static_cast<int>(num_cols);
  params.num_nonzeros = static_cast<int>(ne);

  size_t temp_storage_bytes = 0;
  auto stream = at::cuda::getCurrentCUDAStream();

  // Always do a clean size query with nullptr to avoid any invalid-pointer paths
  void* d_temp_storage = nullptr;
  cudaError_t err = merged::merged_spmv_launch<ValueT, OffsetT>(
      params, /*d_temp_storage=*/nullptr, temp_storage_bytes, /*debug_synchronous=*/false, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "merged_spmv_launch (size query) failed: ", cudaGetErrorString(err));

  if (temp_storage_bytes > 0) {
    // In-place resize only: do not rebind to a new tensor or change device
    TORCH_CHECK(temp_storage.defined(), "temp_storage must be provided as a CUDA uint8 tensor");
    TORCH_CHECK(temp_storage.device() == sx.device(), "temp_storage must be on the same device as inputs");
    if (static_cast<size_t>(temp_storage.numel()) < temp_storage_bytes) {
      temp_storage.resize_({static_cast<long long>(temp_storage_bytes)});
    }
    d_temp_storage = temp_storage.data_ptr();
  }

  err = merged::merged_spmv_launch<ValueT, OffsetT>(
      params, d_temp_storage, temp_storage_bytes, /*debug_synchronous=*/false, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "merged_spmv_launch failed: ", cudaGetErrorString(err));

  return std::make_tuple(out_0_scatter);
}

static std::tuple<torch::Tensor> merged_spmv_launch_bind(
    torch::Tensor sx,
    torch::Tensor x,
    torch::Tensor gather_src_idx,
    torch::Tensor row_end_offsets,
    int64_t num_rows,
    int64_t num_cols,
    torch::Tensor temp_storage
) {
  TORCH_CHECK(sx.device().is_cuda(), "CUDA device required");
  switch (sx.scalar_type()) {
    case torch::kFloat:
      return merged_spmv_launch_typed<float, int>(sx, x, gather_src_idx, row_end_offsets, num_rows, num_cols, temp_storage);
    case torch::kDouble:
      return merged_spmv_launch_typed<double, int>(sx, x, gather_src_idx, row_end_offsets, num_rows, num_cols, temp_storage);
        
    default:
      std::cerr << "Unsupported dtype for ValueT. dtype code: " << static_cast<int>(sx.scalar_type()) << std::endl;
      TORCH_CHECK(false, "Unsupported dtype for ValueT. Use float32 or float64.");
      return std::make_tuple(torch::Tensor());
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "merged_spmv_launch",
      &merged_spmv_launch_bind,
      "Run merged SpMV kernel"
      , py::arg("sx")
      , py::arg("x")
      , py::arg("gather_src_idx")
      , py::arg("row_end_offsets")
      , py::arg("num_rows")
      , py::arg("num_cols")
      , py::arg("temp_storage")
  );
}


