torchrun tutorial/shallow_water_equation/swe_main_profile.py --backend torch --device cpu --comm_backend gloo --tb_worker_name torch_cpu --profile_dir logs ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5 

torchrun tutorial/shallow_water_equation/swe_main_profile.py --backend cpu --device cpu --comm_backend gloo --tb_worker_name cpu_gen --with_modules --export_chrome --profile_dir logs ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5

torchrun tutorial/shallow_water_equation/swe_main_profile.py --backend torch --device cuda --comm_backend nccl --tb_worker_name torch_cuda --nvtx --profile_dir logs ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5

torchrun tutorial/shallow_water_equation/swe_main_profile.py --backend cuda --device cuda --comm_backend nccl --tb_worker_name cuda_gen --nvtx --tb_use_gzip --export_chrome --profile_dir logs ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5