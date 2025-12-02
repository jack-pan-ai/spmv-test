torchrun tutorial/shallow_water_equation/swe_main.py --backend cpu --device cpu --comm_backend gloo --output=res/ ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5 

torchrun tutorial/shallow_water_equation/swe_main.py --backend torch --device cpu --comm_backend gloo --output=res/ ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5 

torchrun tutorial/shallow_water_equation/swe_main.py --backend cuda --device cuda --comm_backend nccl --output=res/ ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5 

torchrun tutorial/shallow_water_equation/swe_main.py --backend torch --device cuda --comm_backend nccl --output=res/ ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5 