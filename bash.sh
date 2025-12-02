# 1️⃣ Start your program normally
torchrun ./tutorial/shallow_water_equation/swe_single_profile.py \
    --backend cuda --device cuda --comm_backend nccl \
    ~/.easier/triangular_100.hdf5 ~/.easier/SW_100.hdf5 &

PID=$!
echo "PID = $PID"

# 2️⃣ Wait until JIT finishes (adjust time as needed)
sleep 75

# 3️⃣ Attach Nsight Systems to the running process
nsys profile --pid $PID \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite true \
    -o /home/panq/dev/FlexSpmv/EASIER/trash/spmv_nsys_easier
