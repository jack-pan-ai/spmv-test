module purge
module load gcc/12.3.0
module load cuda/12.4.0/gcc-12.3.0
# cuda 11.8
#export PATH=$HOME/cuda-11.8/bin:$PATH
#export LD_LIBRARY_PATH=$HOME/cuda-11.8/lib64:$LD_LIBRARY_PATH
#export CUDACXX=$HOME/cuda/11.8/bin/nvcc

conda activate ENV_NAME

export _MAGMA_ROOT_=/home/panq/dev/magma-2.7.2
export LD_LIBRARY_PATH=/home/panq/dev/magma-2.7.2/lib:$LD_LIBRARY_PATH

export CUDA_HOME=$CUDA_ROOT
export CUDADIR=$CUDA_HOME
export _CUB_DIR_=$CUDA_HOME

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
