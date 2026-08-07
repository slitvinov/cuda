n=25
export M_GPU
M_GPU=B200
mcu nvcc launch.cu -O2 -gencode arch=compute_100a,code=sm_100a -cudart shared -Xlinker -s -- a.out -- \
    ./a.out -n $((1<<n)) -o $M_GPU -- $M_GPU.raw $M_GPU.meta
python launch.py $M_GPU > data/$M_GPU
