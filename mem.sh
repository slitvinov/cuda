n=16384
s=$((1<<26))
export M_GPU
M_GPU=B200
mcu nvcc mem.cu -O2 -gencode arch=compute_100a,code=sm_100a -cudart shared -Xlinker -s -- a.out -- \
    ./a.out -n $n -s $s -o $M_GPU -- $M_GPU.raw $M_GPU.meta
