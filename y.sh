h=t4
n=25
export M_GPU
M_GPU=T4
mcu nvcc launch.cu -O2 -gencode arch=compute_100a,code=sm_100a -cudart shared -Xlinker -s -- a.out -- \
    ./a.out -n $((1<<n)) -o $h -- $h.raw $h.meta
python launch.py $h > data/$h
