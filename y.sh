h=b200
n=20
mcu nvcc launch.cu -gencode arch=compute_100a,code=sm_100a -cudart shared -Xlinker -s -- a.out -- \
    ./a.out -n $((1<<n)) -o $h -- $h.raw $h.meta
python launch.py $h > data/$h
