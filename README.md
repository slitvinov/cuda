# cuda

```
nvcc -arch native clock.cu
for j in `seq 0 10`; do for i in `seq 30`; do ./a.out $((1<<i)) > log.$j; done; done
```

./job.sh ncu --clock-control none --metrics gpu__time_duration.sum nvcc -arch native -run h.cu
./job.sh ncu -c 10 --print-summary per-kernel  --clock-control none --metrics gpu__time_duration.sum nvcc -arch native -run max.cu


```
mcu nvcc info.cu -O3 -gencode arch=compute_100a,code=sm_100a -cudart shared -Xlinker -s -- a.out -- ./a.out
```


# References

1. Volkov, V., & Demmel, J. W. (2008, November). Benchmarking GPUs to tune dense linear algebra. In SC'08: Proceedings of the 2008 ACM/IEEE conference on Supercomputing (pp. 1-11). IEEE.
2. Huerta, R., Shoushtary, M. A., Cruz, J. L., & González, A. (2025). Analyzing Modern NVIDIA GPU cores. arXiv preprint arXiv:2503.20481.
