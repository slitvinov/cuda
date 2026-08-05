# cuda

```
mcu nvcc info.cu -O3 -gencode arch=compute_100a,code=sm_100a -cudart shared -Xlinker -s -- a.out -- ./a.out
```


# References

1. Volkov, V., & Demmel, J. W. (2008, November). Benchmarking GPUs to tune dense linear algebra. In SC'08: Proceedings of the 2008 ACM/IEEE conference on Supercomputing (pp. 1-11). IEEE.
2. Huerta, R., Shoushtary, M. A., Cruz, J. L., & González, A. (2025). Analyzing Modern NVIDIA GPU cores. arXiv preprint arXiv:2503.20481.
3. - https://sf-tensor.com/engineering/bitwise-tcgen05
   - https://github.com/sf-tensor/tcgen05
