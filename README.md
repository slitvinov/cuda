# cuda

```
PATH=$PATH:/usr/local/cuda-12.8/bin
```

```
nvcc -arch native clock.cu
for j in `seq 0 10`; do for i in `seq 30`; do ./a.out $((1<<i)) > log.$j; done; done
```

