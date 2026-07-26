# cuda

```
nvcc -arch native clock.cu
for j in `seq 0 10`; do for i in `seq 30`; do ./a.out $((1<<i)) > log.$j; done; done
```

./job.sh ncu --clock-control none --metrics gpu__time_duration.sum nvcc -arch native -run h.cu
./job.sh ncu -c 10 --print-summary per-kernel  --clock-control none --metrics gpu__time_duration.sum nvcc -arch native -run max.cu

