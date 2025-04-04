# Heptagon-Winograd

## 原始仓库
[recruitment-2025-spring](https://github.com/heptagonhust/recruitment-2025-spring)

原始README.md：请参见`README_ORI.md`

## 构建与运行

本项目依赖于**cuda@12.6**（使用cuda@11.8似乎有奇怪的问题），使用make构建。

`make`

在Slurm集群上运行：
`sbatch ./run.sh `

如果运行机器上只有1个GPU，请修改`src/cpu/winograd.cc`中GPU_NUM宏为1.

## 优化结果
```
Layer 0 :  Elapse time 218.646606 ms. (   49.86 GFlops) 
Layer 1 :  Elapse time 190.879345 ms. ( 1218.31 GFlops) 
Layer 2 :  Elapse time 78.370969 ms. ( 1457.04 GFlops) 
Layer 3 :  Elapse time 95.205625 ms. ( 2398.81 GFlops) 
Layer 4 :  Elapse time 38.945675 ms. ( 2826.38 GFlops) 
Layer 5 :  Elapse time 47.721306 ms. ( 4613.26 GFlops) 
Layer 6 :  Elapse time 47.821999 ms. ( 4603.54 GFlops) 
Layer 7 :  Elapse time 47.720989 ms. ( 4613.29 GFlops) 
Layer 8 :  Elapse time 19.616286 ms. ( 5203.46 GFlops) 
Layer 9 :  Elapse time 24.838686 ms. ( 8218.84 GFlops) 
Layer 10:  Elapse time 24.917364 ms. ( 8192.89 GFlops) 
Layer 11:  Elapse time 24.988651 ms. ( 8169.52 GFlops) 
Layer 12:  Elapse time 6.929000 ms. ( 6276.02 GFlops) 
Layer 13:  Elapse time 6.523609 ms. ( 6666.03 GFlops) 
Layer 14:  Elapse time 4.811287 ms. ( 9038.44 GFlops) 
Layer 15:  Elapse time 4.800320 ms. ( 9059.09 GFlops) 
Total elapse time: 0.882738. ( 2543.23 GFlops) 
```
更多详细信息，请参见`test-history.md`