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
Layer 0 :  Elapse time 212.748686 ms. (   51.24 GFlops) 
Layer 1 :  Elapse time 183.533986 ms. ( 1267.07 GFlops) 
Layer 2 :  Elapse time 73.375305 ms. ( 1556.24 GFlops) 
Layer 3 :  Elapse time 90.285937 ms. ( 2529.52 GFlops) 
Layer 4 :  Elapse time 36.099672 ms. ( 3049.21 GFlops) 
Layer 5 :  Elapse time 46.532710 ms. ( 4731.09 GFlops) 
Layer 6 :  Elapse time 45.066675 ms. ( 4885.00 GFlops) 
Layer 7 :  Elapse time 45.095682 ms. ( 4881.86 GFlops) 
Layer 8 :  Elapse time 17.872016 ms. ( 5711.31 GFlops) 
Layer 9 :  Elapse time 23.170392 ms. ( 8810.60 GFlops) 
Layer 10:  Elapse time 23.216327 ms. ( 8793.17 GFlops) 
Layer 11:  Elapse time 23.267666 ms. ( 8773.77 GFlops) 
Layer 12:  Elapse time 6.084601 ms. ( 7146.98 GFlops) 
Layer 13:  Elapse time 5.991379 ms. ( 7258.19 GFlops) 
Layer 14:  Elapse time 5.950371 ms. ( 7308.21 GFlops) 
Layer 15:  Elapse time 5.924304 ms. ( 7340.36 GFlops) 
Total elapse time: 0.844216. ( 2659.28 GFlops) 
```
更多详细信息，请参见`test-history.md`