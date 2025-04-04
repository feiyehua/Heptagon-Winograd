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
Layer 0 :  Elapse time 220.254024 ms. (   49.49 GFlops) 
Layer 1 :  Elapse time 189.580599 ms. ( 1226.66 GFlops) 
Layer 2 :  Elapse time 78.013341 ms. ( 1463.72 GFlops) 
Layer 3 :  Elapse time 94.525337 ms. ( 2416.07 GFlops) 
Layer 4 :  Elapse time 38.886309 ms. ( 2830.70 GFlops) 
Layer 5 :  Elapse time 47.551314 ms. ( 4629.75 GFlops) 
Layer 6 :  Elapse time 47.699690 ms. ( 4615.35 GFlops) 
Layer 7 :  Elapse time 47.696670 ms. ( 4615.64 GFlops) 
Layer 8 :  Elapse time 19.411325 ms. ( 5258.40 GFlops) 
Layer 9 :  Elapse time 24.846315 ms. ( 8216.32 GFlops) 
Layer 10:  Elapse time 24.900675 ms. ( 8198.38 GFlops) 
Layer 11:  Elapse time 25.003354 ms. ( 8164.71 GFlops) 
Layer 12:  Elapse time 7.035653 ms. ( 6180.88 GFlops) 
Layer 13:  Elapse time 7.061323 ms. ( 6158.41 GFlops) 
Layer 14:  Elapse time 6.968657 ms. ( 6240.30 GFlops) 
Layer 15:  Elapse time 6.981293 ms. ( 6229.01 GFlops) 
Total elapse time: 0.886416. ( 2532.67 GFlops) 
```
更多详细信息，请参见`test-history.md`