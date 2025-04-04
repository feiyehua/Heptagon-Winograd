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
Layer 0 :  Elapse time 241.651615 ms. (   45.11 GFlops) 
Layer 1 :  Elapse time 190.811316 ms. ( 1218.75 GFlops) 
Layer 2 :  Elapse time 78.155677 ms. ( 1461.06 GFlops) 
Layer 3 :  Elapse time 95.108986 ms. ( 2401.24 GFlops) 
Layer 4 :  Elapse time 38.569689 ms. ( 2853.93 GFlops) 
Layer 5 :  Elapse time 47.470649 ms. ( 4637.62 GFlops) 
Layer 6 :  Elapse time 47.452688 ms. ( 4639.37 GFlops) 
Layer 7 :  Elapse time 47.467311 ms. ( 4637.94 GFlops) 
Layer 8 :  Elapse time 19.821326 ms. ( 5149.63 GFlops) 
Layer 9 :  Elapse time 24.678628 ms. ( 8272.14 GFlops) 
Layer 10:  Elapse time 24.885972 ms. ( 8203.22 GFlops) 
Layer 11:  Elapse time 24.782658 ms. ( 8237.42 GFlops) 
Layer 12:  Elapse time 6.880681 ms. ( 6320.09 GFlops) 
Layer 13:  Elapse time 7.110039 ms. ( 6116.22 GFlops) 
Layer 14:  Elapse time 7.168055 ms. ( 6066.71 GFlops) 
Layer 15:  Elapse time 7.157644 ms. ( 6075.54 GFlops) 
Total elapse time: 0.909173. ( 2469.28 GFlops) 
```
更多详细信息，请参见`test-history.md`