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
Layer 0 :  Elapse time 213.248332 ms. (   51.12 GFlops) 
Layer 1 :  Elapse time 163.712343 ms. ( 1420.49 GFlops) 
Layer 2 :  Elapse time 68.859339 ms. ( 1658.31 GFlops) 
Layer 3 :  Elapse time 80.984354 ms. ( 2820.05 GFlops) 
Layer 4 :  Elapse time 33.820947 ms. ( 3254.65 GFlops) 
Layer 5 :  Elapse time 39.873679 ms. ( 5521.20 GFlops) 
Layer 6 :  Elapse time 39.976676 ms. ( 5506.98 GFlops) 
Layer 7 :  Elapse time 40.067991 ms. ( 5494.43 GFlops) 
Layer 8 :  Elapse time 16.530673 ms. ( 6174.74 GFlops) 
Layer 9 :  Elapse time 20.323674 ms. ( 10044.70 GFlops) 
Layer 10:  Elapse time 20.395676 ms. ( 10009.24 GFlops) 
Layer 11:  Elapse time 20.346324 ms. ( 10033.52 GFlops) 
Layer 12:  Elapse time 5.293687 ms. ( 8214.79 GFlops) 
Layer 13:  Elapse time 4.890680 ms. ( 8891.72 GFlops) 
Layer 14:  Elapse time 4.874706 ms. ( 8920.85 GFlops) 
Layer 15:  Elapse time 4.813989 ms. ( 9033.37 GFlops) 
Total elapse time: 0.778013. ( 2885.56 GFlops) 
```
更多详细信息，请参见`test-history.md`