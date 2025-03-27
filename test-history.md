## CPU单核测试
### Baseline
`
Layer 0 :  Elapse time 0.013987 ms. (    0.99 GFlops) 
Layer 1 :  Elapse time 0.039339 ms. (    1.76 GFlops) 
Layer 2 :  Elapse time 1.464367 ms. (    1.70 GFlops) 
Layer 3 :  Elapse time 25.849660 ms. (    6.57 GFlops) 
Layer 4 :  Elapse time 826.042652 ms. (    6.58 GFlops) 
Total elapse time: 0.853410. (    6.57 GFlops) 
`

`
Layer 0 :  Elapse time 8186.878602 ms. (    1.33 GFlops) 
Layer 1 :  Elapse time 40464.980682 ms. (    5.75 GFlops) 
Layer 2 :  Elapse time 18053.032001 ms. (    6.33 GFlops) 
Layer 3 :  Elapse time 35898.586353 ms. (    6.36 GFlops) 
Layer 4 :  Elapse time 16899.485032 ms. (    6.51 GFlops) 
Layer 5 :  Elapse time 41035.766999 ms. (    5.36 GFlops) 
Layer 6 :  Elapse time 41052.740335 ms. (    5.36 GFlops) 
Layer 7 :  Elapse time 40832.878272 ms. (    5.39 GFlops) 
Layer 8 :  Elapse time 20145.650625 ms. (    5.07 GFlops) 
Layer 9 :  Elapse time 43018.066645 ms. (    4.75 GFlops) 
Layer 10:  Elapse time 42978.799979 ms. (    4.75 GFlops) 
Layer 11:  Elapse time 43129.189968 ms. (    4.73 GFlops) 
Layer 12:  Elapse time 8022.422632 ms. (    5.42 GFlops) 
Layer 13:  Elapse time 7970.119715 ms. (    5.46 GFlops) 
Layer 14:  Elapse time 7960.529010 ms. (    5.46 GFlops) 
Layer 15:  Elapse time 7955.145677 ms. (    5.47 GFlops) 
Total elapse time: 423.604273. (    5.30 GFlops) 
`

### 1 仅优化了 filter_transform 访存顺序的结果
`
Layer 0 :  Elapse time 0.013669 ms. (    1.01 GFlops) 
Layer 1 :  Elapse time 0.040929 ms. (    1.69 GFlops) 
Layer 2 :  Elapse time 1.454035 ms. (    1.71 GFlops) 
Layer 3 :  Elapse time 26.830991 ms. (    6.33 GFlops) 
Layer 4 :  Elapse time 820.742289 ms. (    6.62 GFlops) 
Total elapse time: 0.849082. (    6.61 GFlops) 

`
`
Layer 0 :  Elapse time 7653.730710 ms. (    1.42 GFlops) 
Layer 1 :  Elapse time 36484.388351 ms. (    6.37 GFlops) 
Layer 2 :  Elapse time 16407.272975 ms. (    6.96 GFlops) 
Layer 3 :  Elapse time 33699.397008 ms. (    6.78 GFlops) 
Layer 4 :  Elapse time 15922.520638 ms. (    6.91 GFlops) 
Layer 5 :  Elapse time 38205.842336 ms. (    5.76 GFlops) 
Layer 6 :  Elapse time 38205.424945 ms. (    5.76 GFlops) 
Layer 7 :  Elapse time 38111.322641 ms. (    5.78 GFlops) 
Layer 8 :  Elapse time 18619.194031 ms. (    5.48 GFlops) 
Layer 9 :  Elapse time 41983.799299 ms. (    4.86 GFlops) 
Layer 10:  Elapse time 42004.275640 ms. (    4.86 GFlops) 
Layer 11:  Elapse time 42003.477971 ms. (    4.86 GFlops) 
Layer 12:  Elapse time 7759.553671 ms. (    5.60 GFlops) 
Layer 13:  Elapse time 7748.500268 ms. (    5.61 GFlops) 
Layer 14:  Elapse time 7748.624007 ms. (    5.61 GFlops) 
Layer 15:  Elapse time 7748.088360 ms. (    5.61 GFlops) 
Total elapse time: 400.305413. (    5.61 GFlops) 
`

### 2 将 filte_transform 转移到GPU上计算的结果
`
Layer 0 :  Elapse time 47.079961 ms. (    0.00 GFlops) 
Layer 1 :  Elapse time 0.056267 ms. (    1.23 GFlops) 
Layer 2 :  Elapse time 0.855605 ms. (    2.91 GFlops) 
Layer 3 :  Elapse time 24.053971 ms. (    7.06 GFlops) 
Layer 4 :  Elapse time 816.472689 ms. (    6.66 GFlops) 
Total elapse time: 0.888518. (    6.31 GFlops) 
`
`
Layer 0 :  Elapse time 7679.061333 ms. (    1.42 GFlops) 
Layer 1 :  Elapse time 36590.525309 ms. (    6.36 GFlops) 
Layer 2 :  Elapse time 16455.760956 ms. (    6.94 GFlops) 
Layer 3 :  Elapse time 34384.700378 ms. (    6.64 GFlops) 
Layer 4 :  Elapse time 16276.542346 ms. (    6.76 GFlops) 
Layer 5 :  Elapse time 38111.253977 ms. (    5.78 GFlops) 
Layer 6 :  Elapse time 38132.897695 ms. (    5.77 GFlops) 
Layer 7 :  Elapse time 38130.178690 ms. (    5.77 GFlops) 
Layer 8 :  Elapse time 18568.888664 ms. (    5.50 GFlops) 
Layer 9 :  Elapse time 41950.209697 ms. (    4.87 GFlops) 
Layer 10:  Elapse time 41962.100347 ms. (    4.86 GFlops) 
Layer 11:  Elapse time 41949.927012 ms. (    4.87 GFlops) 
Layer 12:  Elapse time 7754.538377 ms. (    5.61 GFlops) 
Layer 13:  Elapse time 7742.335717 ms. (    5.62 GFlops) 
Layer 14:  Elapse time 7744.733731 ms. (    5.61 GFlops) 
Layer 15:  Elapse time 7744.950692 ms. (    5.61 GFlops) 
Total elapse time: 401.178605. (    5.60 GFlops) 
`

### 3 将imag_transform转移到GPU上计算的结果
`
Layer 0 :  Elapse time 38.395007 ms. (    0.00 GFlops) 
Layer 1 :  Elapse time 0.175317 ms. (    0.39 GFlops) 
Layer 2 :  Elapse time 1.137018 ms. (    2.19 GFlops) 
Layer 3 :  Elapse time 24.143060 ms. (    7.04 GFlops) 
Layer 4 :  Elapse time 828.822374 ms. (    6.56 GFlops) 
Total elapse time: 0.892673. (    6.28 GFlops) 
`

`
Layer 0 :  Elapse time 38.949966 ms. (    0.00 GFlops) 
Layer 1 :  Elapse time 0.460386 ms. (    0.15 GFlops) 
Layer 2 :  Elapse time 1.597643 ms. (    1.56 GFlops) 
Layer 3 :  Elapse time 22.352695 ms. (    7.60 GFlops) 
Layer 4 :  Elapse time 721.854687 ms. (    7.53 GFlops) 
Total elapse time: 0.785215. (    7.14 GFlops) 
`
`
Layer 0 :  Elapse time 7566.001972 ms. (    1.44 GFlops) 
Layer 1 :  Elapse time 31855.883678 ms. (    7.30 GFlops) 
Layer 2 :  Elapse time 15184.141000 ms. (    7.52 GFlops) 
Layer 3 :  Elapse time 31583.126307 ms. (    7.23 GFlops) 
Layer 4 :  Elapse time 15431.320985 ms. (    7.13 GFlops) 
Layer 5 :  Elapse time 37436.971664 ms. (    5.88 GFlops) 
Layer 6 :  Elapse time 37740.124385 ms. (    5.83 GFlops) 
Layer 7 :  Elapse time 37691.179991 ms. (    5.84 GFlops) 
Layer 8 :  Elapse time 18541.388671 ms. (    5.51 GFlops) 
Layer 9 :  Elapse time 41391.911666 ms. (    4.93 GFlops) 
Layer 10:  Elapse time 41392.256339 ms. (    4.93 GFlops) 
Layer 11:  Elapse time 41394.100666 ms. (    4.93 GFlops) 
Layer 12:  Elapse time 7609.156052 ms. (    5.72 GFlops) 
Layer 13:  Elapse time 7603.738387 ms. (    5.72 GFlops) 
Layer 14:  Elapse time 7603.749355 ms. (    5.72 GFlops) 
Layer 15:  Elapse time 7604.112387 ms. (    5.72 GFlops) 
Total elapse time: 387.629164. (    5.79 GFlops) 
`

### 4 将output_transform放到GPU上
`
Layer 0 :  Elapse time 37.535270 ms. (    0.00 GFlops) 
Layer 1 :  Elapse time 0.658353 ms. (    0.10 GFlops) 
Layer 2 :  Elapse time 2.168655 ms. (    1.15 GFlops) 
Layer 3 :  Elapse time 21.558603 ms. (    7.88 GFlops) 
Layer 4 :  Elapse time 662.261645 ms. (    8.21 GFlops) 
Total elapse time: 0.724183. (    7.74 GFlops) 
`
`
Layer 0 :  Elapse time 6147.348642 ms. (    1.77 GFlops) 
Layer 1 :  Elapse time 30793.569009 ms. (    7.55 GFlops) 
Layer 2 :  Elapse time 14657.356342 ms. (    7.79 GFlops) 
Layer 3 :  Elapse time 30932.409604 ms. (    7.38 GFlops) 
Layer 4 :  Elapse time 15101.811647 ms. (    7.29 GFlops) 
Layer 5 :  Elapse time 36791.375001 ms. (    5.98 GFlops) 
Layer 6 :  Elapse time 36786.189000 ms. (    5.98 GFlops) 
Layer 7 :  Elapse time 36774.437348 ms. (    5.99 GFlops) 
Layer 8 :  Elapse time 18225.524664 ms. (    5.60 GFlops) 
Layer 9 :  Elapse time 41456.574361 ms. (    4.92 GFlops) 
Layer 10:  Elapse time 41385.831992 ms. (    4.93 GFlops) 
Layer 11:  Elapse time 41467.782736 ms. (    4.92 GFlops) 
Layer 12:  Elapse time 7612.705708 ms. (    5.71 GFlops) 
Layer 13:  Elapse time 7622.874022 ms. (    5.70 GFlops) 
Layer 14:  Elapse time 7621.769667 ms. (    5.71 GFlops) 
Layer 15:  Elapse time 7629.914999 ms. (    5.70 GFlops) 
Total elapse time: 381.007475. (    5.89 GFlops) 
`

### 5 在CPU上并行进行某些计算
`
Layer 0 :  Elapse time 89.387337 ms. (    0.00 GFlops) 
Layer 1 :  Elapse time 0.653028 ms. (    0.11 GFlops) 
Layer 2 :  Elapse time 1.711051 ms. (    1.45 GFlops) 
Layer 3 :  Elapse time 7.510026 ms. (   22.62 GFlops) 
Layer 4 :  Elapse time 175.617695 ms. (   30.95 GFlops) 
Total elapse time: 0.274879. (   20.40 GFlops) 
`
`
Layer 0 :  Elapse time 2622.104009 ms. (    4.16 GFlops) 
Layer 1 :  Elapse time 5294.917027 ms. (   43.92 GFlops) 
Layer 2 :  Elapse time 2023.618698 ms. (   56.43 GFlops) 
Layer 3 :  Elapse time 3124.241670 ms. (   73.10 GFlops) 
Layer 4 :  Elapse time 1264.706612 ms. (   87.04 GFlops) 
Layer 5 :  Elapse time 2180.993319 ms. (  100.94 GFlops) 
Layer 6 :  Elapse time 2174.528281 ms. (  101.24 GFlops) 
Layer 7 :  Elapse time 2175.268014 ms. (  101.21 GFlops) 
Layer 8 :  Elapse time 970.446030 ms. (  105.18 GFlops) 
Layer 9 :  Elapse time 1765.080611 ms. (  115.66 GFlops) 
Layer 10:  Elapse time 1763.073285 ms. (  115.79 GFlops) 
Layer 11:  Elapse time 1763.282696 ms. (  115.78 GFlops) 
Layer 12:  Elapse time 345.638037 ms. (  125.82 GFlops) 
Layer 13:  Elapse time 339.621305 ms. (  128.04 GFlops) 
Layer 14:  Elapse time 339.550654 ms. (  128.07 GFlops) 
Layer 15:  Elapse time 340.331316 ms. (  127.78 GFlops) 
Total elapse time: 28.487402. (   78.81 GFlops) 
`
提升显著！

进行性能分析，结果显示主要的性能瓶颈在于：
+ 还没有将矩阵运算转移到GPU上
+ 在Host和Device之间复制数据消耗了过多时间
考虑将矩阵运算迁移到Device上，其他内存密集型任务仍然保留在CPU上运行，并且避免D->H,H->D的大规模数据传输。

### 6 在GPU上进行矩阵乘法
尝试手写矩阵乘法，发现效果不佳。最终选择了使用Cublas。
```
Layer 0 :  Elapse time 2508.474271 ms. (    4.35 GFlops) 
Layer 1 :  Elapse time 3951.502641 ms. (   58.85 GFlops) 
Layer 2 :  Elapse time 1378.393968 ms. (   82.84 GFlops) 
Layer 3 :  Elapse time 1582.231363 ms. (  144.34 GFlops) 
Layer 4 :  Elapse time 655.709028 ms. (  167.87 GFlops) 
Layer 5 :  Elapse time 765.391986 ms. (  287.63 GFlops) 
Layer 6 :  Elapse time 760.832310 ms. (  289.35 GFlops) 
Layer 7 :  Elapse time 761.743069 ms. (  289.01 GFlops) 
Layer 8 :  Elapse time 343.159676 ms. (  297.45 GFlops) 
Layer 9 :  Elapse time 426.395337 ms. (  478.77 GFlops) 
Layer 10:  Elapse time 427.709023 ms. (  477.30 GFlops) 
Layer 11:  Elapse time 432.726701 ms. (  471.76 GFlops) 
Layer 12:  Elapse time 102.845271 ms. (  422.83 GFlops) 
Layer 13:  Elapse time 91.254314 ms. (  476.54 GFlops) 
Layer 14:  Elapse time 90.542634 ms. (  480.29 GFlops) 
Layer 15:  Elapse time 91.135343 ms. (  477.16 GFlops) 
Total elapse time: 14.370047. (  156.23 GFlops) 
```

### 7 重用cublas句柄、并行释放device memory
```
Layer 0 :  Elapse time 2687.819640 ms. (    4.06 GFlops) 
Layer 1 :  Elapse time 3814.441681 ms. (   60.97 GFlops) 
Layer 2 :  Elapse time 1278.712670 ms. (   89.30 GFlops) 
Layer 3 :  Elapse time 1443.269332 ms. (  158.24 GFlops) 
Layer 4 :  Elapse time 582.054615 ms. (  189.12 GFlops) 
Layer 5 :  Elapse time 673.742374 ms. (  326.76 GFlops) 
Layer 6 :  Elapse time 663.888693 ms. (  331.61 GFlops) 
Layer 7 :  Elapse time 667.628288 ms. (  329.75 GFlops) 
Layer 8 :  Elapse time 294.077714 ms. (  347.09 GFlops) 
Layer 9 :  Elapse time 337.957382 ms. (  604.06 GFlops) 
Layer 10:  Elapse time 340.071360 ms. (  600.30 GFlops) 
Layer 11:  Elapse time 335.350355 ms. (  608.75 GFlops) 
Layer 12:  Elapse time 82.052708 ms. (  529.98 GFlops) 
Layer 13:  Elapse time 77.980359 ms. (  557.66 GFlops) 
Layer 14:  Elapse time 79.369704 ms. (  547.90 GFlops) 
Layer 15:  Elapse time 75.077693 ms. (  579.22 GFlops) 
Total elapse time: 13.433495. (  167.12 GFlops) 
```
### 在device上进行output_unpacking
```
Layer 0 :  Elapse time 1507.894675 ms. (    7.23 GFlops) 
Layer 1 :  Elapse time 2647.610664 ms. (   87.83 GFlops) 
Layer 2 :  Elapse time 830.993970 ms. (  137.41 GFlops) 
Layer 3 :  Elapse time 1022.726297 ms. (  223.30 GFlops) 
Layer 4 :  Elapse time 410.699288 ms. (  268.02 GFlops) 
Layer 5 :  Elapse time 567.717393 ms. (  387.78 GFlops) 
Layer 6 :  Elapse time 547.333956 ms. (  402.22 GFlops) 
Layer 7 :  Elapse time 545.069297 ms. (  403.89 GFlops) 
Layer 8 :  Elapse time 249.289672 ms. (  409.45 GFlops) 
Layer 9 :  Elapse time 306.940317 ms. (  665.10 GFlops) 
Layer 10:  Elapse time 307.230393 ms. (  664.47 GFlops) 
Layer 11:  Elapse time 306.181351 ms. (  666.75 GFlops) 
Layer 12:  Elapse time 74.176709 ms. (  586.26 GFlops) 
Layer 13:  Elapse time 65.590620 ms. (  663.00 GFlops) 
Layer 14:  Elapse time 66.000700 ms. (  658.88 GFlops) 
Layer 15:  Elapse time 60.832024 ms. (  714.86 GFlops) 
Total elapse time: 9.516287. (  235.91 GFlops) 
```