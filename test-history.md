## CPU单核测试
### Baseline
```
Layer 0 :  Elapse time 0.013987 ms. (    0.99 GFlops) 
Layer 1 :  Elapse time 0.039339 ms. (    1.76 GFlops) 
Layer 2 :  Elapse time 1.464367 ms. (    1.70 GFlops) 
Layer 3 :  Elapse time 25.849660 ms. (    6.57 GFlops) 
Layer 4 :  Elapse time 826.042652 ms. (    6.58 GFlops) 
Total elapse time: 0.853410. (    6.57 GFlops) 
```

```
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
```

### 1 仅优化了 filter_transform 访存顺序的结果
```
Layer 0 :  Elapse time 0.013669 ms. (    1.01 GFlops) 
Layer 1 :  Elapse time 0.040929 ms. (    1.69 GFlops) 
Layer 2 :  Elapse time 1.454035 ms. (    1.71 GFlops) 
Layer 3 :  Elapse time 26.830991 ms. (    6.33 GFlops) 
Layer 4 :  Elapse time 820.742289 ms. (    6.62 GFlops) 
Total elapse time: 0.849082. (    6.61 GFlops) 
```
```
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
```

### 2 将 filte_transform 转移到GPU上计算的结果
```
Layer 0 :  Elapse time 47.079961 ms. (    0.00 GFlops) 
Layer 1 :  Elapse time 0.056267 ms. (    1.23 GFlops) 
Layer 2 :  Elapse time 0.855605 ms. (    2.91 GFlops) 
Layer 3 :  Elapse time 24.053971 ms. (    7.06 GFlops) 
Layer 4 :  Elapse time 816.472689 ms. (    6.66 GFlops) 
Total elapse time: 0.888518. (    6.31 GFlops) 
```
```
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
```

### 3 将imag_transform转移到GPU上计算的结果
```
Layer 0 :  Elapse time 38.395007 ms. (    0.00 GFlops) 
Layer 1 :  Elapse time 0.175317 ms. (    0.39 GFlops) 
Layer 2 :  Elapse time 1.137018 ms. (    2.19 GFlops) 
Layer 3 :  Elapse time 24.143060 ms. (    7.04 GFlops) 
Layer 4 :  Elapse time 828.822374 ms. (    6.56 GFlops) 
Total elapse time: 0.892673. (    6.28 GFlops) 
```

```
Layer 0 :  Elapse time 38.949966 ms. (    0.00 GFlops) 
Layer 1 :  Elapse time 0.460386 ms. (    0.15 GFlops) 
Layer 2 :  Elapse time 1.597643 ms. (    1.56 GFlops) 
Layer 3 :  Elapse time 22.352695 ms. (    7.60 GFlops) 
Layer 4 :  Elapse time 721.854687 ms. (    7.53 GFlops) 
Total elapse time: 0.785215. (    7.14 GFlops) 
```
```
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
```

### 4 将output_transform放到GPU上
```
Layer 0 :  Elapse time 37.535270 ms. (    0.00 GFlops) 
Layer 1 :  Elapse time 0.658353 ms. (    0.10 GFlops) 
Layer 2 :  Elapse time 2.168655 ms. (    1.15 GFlops) 
Layer 3 :  Elapse time 21.558603 ms. (    7.88 GFlops) 
Layer 4 :  Elapse time 662.261645 ms. (    8.21 GFlops) 
Total elapse time: 0.724183. (    7.74 GFlops) 
```
```
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
```

### 5 在CPU上并行进行某些计算
```
Layer 0 :  Elapse time 89.387337 ms. (    0.00 GFlops) 
Layer 1 :  Elapse time 0.653028 ms. (    0.11 GFlops) 
Layer 2 :  Elapse time 1.711051 ms. (    1.45 GFlops) 
Layer 3 :  Elapse time 7.510026 ms. (   22.62 GFlops) 
Layer 4 :  Elapse time 175.617695 ms. (   30.95 GFlops) 
Total elapse time: 0.274879. (   20.40 GFlops) 
```
```
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
```
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
### 8 在device上进行output_unpacking
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

### 9 使用cudaHostAlloc()和cudaHostGetDevicePointer()处理packed_image
```
Layer 0 :  Elapse time 1329.838037 ms. (    8.20 GFlops) 
Layer 1 :  Elapse time 1901.472648 ms. (  122.30 GFlops) 
Layer 2 :  Elapse time 710.987965 ms. (  160.61 GFlops) 
Layer 3 :  Elapse time 989.209970 ms. (  230.87 GFlops) 
Layer 4 :  Elapse time 366.516987 ms. (  300.33 GFlops) 
Layer 5 :  Elapse time 533.113639 ms. (  412.95 GFlops) 
Layer 6 :  Elapse time 534.340302 ms. (  412.00 GFlops) 
Layer 7 :  Elapse time 533.324321 ms. (  412.79 GFlops) 
Layer 8 :  Elapse time 228.515069 ms. (  446.68 GFlops) 
Layer 9 :  Elapse time 309.005340 ms. (  660.65 GFlops) 
Layer 10:  Elapse time 308.699369 ms. (  661.31 GFlops) 
Layer 11:  Elapse time 310.138305 ms. (  658.24 GFlops) 
Layer 12:  Elapse time 57.237705 ms. (  759.75 GFlops) 
Layer 13:  Elapse time 67.476988 ms. (  644.46 GFlops) 
Layer 14:  Elapse time 50.674995 ms. (  858.15 GFlops) 
Layer 15:  Elapse time 50.034046 ms. (  869.14 GFlops) 
Total elapse time: 8.280586. (  271.12 GFlops) 
```

### 10 重复使用已经分配过的内存
```
Layer 0 :  Elapse time 3522.809664 ms. (    3.09 GFlops) 
Layer 1 :  Elapse time 1667.764346 ms. (  139.44 GFlops) 
Layer 2 :  Elapse time 680.066029 ms. (  167.91 GFlops) 
Layer 3 :  Elapse time 665.930986 ms. (  342.95 GFlops) 
Layer 4 :  Elapse time 317.683379 ms. (  346.49 GFlops) 
Layer 5 :  Elapse time 334.013303 ms. (  659.11 GFlops) 
Layer 6 :  Elapse time 332.919041 ms. (  661.27 GFlops) 
Layer 7 :  Elapse time 326.588710 ms. (  674.09 GFlops) 
Layer 8 :  Elapse time 168.953975 ms. (  604.14 GFlops) 
Layer 9 :  Elapse time 211.095651 ms. (  967.07 GFlops) 
Layer 10:  Elapse time 210.851669 ms. (  968.19 GFlops) 
Layer 11:  Elapse time 210.443974 ms. (  970.07 GFlops) 
Layer 12:  Elapse time 45.277675 ms. (  960.44 GFlops) 
Layer 13:  Elapse time 43.272336 ms. ( 1004.95 GFlops) 
Layer 14:  Elapse time 40.895065 ms. ( 1063.37 GFlops) 
Layer 15:  Elapse time 40.778399 ms. ( 1066.41 GFlops) 
Total elapse time: 8.819344. (  254.55 GFlops) 
```
由于内存尺寸较大，第一次分配所用时间较长。不过对于后续计算，速度提升明显。

### 使用float4进行向量化操作
```
Layer 0 :  Elapse time 3386.732658 ms. (    3.22 GFlops) 
Layer 1 :  Elapse time 1497.247696 ms. (  155.32 GFlops) 
Layer 2 :  Elapse time 616.797288 ms. (  185.13 GFlops) 
Layer 3 :  Elapse time 731.389602 ms. (  312.25 GFlops) 
Layer 4 :  Elapse time 273.213704 ms. (  402.89 GFlops) 
Layer 5 :  Elapse time 345.844666 ms. (  636.56 GFlops) 
Layer 6 :  Elapse time 353.331327 ms. (  623.07 GFlops) 
Layer 7 :  Elapse time 346.011400 ms. (  636.25 GFlops) 
Layer 8 :  Elapse time 176.636616 ms. (  577.87 GFlops) 
Layer 9 :  Elapse time 226.193349 ms. (  902.53 GFlops) 
Layer 10:  Elapse time 219.209035 ms. (  931.28 GFlops) 
Layer 11:  Elapse time 214.300315 ms. (  952.61 GFlops) 
Layer 12:  Elapse time 34.377019 ms. ( 1264.99 GFlops) 
Layer 13:  Elapse time 36.508004 ms. ( 1191.15 GFlops) 
Layer 14:  Elapse time 35.447677 ms. ( 1226.78 GFlops) 
Layer 15:  Elapse time 35.785357 ms. ( 1215.20 GFlops) 
Total elapse time: 8.529026. (  263.22 GFlops) 
```

### 不重复使用内存，而是重复利用一片预先申请的内存
```
Layer 0 :  Elapse time 2752.175967 ms. (    3.96 GFlops) 
Layer 1 :  Elapse time 872.197310 ms. (  266.63 GFlops) 
Layer 2 :  Elapse time 342.701356 ms. (  333.21 GFlops) 
Layer 3 :  Elapse time 477.219661 ms. (  478.56 GFlops) 
Layer 4 :  Elapse time 204.270363 ms. (  538.87 GFlops) 
Layer 5 :  Elapse time 271.559000 ms. (  810.69 GFlops) 
Layer 6 :  Elapse time 271.550020 ms. (  810.72 GFlops) 
Layer 7 :  Elapse time 271.502654 ms. (  810.86 GFlops) 
Layer 8 :  Elapse time 139.375925 ms. (  732.35 GFlops) 
Layer 9 :  Elapse time 175.503969 ms. ( 1163.19 GFlops) 
Layer 10:  Elapse time 175.465663 ms. ( 1163.45 GFlops) 
Layer 11:  Elapse time 175.499678 ms. ( 1163.22 GFlops) 
Layer 12:  Elapse time 24.385373 ms. ( 1783.30 GFlops) 
Layer 13:  Elapse time 20.622333 ms. ( 2108.71 GFlops) 
Layer 14:  Elapse time 17.397642 ms. ( 2499.57 GFlops) 
Layer 15:  Elapse time 17.552296 ms. ( 2477.54 GFlops) 
Total elapse time: 6.208979. (  361.57 GFlops) 
```

### 调整预分配内存大小
```
Layer 0 :  Elapse time 702.154318 ms. (   15.52 GFlops) 
Layer 1 :  Elapse time 845.278263 ms. (  275.12 GFlops) 
Layer 2 :  Elapse time 328.998407 ms. (  347.08 GFlops) 
Layer 3 :  Elapse time 460.026979 ms. (  496.45 GFlops) 
Layer 4 :  Elapse time 196.329037 ms. (  560.67 GFlops) 
Layer 5 :  Elapse time 262.057304 ms. (  840.09 GFlops) 
Layer 6 :  Elapse time 262.097359 ms. (  839.96 GFlops) 
Layer 7 :  Elapse time 262.052298 ms. (  840.10 GFlops) 
Layer 8 :  Elapse time 131.892681 ms. (  773.91 GFlops) 
Layer 9 :  Elapse time 166.488330 ms. ( 1226.18 GFlops) 
Layer 10:  Elapse time 166.129669 ms. ( 1228.83 GFlops) 
Layer 11:  Elapse time 166.431030 ms. ( 1226.61 GFlops) 
Layer 12:  Elapse time 23.096402 ms. ( 1882.83 GFlops) 
Layer 13:  Elapse time 19.392967 ms. ( 2242.39 GFlops) 
Layer 14:  Elapse time 19.431273 ms. ( 2237.97 GFlops) 
Layer 15:  Elapse time 19.184987 ms. ( 2266.70 GFlops) 
Total elapse time: 4.031041. (  556.93 GFlops) 
```
主机端的不可分页内存分配开销较大，尝试减小预分配内存的大小来加快计算。

### 将image_transform移回GPU上运行
```
Layer 0 :  Elapse time 402.759711 ms. (   27.07 GFlops) 
Layer 1 :  Elapse time 465.573947 ms. (  499.49 GFlops) 
Layer 2 :  Elapse time 239.952723 ms. (  475.89 GFlops) 
Layer 3 :  Elapse time 273.103952 ms. (  836.24 GFlops) 
Layer 4 :  Elapse time 154.797633 ms. (  711.09 GFlops) 
Layer 5 :  Elapse time 174.030304 ms. ( 1265.01 GFlops) 
Layer 6 :  Elapse time 174.055656 ms. ( 1264.83 GFlops) 
Layer 7 :  Elapse time 174.161355 ms. ( 1264.06 GFlops) 
Layer 8 :  Elapse time 116.247336 ms. (  878.06 GFlops) 
Layer 9 :  Elapse time 128.268957 ms. ( 1591.54 GFlops) 
Layer 10:  Elapse time 128.501336 ms. ( 1588.66 GFlops) 
Layer 11:  Elapse time 128.822009 ms. ( 1584.71 GFlops) 
Layer 12:  Elapse time 18.419345 ms. ( 2360.92 GFlops) 
Layer 13:  Elapse time 15.098015 ms. ( 2880.28 GFlops) 
Layer 14:  Elapse time 11.871338 ms. ( 3663.15 GFlops) 
Layer 15:  Elapse time 11.783679 ms. ( 3690.40 GFlops) 
Total elapse time: 2.617447. (  857.71 GFlops) 
```
发现在image_transform部分还是有严重的性能瓶颈：一部分是执行cuda核函数耗时较长，另一部分是在CPU上工作时间较长。将image_transform移回了GPU。

### 使用多卡并行计算
```
Layer 0 :  Elapse time 263.965050 ms. (   41.30 GFlops) 
Layer 1 :  Elapse time 242.462715 ms. (  959.12 GFlops) 
Layer 2 :  Elapse time 124.992053 ms. (  913.58 GFlops) 
Layer 3 :  Elapse time 142.429988 ms. ( 1603.45 GFlops) 
Layer 4 :  Elapse time 80.421050 ms. ( 1368.74 GFlops) 
Layer 5 :  Elapse time 90.396007 ms. ( 2435.40 GFlops) 
Layer 6 :  Elapse time 97.833316 ms. ( 2250.26 GFlops) 
Layer 7 :  Elapse time 90.494633 ms. ( 2432.75 GFlops) 
Layer 8 :  Elapse time 60.083310 ms. ( 1698.85 GFlops) 
Layer 9 :  Elapse time 67.023675 ms. ( 3045.87 GFlops) 
Layer 10:  Elapse time 66.790978 ms. ( 3056.48 GFlops) 
Layer 11:  Elapse time 67.115704 ms. ( 3041.69 GFlops) 
Layer 12:  Elapse time 10.475318 ms. ( 4151.33 GFlops) 
Layer 13:  Elapse time 8.117040 ms. ( 5357.44 GFlops) 
Layer 14:  Elapse time 8.020004 ms. ( 5422.26 GFlops) 
Layer 15:  Elapse time 7.972002 ms. ( 5454.91 GFlops) 
Total elapse time: 1.428593. ( 1571.48 GFlops) 
```
发现任务本身有非常好的并行性：每个batch间互不关联。

将batch分成两半放到两个GPU上计算。由于性能瓶颈主要在于内存和显存之间的数据传输，性能提升比例接近一倍。

### 优化内存拷贝方式：使用cudaMemcpy3D
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
发现cudaMemcpy3D复制内存实在是太慢了，使用对齐内存的收益小于复制内存的代价。所以将输入输出的内存操作换成了普通一维内存复制。