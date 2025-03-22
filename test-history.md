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


### 进行简单的性能分析：
```
Filter took 0 milliseconds to execute.
Image took 573 milliseconds to execute.
Sgemm took 2877 milliseconds to execute.
Output took 11678 milliseconds to execute.
Filter took 0 milliseconds to execute.
Image took 544 milliseconds to execute.
Sgemm took 2874 milliseconds to execute.
Output took 11021 milliseconds to execute.
Filter took 0 milliseconds to execute.
Image took 546 milliseconds to execute.
Sgemm took 2875 milliseconds to execute.
Output took 10698 milliseconds to execute.
Layer 0 :  Elapse time 14775.998036 ms. (    0.74 GFlops) 
```