# 
## spack
在集群上使用spack：
`source /data/spack/share/spack/setup-env.sh`

### 使用了的spack包


## 进行了的优化
### 计算时的内存访问优化
``` cpp
typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
for (int64_t idx = 0; idx < collapsed_dim_size; idx++) { 
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      z4 = M_tensor[0][w][idx];
      z4 = M_tensor[1][w][idx];
      ...
    }
}
```
每次计算时，先逐`ti.tile_in_w`