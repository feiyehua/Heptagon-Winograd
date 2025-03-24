# 
## slurm 集群管理

通过`scontrol show node`可以看到集群拥有的GPU资源，受限于相关要求不在此直接粘贴。

## spack
在集群上使用spack：
`source /data/spack/share/spack/setup-env.sh`
`spack env activate -p winograd`
### 使用了的spack包

## 性能分析
`numactl --cpunodebind=0-3 --membind=0-3 perf stat -ddd perf record -e cycles:u -g -- ./winograd conf/vgg16.conf `

使用nsys
`nsys profile --stats=true -o winograd1 ./winograd conf/vgg16.conf`
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

## cudaMalloc3D
```
int main()
{
    cudaPitchedPtr pitchedDevPtr;
    cudaExtent extent=make_cudaExtent(3,13,47);
    cudaMalloc3D(&pitchedDevPtr,extent);
    printf("%ld %ld %ld\n",pitchedDevPtr.pitch,pitchedDevPtr.xsize,pitchedDevPtr.ysize);
}
```
`512 3 13`

结论：cudaExtent中，width对应x，height对应y，depth对应z，且z会被填充以实现内存对齐。