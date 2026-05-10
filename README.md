# Sparse mat-vec on WGPU

Fast sparse matrix-vector multiply, even on modest integrated GPUs!

https://github.com/user-attachments/assets/32bf8f5d-559b-4eab-af23-86d6f0050306

## Installation

Installing this currently requires you to clone and install fastplotlib 
from source using the `ndwidget` branch. You can try out 
the kernels with random matrices using `sanity_check_gen_data.py`.

Running the benchmarks and viz requires the datasets and `masknmf` from the `cached-dmr-arrays` branch, 
contact me if you need help with installation.

Install `fastplotlib`:

```bash
git clone https://github.com/fastplotlib/fastplotlib.git
git checkout ndwidget

cd fastplotlib

# install with extras in place
pip install -e ".[imgui]"
```

Install `masknmf`:

```bash
git clone https://github.com/apasarkar/masknmf-toolbox.git
git checkout cached-dmr-arrays

pip install .
```

## Layout

### project dir

#### `_spmv.py`

* `ComputeShader`: helper class adopted from pygfx, modified to be compatible with basic uniform buffers. 
This class manages bindings (i.e. describe data that gets binded to the GPU), and the pipeline. We can 
update the uniform buffer and re-run the compute pipeline to compute and render a specific frame of data.

* `create_storage_buffer`: helper class to create storage buffers for our data on the GPU

* `MatrixCSR`: dataclass to organize the buffers that represent a CSR matrix

* `SpMVImage`: the main class that I wrote. This takes a sparse CSR matrix `A` representing spatial components
of neurons, and a dense matrix `C` representing temporal components. IT uses `ComputeShader` to send the data to 
the GPU, run the CSR scalar or CSR vector compute shader (kernel), based on a `kwarg`, and it also manages a 
fastplotlib `TextureArray` which allows us to visualize the results as new frames are reconstructed from `A` and `C`.

#### shaders (compute kernels)

* `matvec.wgsl` - not used, just a basic naive mat-vec kernel that operates on dense arrays
`matvec_coalesced.wgsl` - basic mat-vec that uses coalesced as opposed to strided memory access for more efficient reads. See: https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/

* `spmv_csr_scalar.wgsl` - basic naive sparse CSR mat-vec implementation

* `spmv_csr_vector.wgsl` - vector form of sparse CSR mat-vec implementation that performs more efficient reads by parallelizing across local invocations (a.k.a. threads) within a row

### Example files

* `sanity_check_gen_data.py`: performs sanity check on the `spmv_csr_scalar.wgsl` and `spmv_csr_vector.wgsl` kernels against scipy sparse 
results by generating random data. This also ensure the entire pipeline from the `SpMVImage` and `ComputeShader` classes are also working correctly. We 
expect the diff between the scipy.sparse result and my kernel to be 0.0 or within machine precision for float32, 2^-23 or ~1.19 x 10^-7.

* `sanity_check.py`: perform sanity check similar to the gen_data above, but uses a real dataset loaded from masknmf.

* `run_viz.py`: visualizes a reconstruction as fast as it can possibly be computed & rendered using `SpMVImage`.

The rest of the file are various benchmarking files.

## Benchmarks

First, comparison with the torch implemention in [masknmf](https://github.com/apasarkar/masknmf-toolbox). The wgsl compute shader is at-par or better than torch's sparse mat-vec. Torch's sparse operations in general are not the best part of the library so maybe not too surprising?

The "denoise" is a denser sparse matrix with ~43 nnz/row, and "demix" is much less dense at ~0.4 nnz/row.

<img width="958" height="536" alt="image" src="https://github.com/user-attachments/assets/edcd3f7e-041a-4734-bc1e-4b597f7a2776" />

If we just compare the wgsl kernels across devices, we can see that they perform quite well on non-fancy hardware. AMD GPUs seem to be more picky about the scalar vs. vector CSR kernels, but it's impressive that even an integrated GPU performs really well when the kernel is chosen correctly based on the matrix density!

<img width="1552" height="481" alt="image" src="https://github.com/user-attachments/assets/5a83155e-83af-46ba-b372-63ca0ed0d85d" />
