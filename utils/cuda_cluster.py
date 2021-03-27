import os, time
import cudf, cupy, time, rmm
import dask as dask, dask_cudf
from dask.distributed import Client, wait, progress
from dask_cuda import LocalCUDACluster
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"]="0"

cluster = LocalCUDACluster()
client = Client(cluster)
client