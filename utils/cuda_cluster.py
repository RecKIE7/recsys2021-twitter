import sys
sys.path.append('..')

import os, time
import cudf, cupy, time, rmm
import dask as dask, dask_cudf
from dask.distributed import Client, wait, progress
from dask_cuda import LocalCUDACluster
import subprocess
import core.config as conf

workers = ', '.join([str(i) for i in range(conf.n_workers)])
os.environ["CUDA_VISIBLE_DEVICES"] = workers

cluster = LocalCUDACluster()
client = Client(cluster)
