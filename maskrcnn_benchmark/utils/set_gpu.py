# https://github.com/bamos/setGPU/blob/master/setGPU.py

import os
import gpustat
import random

stats = gpustat.GPUStatCollection.new_query()
ids = list(map(lambda gpu: int(gpu.entry['index']), stats))
sorted_gpu = sorted(ids)

print("GPU list: {}".format(",".join(map(str, sorted_gpu))))
