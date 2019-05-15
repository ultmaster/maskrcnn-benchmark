# https://github.com/bamos/setGPU/blob/master/setGPU.py

import os
import gpustat
import random

stats = gpustat.GPUStatCollection.new_query()
ids = map(lambda gpu: int(gpu.entry['index']), stats)
ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
pairs = list(zip(ids, ratios))
random.shuffle(pairs)
sorted_gpu = list(map(lambda t: t[0], sorted(pairs, key=lambda x: x[1])))

print("GPU list: {}".format(",".join(map(str, sorted_gpu))))
