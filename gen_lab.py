import os

from glob import glob
from os.path import join, basename, splitext
from nnmnkwii.io import hts
import config
from util import merge_sil


# Copy *.lab files from the original archive and round them off to 5ms(universal frame period for dnn-based voice synthesis?)

for name in ["mono", "full"]:
    files = sorted(glob(join(config.nit_song070_root, "data/labels", name, "*.lab")))
    dst_dir = join(config.out_dir, "original_" + name + "_round")
    os.makedirs(dst_dir, exist_ok=True)

    for path in files:
        lab = hts.load(path)
        name = basename(path)

        for x in range(len(lab)):
            lab.start_times[x] = round(lab.start_times[x] / 50000) * 50000
            lab.end_times[x] = round(lab.end_times[x] / 50000) * 50000

        # Check if rounding is done property
        if name == "mono":
            for i in range(len(lab)-1):
                if lab.end_times[i] != lab.start_times[i+1]:
                    print(path)
                    print(i, lab[i])
                    print(i+1, lab[i+1])
                    import ipdb; ipdb.set_trace()

        with open(join(dst_dir, name), "w") as of:
            of.write(str(lab))
