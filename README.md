# nnsvs_nit_song070

This is the collection of helper scripts to make use of [NIT_SONG070 training demo of HTS](http://hts.sp.nitech.ac.jp/?Download) from DNN-based singing voice synthesis (SVS) systems. Almost all codes are derived from [kiritan_singing](https://github.com/r9y9/kiritan_singing).

This is my study work to inspect the data preparation flow of [NNSVS](https://github.com/r9y9/nnsvs) and may not be useful for almost anyone.


## Requirements

- nnmnkwii: https://github.com/r9y9/nnmnkwii
- soundfile
- numpy
- tqdm
- jaconv

## How to use

This repository does not include audio files. To generate data, please download HTS-demo_NIT-SONG070-F001.tar.bz2 by yourself and change `nit_song070_root` in `common.py` based on your environment. Then, please run:

```
run.sh
```

The directory tree generated by this script is the same as kiritan_singing does.
