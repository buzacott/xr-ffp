## xr-ffp

This repo contains Python scripts to calculate flux footprints with the Kljun
et al. 2015 FFP model and store them in netCDF files. The code takes advantage
of dask to calculate the footprints in parallel and to reduce memory usage by
streaming the results to disk. There are probably some optimisations that could
be made still but overall things works quite well.

The footprint model needs to be downloaded from
https://footprint.kljun.net/download.php and the `FFP_Python` folder added to
the directory for the functions to work.

Create the conda environment with: 
```
conda env create -f environment.yml
```

The example can be run with:

```
conda activate xrffp
python xr-ffp.py
```

The example creates two result files demonstrating lossless and lossy
compression. One month of half-hourly footprints with lossless compression with
the params `domain=[-300, 300, -300, 300]` and `dy=dx=1` results in a ~500MB
file, while with lossy uint16 compression the results file is only ~25MB!

The `analyse_results.ipynb` notebook quickly compares the compressed `.nc` files
and also demonstrates how to calculate the relative contribution of an area of
interest with the files.