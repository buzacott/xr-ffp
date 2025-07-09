import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster, progress
from xr_ffp_funcs import *

# Read in input data
data = pd.read_csv(
    'data/cdata.csv',
    usecols=['datetime', 'USTAR', 'TOWER_HEIGHT', 'DISPLACEMENT_HEIGHT',
             'ROUGHNESS_LENGTH', 'WS', 'WD', 'MO_LENGTH', 'V_SIGMA', 'PBLH']
)
data['datetime'] = pd.to_datetime(data['datetime'])
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
# Add height above displacement height
data['zm'] = data['TOWER_HEIGHT'] - data['DISPLACEMENT_HEIGHT']

# Footprint parameters
domain = [-300, 300, -300, 300]
dy = 1
dx = 1
y = np.arange(domain[2], domain[3]+1, dy, dtype=np.int16)
x = np.arange(domain[0], domain[1]+1, dx, dtype=np.int16)


# Make output dir
dst = 'results'
make_dir(dst)


if __name__ == '__main__':
    
    # Set up dask cluster/client
    cluster = LocalCluster(n_workers=4,
                           threads_per_worker=1,
                           processes=True,
                           dashboard_address=None)
    client = Client(cluster)

    # Process each year-month separately
    for (year,month), cdata in data.groupby(['year', 'month']):
        
        print(f'Processing {year}-{month:02}')
        # Drop group cols, not needed
        cdata = cdata.drop(['year', 'month'], axis=1)

        fname = os.path.join(dst, f'{year}-{month:02}_ffp.nc')

        try:
            # Initialise array, copy template, and setup job to calc footprints
            xds = init_array(y, x, cdata)
            template = xds['fp']
            xds['fp'] = xds.map_blocks(ffp_wrapper,
                                       template=template,
                                       kwargs={"domain": domain,
                                               "dy": dy,
                                               "dx": dx})
            
            # Write full file to disk to reduce memory usage
            write_job = xds.to_netcdf(fname,
                                      mode='w',
                                      engine='netcdf4',
                                      compute=False)
            
            # Start job and monitor progress
            write_job = write_job.persist()
            progress(write_job)
            write_job.compute()

        except KeyboardInterrupt:
                sys.exit(0)
    
        print(f'Compressing...')
        # Compression could be streamed to disk during the mapping but in my
        # testing it was very slow
        
        # Load the dataset in fully
        with xr.open_dataset(fname) as ds:
            xds = ds.load()

        #-------------------#
        # Lossless encoding #
        #-------------------#
        # Uncompressed month of footprints is around 2GB with the example
        # parameters (dx=dy=1, x=y=601)
        # With lossless compression that goes to around 500MB
        encoding = {
            "fp": {
                "zlib": True
            }
        }
        
        # Write, overwrite uncompressed file
        xds.to_netcdf(fname,
                      mode='w',
                      engine='netcdf4',
                      encoding=encoding)

        #----------------#
        # Lossy encoding #
        #----------------#
        # Lossy encoding by converting footprints to uint16 and applying zlib
        # Using uint16 since footprint data is all >= 0
        # Valid range is 0 to 2**16 - 2
        # Missing value is set to 2 ** 16 - 1
        # Error quite low overall, ~<0.005 in fp values
        # In weighted area extraction, ~<0.1%
        # Compressed file: ~25MB!

        # Get min and max of data
        fp_min = np.nanmin(xds.fp.values)
        fp_max = np.nanmax(xds.fp.values)
        add_offset = fp_min
        scale_factor = (fp_max - fp_min) / (2**16 - 2)

        encoding = {
            "fp": {
                "dtype": "uint16",
                "scale_factor": scale_factor,
                "add_offset": add_offset,
                'missing_value': 2**16 - 1,
                '_FillValue': 2**16 - 1,
                "zlib": True
            }
        }
        # Write
        xds.to_netcdf(os.path.join(dst, f'{year}-{month:02}_ffp_lossy.nc'),
                      mode='w',
                      engine='netcdf4',
                      encoding=encoding)

        xds.close()
    
    client.close()
    cluster.close()