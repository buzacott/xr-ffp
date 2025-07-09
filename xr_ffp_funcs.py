import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import xarray as xr
from FFP_Python.calc_footprint_FFP_climatology import FFP_climatology
import dask.array as da

#------------------------------------------------------------------------------#
class Suppressor():

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, exception_type, value, traceback):
        sys.stdout = self.stdout
        if exception_type is not None:
            # Do normal exception handling
            raise Exception(f"Got exception: {exception_type} {value} {traceback}")

    def write(self, x): pass

    def flush(self): pass

def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def init_array(y, x, cdata):
    # Initialise xarray that will store the footprints, as well as the input
    # data for the model
    xds = xr.Dataset(
        coords={
            'time': (['time'], cdata['datetime']),
            'y': (['y'], y),
            'x': (['x'], x)
        },
        data_vars={
            'fp': (
                ['time', 'y', 'x'],
                da.empty((cdata['datetime'].size, y.size, x.size),
                         chunks=(1, y.size, x.size),
                         dtype='float32'),
            )
        } | {c: (['time'], cdata[c].astype(np.float32)) for c in cdata.columns[1:]}
    )

    return xds

def ffp_wrapper(i, domain, dx ,dy):
    # i one timestamp of the xarray object
    # Run FFP model, if it fails output nan array
    out = i['fp'].copy()
    try:
        with Suppressor():
            fp = FFP_climatology(
                zm=i['zm'].values[0],
                z0=i['ROUGHNESS_LENGTH'].values[0],
                umean=i['WS'].values[0],
                h=i['PBLH'].values[0],
                ol=i['MO_LENGTH'].values[0],
                sigmav=i['V_SIGMA'].values[0],
                ustar=i['USTAR'].values[0],
                wind_dir=i['WD'].values[0],
                domain=domain,
                dy=dy,
                dx=dx,
                rslayer=1,
                smooth_data=1,
                verbosity=0,
                fig=False
            )
        fp = fp['fclim_2d'].astype(np.float32)
    except:
        fp = np.nan

    out = i['fp'].copy()
    out[:] = fp

    return out