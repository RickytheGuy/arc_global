import os, glob
import json
import re
import warnings
import threading
import logging
from multiprocessing import Pool

# from tqdm.contrib.concurrent import process_map
import tqdm
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from osgeo import gdal, ogr, osr

from arc import Arc
from curve2flood import Curve2Flood_MainFunction

CWD = os.path.dirname(os.path.abspath(__file__))
ALL_DEMS = os.path.join(CWD, "DEMs_for_Entire_World")
OUTPUT_DIR = os.path.join(CWD, "output")
STREAMLINES = os.path.join(CWD, "streamlines")
LANDCOVER_DIR = os.path.join(CWD, "land_use")
BOUNDS_JSON = os.path.join(CWD, "stream_bounds.json")
MANNINGS_TABLE = os.path.join(CWD, "ESA_land_use.txt")
OCEANS_PQ = os.path.join(CWD, "seas_buffered.parquet")

RETURN_PERIODS = [2, 5, 10, 25, 50, 100]
# RETURN_PERIODS = [50, 100]
RETURN_PERIODS_ZARR_URL = "s3://rfs-v2/retrospective/return-periods.zarr"
MONTHLY_ZARR_URL = "s3://rfs-v2/retrospective/monthly-timeseries.zarr"
BUFFER_DISTANCE = 0.1
FABDEM_PATTERN = re.compile(r'([NS])(\d+)([EW])(\d+)')
STORAGE_OPTIONS={"anon": True, 'config_kwargs': {'response_checksum_validation':'when_required'}}
# OCEANS_DF = gpd.read_parquet(OCEANS_PQ)

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s',
)

def get_fabdem_in_extent(minx: float, 
                         miny: float, 
                         maxx: float, 
                         maxy: float) -> list[str]:
    output = []

    for file in glob.glob(os.path.join(ALL_DEMS, "*", "*.tif")):
        basename = os.path.basename(file)
        match = FABDEM_PATTERN.search(basename)
        if match:
            ns = 1 if match.group(1) == 'N' else -1
            ew = 1 if match.group(3) == 'E' else -1
            lat = ns * int(match.group(2))
            lon = ew * int(match.group(4))
            if minx <= lon + 1 and maxx >= lon and miny <= lat + 1 and maxy >= lat:
                output.append(file)        
    return output

def clean_stream_raster(stream_raster: str, num_passes: int = 2) -> None:
    """
    This function comes from Mike Follum's ARC at https://github.com/MikeFHS/automated-rating-curve
    """
    assert num_passes > 0, "num_passes must be greater than 0"
    
    # Get stream raster
    stream_ds: gdal.Dataset = gdal.Open(stream_raster, gdal.GA_Update)
    array: np.ndarray = stream_ds.ReadAsArray().astype(np.int64)
    
    # Create an array that is slightly larger than the STRM Raster Array
    array = np.pad(array, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    
    row_indices, col_indices = array.nonzero()
    num_nonzero = len(row_indices)
    
    for _ in range(num_passes):
        # First pass is just to get rid of single cells hanging out not doing anything
        p_count = 0
        p_percent = (num_nonzero + 1) / 100.0
        n=0
        for x in range(num_nonzero):
            if x >= p_count * p_percent:
                p_count = p_count + 1
            r = row_indices[x]
            c = col_indices[x]
            if array[r,c] <= 0:
                continue

            # Left and Right cells are zeros
            if array[r,c + 1] == 0 and array[r, c - 1] == 0:
                # The bottom cells are all zeros as well, but there is a cell directly above that is legit
                if (array[r+1,c-1]+array[r+1,c]+array[r+1,c+1])==0 and array[r-1,c]>0:
                    array[r,c] = 0
                    n=n+1
                # The top cells are all zeros as well, but there is a cell directly below that is legit
                elif (array[r-1,c-1]+array[r-1,c]+array[r-1,c+1])==0 and array[r+1,c]>0:
                    array[r,c] = 0
                    n=n+1
            # top and bottom cells are zeros
            if array[r,c]>0 and array[r+1,c]==0 and array[r-1,c]==0:
                # All cells on the right are zero, but there is a cell to the left that is legit
                if (array[r+1,c+1]+array[r,c+1]+array[r-1,c+1])==0 and array[r,c-1]>0:
                    array[r,c] = 0
                    n=n+1
                elif (array[r+1,c-1]+array[r,c-1]+array[r-1,c-1])==0 and array[r,c+1]>0:
                    array[r,c] = 0
                    n=n+1
        
        
        # This pass is to remove all the redundant cells
        n = 0
        p_count = 0
        p_percent = (num_nonzero + 1) / 100.0
        for x in range(num_nonzero):
            if x >= p_count * p_percent:
                p_count = p_count + 1
            r = row_indices[x]
            c = col_indices[x]
            value = array[r,c]
            if value<=0:
                continue

            if array[r+1,c] == value and (array[r+1, c+1] == value or array[r+1, c-1] == value):
                if array[r+1,c-1:c+2].max() == 0:
                    array[r+ 1 , c] = 0
                    n = n + 1
            elif array[r-1,c] == value and (array[r-1, c+1] == value or array[r-1, c-1] == value):
                if array[r-1,c-1:c+2].max() == 0:
                    array[r- 1 , c] = 0
                    n = n + 1
            elif array[r,c+1] == value and (array[r+1, c+1] == value or array[r-1, c+1] == value):
                if array[r-1:r+1,c+2].max() == 0:
                    array[r, c + 1] = 0
                    n = n + 1
            elif array[r,c-1] == value and (array[r+1, c-1] == value or array[r-1, c-1] == value):
                if array[r-1:r+1,c-2].max() == 0:
                        array[r, c - 1] = 0
                        n = n + 1
    
    # Write the cleaned array to the raster
    stream_ds.WriteArray(array[1:-1, 1:-1])

def get_linknos(stream_raster: str,) -> np.ndarray:
    ds: gdal.Dataset = gdal.Open(stream_raster)
    array = ds.ReadAsArray()
    linknos = np.unique(array)
    if linknos[0] == 0:
        linknos = linknos[1:]
    return linknos

def get_base_max(stream_raster: str,
                 base_max_file: str,) -> None:
    linknos = get_linknos(stream_raster)
    base_max_file = os.path.abspath(base_max_file)
    if os.path.exists(base_max_file):
        return

    # Open the return period zarr
    # Select return period 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with xr.open_zarr(RETURN_PERIODS_ZARR_URL, storage_options=STORAGE_OPTIONS) as ds:
            # Filter to only include existing values
            # existing = set(ds['river_id'].values)
            # linknos = [r for r in linknos if r in existing]

            df_max = (
                ds.sel(river_id=linknos, return_period=100)
                .to_dataframe()
                .drop(columns='return_period')
            )

    # Adjust the rp100 to be a little bigger
    df_max['logpearson3'] = df_max['logpearson3'].fillna(df_max['gumbel'])
    df_max = df_max.drop(columns='gumbel')
    df_max['logpearson3'] = df_max['logpearson3'] * 1.5 + 50

    # Now open daily zarr and get median
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_base = (
            xr.open_zarr(MONTHLY_ZARR_URL, storage_options=STORAGE_OPTIONS)
            .sel(river_id=linknos)
            .median(dim='time')
            .to_dataframe()
        )

    # Merge and save csv
    (
        df_base.merge(df_max, on='river_id')
        .rename(columns={'Q':'median', 'logpearson3':'max'})
        .round(2)
        .to_csv(base_max_file)
    )
    
def get_return_period(stream_raster: str,
                      rp: list[int], 
                      flow_dir: str,) -> None:
    linknos = get_linknos(stream_raster)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_zarr(RETURN_PERIODS_ZARR_URL, storage_options=STORAGE_OPTIONS)
        # Filter linknos to only include existing values
        # existing = set(ds['river_id'].values)
        # linknos = [r for r in linknos if r in existing]
    try:
        df = ds.sel(river_id=linknos, return_period=rp).to_dataframe()
    except KeyError:
        print(f"Return period {rp} not found in the dataset. Available return periods are {', '.join(ds['return_period'].values.astype(str))}")
        return
    
    df['logpearson3'] = df['logpearson3'].fillna(df['gumbel'])
    for r in rp:
        if os.path.exists(os.path.join(flow_dir, f"rp{r}.csv")):
            continue
        (
            df.T[r]
            .T['logpearson3']
            .round(2)
            .to_csv(os.path.join(flow_dir, f"rp{r}.csv"))
        )

def opens_right(path: str) -> bool:
    if path.endswith('.parquet'):
        try:
            pd.read_parquet(path)
            return True
        except:
            return False
        
    if path.endswith('.csv'):
        try:
            pd.read_csv(path)
            return True
        except:
            return False
        
    try:
        gdal.Open(path)
        return True
    except RuntimeError:
        return False
    
def clip_ocean(floodmap: str, oceans_array: np.ndarray) -> None:
    # We will open the floodmap in update mode, open and rasterize the applicable extent of the ocean, and then clip the ocean from the floodmap
    # Open the floodmap
    with gdal.Open(floodmap, gdal.GA_Update) as flood_ds:
        flood_array = flood_ds.ReadAsArray()

        # Clip the ocean from the floodmap
        flood_array[oceans_array == 1] = 0
        flood_ds.WriteArray(flood_array)
        flood_ds.FlushCache()

    
def process(dem: str):
    try:
        # First check if any streams intersect the DEM
        # If not, skip
        with open(BOUNDS_JSON) as f:
            bounds: dict[str, tuple[float, ...]] = json.load(f)

        # Get min/max
        with gdal.Open(dem) as src:
            minx, miny, maxx, maxy = src.GetGeoTransform()[0], src.GetGeoTransform()[3], src.GetGeoTransform()[0] + src.RasterXSize * src.GetGeoTransform()[1], src.GetGeoTransform()[3] + src.RasterYSize * src.GetGeoTransform()[5]

        if miny > maxy:
            miny, maxy = maxy, miny

        minx -= BUFFER_DISTANCE
        maxx += BUFFER_DISTANCE
        miny -= BUFFER_DISTANCE
        maxy += BUFFER_DISTANCE

        stream_files = []
        for f, bbox in bounds.items():
            if bbox[0] <= maxx and bbox[1] <= maxy and bbox[2] >= minx and bbox[3] >= miny:
                stream_files.append(os.path.join(STREAMLINES, f))
        if not stream_files:
            logging.info(f"No streams found in {dem}")
            return
        
        dem_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(dem)), os.path.basename(dem).replace('.tif', ''))
        os.makedirs(dem_dir, exist_ok=True)
        
        # Buffer first
        buffered = os.path.join(dem_dir, os.path.basename(dem).replace('.tif', '.vrt'))

        # Get all DEMs in extent
        candidates = get_fabdem_in_extent(minx, miny, maxx, maxy)
        assert dem in candidates
        if len(candidates) > 1:
            logging.info(f"Buffering {dem} with {len(candidates) - 1} other DEMs")
            vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest',
                                                    outputBounds=(minx, miny, maxx, maxy))
            gdal.BuildVRT(buffered, candidates, options=vrt_options)

            dem = buffered

        # Now rasterize
        # Load the dem
        ds: gdal.Dataset = gdal.Open(dem)
        gt = ds.GetGeoTransform()
        proj: str = ds.GetProjection()
        width = ds.RasterXSize
        height = ds.RasterYSize
        ds = None

        # Create output raster
        stream_raster = os.path.join(dem_dir, 'streams.tif')
        if not os.path.exists(stream_raster) or not opens_right(stream_raster):
            logging.info(f"Rasterizing streams for {dem}")
            raster_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(stream_raster, width, height, 1, gdal.GDT_Int32, options=['COMPRESS=LZW', 'PREDICTOR=2'])
            raster_ds.SetGeoTransform(gt)
            raster_ds.SetProjection(proj)

            # Rasterize the streams
            for streams_file in stream_files:
                stream_ds: gdal.Dataset = ogr.Open(streams_file)
                layer = stream_ds.GetLayer()
                gdal.RasterizeLayer(raster_ds, 
                                    [1], 
                                    layer, 
                                    options=[f"ATTRIBUTE=LINKNO"],)

            # Clean up
            raster_ds.FlushCache()
            raster_ds = None

        if not len(get_linknos(stream_raster)):
            logging.info(f"No streams found in {dem}")
            # No streams here
            return

        # Get base max
        base_max_file = os.path.join(dem_dir, 'base_max.csv')
        if not os.path.exists(base_max_file) or not opens_right(base_max_file):
            logging.info(f"Getting base max for {dem}")
            base_max_thread = threading.Thread(target=get_base_max, args=(stream_raster, base_max_file))
            base_max_thread.start()
        else:
            base_max_thread = None

        # Get flow files
        rp_thread = threading.Thread(target=get_return_period, args=(stream_raster, RETURN_PERIODS, dem_dir))
        rp_thread.start()

        # Clean the raster
        clean_stream_raster(stream_raster)

        # Now crop and make land cover 
        esa_df = gpd.read_file('esa_tiles.gpkg')
        bbox = box(minx, miny, maxx, maxy)
        intersecting_tiles = esa_df[esa_df.intersects(bbox)]
        tiles = set(intersecting_tiles['ll_tile'])

        landcover_files = []
        for tile in tiles:
            landcover_files.extend(glob.glob(os.path.join(LANDCOVER_DIR, f'{tile}.tif')))

        land_use_file = os.path.join(dem_dir, 'land_use.vrt')
        logging.info(f"Creating land use raster for {dem}")
        # Save landcover vrt
        options = gdal.BuildVRTOptions(outputBounds=[gt[0], gt[3] + height * gt[5], gt[0] + width * gt[1], gt[3]], 
                                            srcNodata=0, 
                                            xRes=abs(gt[1]), 
                                            yRes=abs(gt[5]),
                                            outputSRS=proj,)
        gdal.BuildVRT(land_use_file, landcover_files, options=options)

        # Create main input files
        mifns = []
        for rp in RETURN_PERIODS:
            main_input_file = os.path.join(dem_dir, f'main_input_rp{rp}.txt')
            mifns.append(main_input_file)
            with open(main_input_file, 'w') as f:
                f.write("# Main input file for ARC and Curve2Flood\n\n")

                f.write("\n# Input files - Required\n")
                f.write(f"DEM_File\t{dem}\n")
                f.write(f"Stream_File\t{stream_raster}\n")
                f.write(f"LU_Raster_SameRes\t{land_use_file}\n")
                f.write(f"LU_Manning_n\t{MANNINGS_TABLE}\n")
                f.write(f"Flow_File\t{base_max_file}\n")
                f.write(f"COMID_Flow_File\t{os.path.join(dem_dir, f'rp{rp}.csv')}\n")

                f.write("\n# Output files - Required\n")
                f.write(f"Print_VDT_Database\t{os.path.join(dem_dir, 'vdt.parquet')}\n")

                f.write("\n# Output files - Optional\n")
                f.write(f"OutFLD\t{os.path.join(dem_dir, f'flood_rp{rp}.tif')}\n")

                f.write("\n# Parameters - Required\n")
                f.write(f"Flow_File_ID\triver_id\n")
                f.write(f"Flow_File_BF\tmedian\n")
                f.write(f"Flow_File_QMax\tmax\n")
                f.write(f"Spatial_Units\tdeg\n")

                f.write("\n# Parameters - Optional\n")
                f.write(f"Low_Spot_Range\t{3}\n")
                f.write(f"TopWidthPlausibleLimit\t{5000}\n") # Matches x-section distance
                f.write(f"Flood_WaterLC_and_STRM_Cells\t{True}\n")

                f.write("\n# Optional ARC Bathymetry\n")
                f.write(f"BATHY_Out_File\t{os.path.join(dem_dir, 'bathymetry.tif')}\n")

        # Run ARC
        if base_max_thread: base_max_thread.join()
        if not os.path.exists(os.path.join(dem_dir, 'vdt.parquet')) or not opens_right(os.path.join(dem_dir, 'vdt.parquet')):
            logging.info(f"Running ARC for {dem}")
            Arc(main_input_file, True).run()

        oceans_ready = False
        rp_thread.join()

        # Run Curve2Flood
        for i, mifn in enumerate(mifns):
            floodmap = os.path.join(dem_dir, f'flood_rp{RETURN_PERIODS[i]}.tif')
            if os.path.exists(floodmap) and opens_right(floodmap):
                logging.info(f"Skipping {floodmap}")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logging.info(f"Running Curve2Flood for {dem}")
                Curve2Flood_MainFunction(mifn)

            if not oceans_ready:
                oceans_ready = True
                flood_ds: gdal.Dataset = gdal.Open(floodmap)
                gt = flood_ds.GetGeoTransform()
                proj = flood_ds.GetProjection()
                width = flood_ds.RasterXSize
                height = flood_ds.RasterYSize

                options = gdal.RasterizeOptions(format='MEM', 
                                                outputType=gdal.GDT_Byte, 
                                                burnValues=[1], 
                                                xRes=abs(gt[1]), 
                                                yRes=abs(gt[5]), 
                                                outputBounds=[gt[0], gt[3] + height * gt[5], gt[0] + width * gt[1], gt[3],], 
                                                outputSRS=proj)
                oceans_ds: gdal.Dataset = gdal.Rasterize("", OCEANS_PQ, options=options)
                oceans_array = oceans_ds.ReadAsArray()

            logging.info(f"Clipping ocean for {dem}")
            clip_ocean(floodmap, oceans_array)
            
            # return
    except:
        import traceback
        print(traceback.format_exc())


if __name__ == '__main__':
    # for dem in glob.glob(os.path.join(ALL_DEMS, "*", "*.tif"))[-16000:]:
    #     process(dem)
    #     break
    # process('/home/ec2-user/world_flood_maps/DEMs_for_Entire_World/N00W060-N10W050_FABDEM_V1-2/N00W051_FABDEM_V1-2.tif')
    files = glob.glob(os.path.join(ALL_DEMS, "*", "*.tif"))
    with tqdm.tqdm(files, desc="Processing DEMs", unit="tiles") as pbar, Pool(os.cpu_count() - 40) as pool:
        for _ in pool.imap_unordered(process, files):
            pbar.update()
        
    # process_map(process, glob.glob(os.path.join(ALL_DEMS, "*", "*.tif")), max_workers=os.cpu_count() - 36, chunksize=1, desc="Processing DEMs", unit="DEM", max_retries=3, error_callback=lambda e: logging.error(f"Error processing {e}"))
    print("DONNNNNE")





