import glob, os, shutil, logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import geopandas as gpd
from osgeo import gdal
import tqdm

gdal.UseExceptions()
# for f in glob.glob(r"E:\geoglows_v3\geoglows_v3\hydrography\vpu=*\streams_mapping_*.gpkg"):
#     dest = os.path.join(r"C:\Users\lrr43\Documents\world_flood_maps\streamlines", os.path.basename(f))
#     shutil.copy(f, dest)

CB_FMT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
ESA_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

def try_open(f) -> bool:
    try:
        ds = gdal.Open(f)
        return True
    except Exception as e:
        return False

def _download_esa_tile(tile: str, 
                      output_dir: str = '/home/ec2-user/world_flood_maps/land_use') -> str:
    output_dir = os.path.abspath(output_dir)

    # Save as a geoparquet to save disk space and read/write times
    tile_file = os.path.join(output_dir, f'{tile}.tif')
    if os.path.exists(tile_file) and try_open(tile_file):
        logging.debug(f"{tile_file} already exists")
        return tile_file
    
    # If not cached, download it
    tile_url = f"{ESA_BASE_URL}/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    
    # Use gdal to create a local copy with compression (literally saves GB of space)
    # with tqdm(total=100, desc=tile, bar_format=CB_FMT) as pbar:
    options = gdal.TranslateOptions(format='GTiff', 
                                    creationOptions=['COMPRESS=DEFLATE', 'PREDICTOR=2'],)

    gdal.Translate(tile_file, tile_url, options=options)

def thread_function(n, o, pbar):
    result = _download_esa_tile(n, o)
    pbar.update(1)  # Update progress bar after each task
    return result

if __name__ == "__main__":
    import geopandas as gpd

    ESA_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    # esa_df: gpd.GeoDataFrame = gpd.read_file(f"{ESA_BASE_URL}/esa_worldcover_grid.geojson")
    esa_df = gpd.read_file('esa_tiles.gpkg')
    # esa_df.to_file('esa_tiles.gpkg')
    tiles = set(esa_df['ll_tile'])

    max_threads = min(os.cpu_count(), len(tiles)) * 3
    print(max_threads)

    # with tqdm.tqdm(total=len(tiles), desc="Processing", unit="tile") as pbar:
    #     with ThreadPoolExecutor(max_threads) as executor:
    #         results = list(executor.map(lambda n: thread_function(n, r"C:\Users\lrr43\Documents\world_flood_maps\land_use", pbar), tiles))
    with Pool(max_threads) as p:
      r = list(tqdm.tqdm(p.imap(_download_esa_tile, tiles), total=len(tiles)))
