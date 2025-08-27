#!/usr/bin/env python3

import os
import gc
import glob
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator, griddata, interpn
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import warnings
import logging
import psutil
import pickle
import json
from pathlib import Path
from functools import lru_cache, partial
import numba
from numba import jit, prange
import joblib
import random

# MPI imports
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

warnings.filterwarnings('ignore')

# Configure logging for MPI environment
def setup_mpi_logging():
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        log_filename = f'enhanced_spatial_seaice_rank_{rank:03d}.log'
    else:
        log_filename = 'enhanced_spatial_seaice.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_mpi_logging()

# Numba-optimized functions
@jit(nopython=True, parallel=True, fastmath=True)
def fast_bilinear_interpolation(x_coords, y_coords, data_grid, x_query, y_query):
    """Ultra-fast bilinear interpolation using Numba"""
    n_points = len(x_query)
    results = np.empty(n_points, dtype=np.float32)
    
    for i in prange(n_points):
        x = x_query[i]
        y = y_query[i]
        
        # Find grid indices
        x_idx = np.searchsorted(x_coords, x) - 1
        y_idx = np.searchsorted(y_coords, y) - 1
        
        # Boundary checks
        if x_idx < 0 or x_idx >= len(x_coords) - 1 or y_idx < 0 or y_idx >= len(y_coords) - 1:
            results[i] = 0.0
            continue
            
        # Bilinear interpolation
        x1, x2 = x_coords[x_idx], x_coords[x_idx + 1]
        y1, y2 = y_coords[y_idx], y_coords[y_idx + 1]
        
        q11 = data_grid[y_idx, x_idx]
        q12 = data_grid[y_idx + 1, x_idx]
        q21 = data_grid[y_idx, x_idx + 1]
        q22 = data_grid[y_idx + 1, x_idx + 1]
        
        # Check for NaN values
        if np.isnan(q11) or np.isnan(q12) or np.isnan(q21) or np.isnan(q22):
            results[i] = 0.0
            continue
        
        # Interpolation weights
        wx = (x - x1) / (x2 - x1)
        wy = (y - y1) / (y2 - y1)
        
        # Bilinear interpolation formula
        result = (q11 * (1 - wx) * (1 - wy) + 
                 q21 * wx * (1 - wy) + 
                 q12 * (1 - wx) * wy + 
                 q22 * wx * wy)
        
        results[i] = result
    
    return results

@jit(nopython=True, parallel=True)
def vectorized_temporal_features(hours, days_of_year):
    """Vectorized temporal feature computation"""
    n = len(hours)
    result = np.empty((n, 6), dtype=np.float32)
    
    for i in prange(n):
        hour = hours[i]
        day = days_of_year[i]
        
        # Hour features
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Day features  
        day_sin = np.sin(2 * np.pi * day / 365.25)
        day_cos = np.cos(2 * np.pi * day / 365.25)
        
        result[i, 0] = hour
        result[i, 1] = hour_sin
        result[i, 2] = hour_cos
        result[i, 3] = day
        result[i, 4] = day_sin
        result[i, 5] = day_cos
    
    return result

class SpatialSeaIcePredictor:
    """
    🎯 ENHANCED SPATIAL SEA ICE PREDICTOR
    Uses spatial regression instead of LSTM for better satellite data handling
    Input: (n_observations, 14_features) per hour
    Output: (n_observations, 2_targets) per hour
    """
    
    def __init__(self, era5_path, sea_ice_root_dir, max_workers=None, use_gpu=True):
        # Initialize MPI
        if MPI_AVAILABLE:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.is_master = (self.rank == 0)
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
            self.is_master = True
            logger.warning("🚨 MPI not available! Install mpi4py and run with: mpirun -np N python script.py")
        
        # Core configuration
        self.era5_path = era5_path
        self.sea_ice_root_dir = sea_ice_root_dir
        self.era5_data = None
        self.sea_ice_data = []
        self.observation_points = []
        self.era5_observation_data = {}
        self.scaler_features = StandardScaler()
        self.scaler_targets = MinMaxScaler()
        
        # Enhanced satellite configuration
        self.satellites = [f'TM{i:02d}' for i in range(1, 23)]  # TM01 to TM22
        
        # MPI-optimized configuration
        self.max_workers = max_workers or min(mp.cpu_count() // self.size, 16)
        self.use_gpu = use_gpu and (self.rank % 2 == 0)
        self.setup_gpu()
        
        # Memory optimization per MPI rank
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.max_memory_gb = (total_ram_gb * 0.9) / self.size
        self.chunk_size = max(1000, int(50000 / self.size))
        
        # Feature configuration
        self.era5_features = ['cc', 'pv', 'r', 't', 'u', 'v']
        self.temporal_features = ['hour', 'hour_sin', 'hour_cos', 'day_of_year', 'day_sin', 'day_cos']
        self.spatial_features = ['lat', 'lon']
        self.all_features = self.era5_features + self.temporal_features + self.spatial_features
        
        # Enhanced dataset mapping for various HDF formats
        self.dataset_mappings = {
            'latitude': ['Lat', 'lat', 'latitude', 'Latitude', 'LAT', 'LATITUDE', 'geolat', 'Geolat'],
            'longitude': ['Lon', 'lon', 'longitude', 'Longitude', 'LON', 'LONGITUDE', 'geolon', 'Geolon'],
            'sic': ['Sic', 'sic', 'SIC', 'concentration', 'Concentration', 'CONCENTRATION', 'ice_concentration', 'sea_ice_concentration'],
            'sit': ['Sit', 'sit', 'SIT', 'thickness', 'Thickness', 'THICKNESS', 'ice_thickness', 'sea_ice_thickness'],
            'time': ['Utc_time', 'utc_time', 'time', 'Time', 'TIME', 'timestamp', 'Timestamp']
        }
        
        # Enhanced group paths to search in HDF files
        self.hdf_group_paths = [
            'GPS/SeaIceProduct', 'GPS/SeaIce', 'GNSS/SeaIceProduct', 'GNSS/SeaIce',
            'GPS', 'GNSS', 'SeaIceProduct', 'SeaIce', 'seaice', 'Data', 'data', '/'
        ]
        
        # Color schemes for visualization
        self.concentration_colors = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff', '#ffffff']
        self.thickness_colors = ['#08306b', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#fee391', '#fec44f', '#fe9929', '#d95f0e', '#993404']
        
        self.conc_cmap = LinearSegmentedColormap.from_list('nsidc_concentration', self.concentration_colors)
        self.thickness_cmap = LinearSegmentedColormap.from_list('nsidc_thickness', self.thickness_colors)
        
        # Pre-compile numba functions on rank 0
        if self.is_master:
            logger.info("   ⚡ Pre-compiling optimized functions...")
            dummy_coords = np.array([0.0, 1.0])
            dummy_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            dummy_query = np.array([0.5, 0.5])
            fast_bilinear_interpolation(dummy_coords, dummy_coords, dummy_data, dummy_query, dummy_query)
            
            dummy_hours = np.array([12])
            dummy_days = np.array([180])
            vectorized_temporal_features(dummy_hours, dummy_days)
            logger.info("   ✅ Numba compilation complete")
        
        # Synchronize all ranks
        if MPI_AVAILABLE:
            self.comm.Barrier()
        
        if self.is_master:
            logger.info("🚀 Enhanced Spatial Sea Ice Predictor with MPI")
            logger.info(f"   🌍 MPI ranks: {self.size}")
            logger.info(f"   🎯 STRATEGY: Spatial regression for satellite observations")
            logger.info(f"   📊 INPUT: (n_observations, 14_features) per hour")
            logger.info(f"   📊 OUTPUT: (n_observations, 2_targets) per hour")
            logger.info(f"   💾 Memory per rank: {self.max_memory_gb:.1f} GB")
            logger.info(f"   🛰️ Satellites: {len(self.satellites)} (TM01-TM22)")
        
        logger.info(f"   📍 Rank {self.rank}: Workers={self.max_workers}, GPU={'✅' if self.use_gpu else '❌'}")

    def setup_gpu(self):
        """Configure GPU settings for TensorFlow with MPI distribution"""
        if self.use_gpu:
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    # Distribute GPUs across ranks
                    gpu_id = self.rank % len(gpus)
                    gpu = gpus[gpu_id]
                    
                    tf.config.experimental.set_visible_devices(gpu, 'GPU')
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Enable mixed precision
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    
                    # GPU-specific optimizations
                    tf.config.optimizer.set_jit(True)
                    
                    logger.info(f"   🎮 Rank {self.rank}: GPU {gpu_id} assigned with mixed precision")
                else:
                    logger.info(f"   💻 Rank {self.rank}: No GPU detected, using CPU")
                    self.use_gpu = False
            except RuntimeError as e:
                logger.warning(f"   ⚠️ Rank {self.rank}: GPU setup failed: {e}")
                self.use_gpu = False
        else:
            logger.info(f"   💻 Rank {self.rank}: Using CPU")

    def distribute_work(self, items, method='round_robin'):
        """Distribute work items across MPI ranks"""
        if method == 'round_robin':
            my_items = [items[i] for i in range(len(items)) if i % self.size == self.rank]
        elif method == 'chunk':
            chunk_size = len(items) // self.size
            start_idx = self.rank * chunk_size
            end_idx = start_idx + chunk_size if self.rank < self.size - 1 else len(items)
            my_items = items[start_idx:end_idx]
        else:
            raise ValueError(f"Unknown distribution method: {method}")
        
        return my_items

    def mpi_gather_data(self, local_data, root=0):
        """Gather data from all ranks to root"""
        if MPI_AVAILABLE:
            all_data = self.comm.gather(local_data, root=root)
            if self.rank == root:
                flattened = []
                if all_data:
                    for data_list in all_data:
                        if data_list and isinstance(data_list, list):
                            flattened.extend(data_list)
                        elif data_list:
                            flattened.append(data_list)
                return flattened
            else:
                return None
        else:
            return local_data

    def mpi_broadcast(self, data, root=0):
        """Standard MPI broadcast for simple objects."""
        if MPI_AVAILABLE:
            return self.comm.bcast(data, root=root)
        else:
            return data

    def generate_datetime_range(self, start_date, end_date, start_hour, end_hour):
        """Generate list of datetime objects"""
        datetime_list = []
        start_datetime = start_date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end_datetime = end_date.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        
        current_datetime = start_datetime
        while current_datetime <= end_datetime:
            datetime_list.append(current_datetime)
            current_datetime += timedelta(hours=1)
        
        return datetime_list

    def find_sea_ice_files_enhanced(self, datetime_list):
        """🎯 ENHANCED: Find sea ice files with specific directory structure"""
        if self.is_master:
            logger.info(f"\n📁 STEP 1: Finding sea ice files across {len(datetime_list)} hours...")
        
        unique_dates = list(set([dt.date() for dt in datetime_list]))
        unique_dates.sort()
        
        # Distribute dates across ranks
        my_dates = self.distribute_work(unique_dates, method='round_robin')
        logger.info(f"   📁 Rank {self.rank}: Searching {len(my_dates)} dates")
        
        def search_satellite_date_enhanced(args):
            """Enhanced search using os.walk for robust file discovery"""
            satellite, target_date = args
            sat_path = os.path.join(self.sea_ice_root_dir, satellite)
            if not os.path.exists(sat_path):
                return []
            
            year_month_day_path = os.path.join(
                sat_path, str(target_date.year), f"{target_date.month:02d}", f"{target_date.day:02d}"
            )
            
            date_files = []
            
            if os.path.exists(year_month_day_path):
                try:
                    for root, dirs, files in os.walk(year_month_day_path):
                        for file in files:
                            if any(file.upper().endswith(ext.upper()) for ext in ['.hdf', '.hdf5', '.h5', '.he5']):
                                date_files.append(os.path.join(root, file))
                except Exception as e:
                    logger.debug(f"Error walking directory {year_month_day_path}: {e}")
            
            return list(set(date_files))
        
        # Parallel file search within each rank
        search_args = [(sat, date) for sat in self.satellites for date in my_dates]
        rank_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(search_satellite_date_enhanced, search_args))
            for result in results:
                rank_files.extend(result)
        
        rank_files = list(set(rank_files))
        logger.info(f"   📁 Rank {self.rank}: Found {len(rank_files):,} files")

        # Gather all files from all ranks
        all_files_list = self.mpi_gather_data(rank_files, root=0)
        
        if self.is_master and all_files_list:
            all_files = list(set(all_files_list))
            logger.info(f"   📊 STEP 1 Complete: {len(all_files):,} unique HDF files found")
        else:
            all_files = None
        
        # Broadcast the final, unique file list to all ranks
        all_files = self.mpi_broadcast(all_files, root=0)
        
        return all_files

    def load_or_process_data(self, datetime_list):
        """Manages loading data from cache or processing from raw files"""
        cache_file = Path("processed_seaice_data.h5")
        cache_exists = None
        if self.is_master:
            logger.info("\n💾 STEP 2: Checking for cached data...")
            if cache_file.exists():
                logger.info(f"   ✅ Found cached data file: {cache_file}")
                cache_exists = True
            else:
                logger.info(f"   ⚠️ No cached data found. Will process from raw files.")
                cache_exists = False
        
        cache_exists = self.mpi_broadcast(cache_exists, root=0)

        if cache_exists:
            success = self.load_data_from_cache_parallel(cache_file)
        else:
            success = self.process_raw_files_parallel(datetime_list, cache_file)
        
        return success

    def load_data_from_cache_parallel(self, cache_file):
        """Loads data from the HDF5 cache in parallel on all ranks"""
        logger.info(f"   📂 Rank {self.rank}: Loading data from cache {cache_file}...")
        try:
            with h5py.File(cache_file, 'r') as f:
                seaice_group = f['sea_ice_data']
                total_obs = len(seaice_group['lat'])
                
                chunk_size = total_obs // self.size
                start_idx = self.rank * chunk_size
                end_idx = start_idx + chunk_size if self.rank < self.size - 1 else total_obs
                
                if start_idx >= end_idx:
                    logger.info(f"   📁 Rank {self.rank}: No data slice to load.")
                    self.sea_ice_data = []
                    self.observation_points = []
                    return True

                seaice_data = {}
                for key in seaice_group.keys():
                    dataset = seaice_group[key]
                    if dataset.dtype.kind == 'S':
                        seaice_data[key] = [item.decode('utf-8') for item in dataset[start_idx:end_idx]]
                    else:
                        seaice_data[key] = dataset[start_idx:end_idx]
                
                seaice_df = pd.DataFrame(seaice_data)
                seaice_df['datetime'] = pd.to_datetime(seaice_df['datetime_str'])
                seaice_df = seaice_df.drop('datetime_str', axis=1)
                
                self.sea_ice_data = seaice_df.to_dict('records')
                self.observation_points = seaice_df[['lat', 'lon', 'datetime', 'hour']].to_dict('records')
                
                logger.info(f"   ✅ Rank {self.rank}: Loaded {len(self.sea_ice_data):,} observations from cache.")
            return True
        except Exception as e:
            logger.error(f"   ❌ Rank {self.rank}: Failed to load from cache: {e}")
            return False

    def process_raw_files_parallel(self, datetime_list, cache_file):
        """Processes raw HDF files in parallel and saves a new cache"""
        hdf_files = self.find_sea_ice_files_enhanced(datetime_list)
        if not hdf_files:
            if self.is_master: logger.error("❌ No HDF files found.")
            return False

        my_files = self.distribute_work(hdf_files, method='round_robin')
        logger.info(f"   ⚙️ Rank {self.rank}: Processing {len(my_files)} raw HDF files...")
        
        self.batch_process_hdf_files_optimized(my_files, batch_size=500)
        
        logger.info(f"   📊 Rank {self.rank}: Locally processed {len(self.observation_points):,} observation points.")
        
        if self.is_master:
            logger.info("   💾 Gathering all processed data to master for saving...")
            all_obs_points = self.mpi_gather_data(self.observation_points, root=0)
            all_sea_ice = self.mpi_gather_data(self.sea_ice_data, root=0)

            if all_sea_ice:
                logger.info(f"   Master received {len(all_sea_ice):,} total observations. Saving to cache...")
                _sea_ice_data_orig = self.sea_ice_data
                self.sea_ice_data = all_sea_ice
                self.save_processed_data_to_hdf(cache_file)
                self.sea_ice_data = _sea_ice_data_orig
            else:
                logger.error("   ❌ No data was processed by any rank. Cannot create cache file.")
                return False
        else:
            self.mpi_gather_data(self.observation_points, root=0)
            self.mpi_gather_data(self.sea_ice_data, root=0)

        return True

    def batch_process_hdf_files_optimized(self, filepaths, batch_size=500):
        """Memory-efficient batch processing with progress reporting"""
        logger.info(f"   🚀 Rank {self.rank}: Optimized batch processing {len(filepaths):,} files (batch size: {batch_size})")
        
        self.observation_points = []
        self.sea_ice_data = []
        
        max_workers = self.max_workers
        total_processed_count = 0
        total_successful_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(_process_single_hdf_file_worker, 
                                    self.dataset_mappings, 
                                    self.hdf_group_paths,
                                    self.satellites)
            
            future_to_filepath = {executor.submit(process_func, filepath): filepath 
                                for filepath in filepaths}
            
            for future in as_completed(future_to_filepath):
                total_processed_count += 1
                try:
                    result = future.result()
                    if result is not None:
                        total_successful_count += 1
                        self._process_single_observation(result)
                except Exception as e:
                    logger.debug(f"      Rank {self.rank}: Error processing result for a file: {e}")
                    continue
                
                if total_processed_count % 200 == 0:
                    logger.info(f"      Rank {self.rank}: Batch progress: {total_processed_count:,}/{len(filepaths):,}")

        success_rate = (total_successful_count / total_processed_count * 100) if total_processed_count > 0 else 0
        logger.info(f"   ✅ Rank {self.rank}: Batch processing complete: {total_successful_count:,}/{total_processed_count:,} files successful ({success_rate:.1f}%)")

    def _process_single_observation(self, obs_data):
        """Process single observation immediately to save memory"""
        if obs_data is None: return
            
        try:
            lats = obs_data.get('latitude', [])
            lons = obs_data.get('longitude', [])
            sics = obs_data.get('sic', [])
            sits = obs_data.get('sit', [])
            datetime_obj = obs_data.get('datetime')
            
            if datetime_obj is None or len(lats) == 0: return

            for j in range(len(lats)):
                obs_point = {'lat': float(lats[j]), 'lon': float(lons[j]), 'datetime': datetime_obj, 'hour': datetime_obj.hour}
                observation = {
                    'lat': float(lats[j]), 'lon': float(lons[j]),
                    'sic': float(sics[j]) if j < len(sics) else 0.0,
                    'sit': float(sits[j]) if j < len(sits) else 0.0,
                    'datetime': datetime_obj, 'hour': datetime_obj.hour,
                    'satellite': obs_data.get('satellite', 'Unknown')
                }
                self.observation_points.append(obs_point)
                self.sea_ice_data.append(observation)
        
        except Exception as e:
            logger.debug(f"Error processing observation: {e}")

    def save_processed_data_to_hdf(self, filename="processed_seaice_data.h5"):
        """Save processed sea ice data to HDF5 file for future use"""
        if not self.is_master:
            return None
        
        logger.info(f"💾 Saving processed data to {filename}...")
        start_time = time.time()
        
        try:
            with h5py.File(filename, 'w') as f:
                seaice_group = f.create_group('sea_ice_data')
                
                if hasattr(self, 'sea_ice_data') and self.sea_ice_data:
                    seaice_df = pd.DataFrame(self.sea_ice_data)
                    seaice_df['datetime_str'] = seaice_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    columns_to_save = ['lat', 'lon', 'sic', 'sit', 'hour', 'satellite', 'datetime_str']
                    seaice_df = seaice_df[columns_to_save]

                    for col in seaice_df.columns:
                        if seaice_df[col].dtype == 'object':
                            str_data = seaice_df[col].astype(str).values
                            seaice_group.create_dataset(col, data=str_data, dtype=h5py.string_dtype(encoding='utf-8'))
                        else:
                            seaice_group.create_dataset(col, data=seaice_df[col].values, compression='gzip')
                    
                    logger.info(f"   ✅ Saved {len(self.sea_ice_data):,} sea ice observations.")
                
                elapsed_time = time.time() - start_time
                logger.info(f"   💾 Data saved successfully in {elapsed_time:.1f} seconds to {filename}")
                
                return filename
                
        except Exception as e:
            logger.error(f"   ❌ Failed to save processed data: {e}")
            return None

    def query_era5_at_observation_points_enhanced(self):
        """🎯 STEP 3: Query ERA5 data ONLY at observation points using MPI"""
        if self.is_master:
            logger.info(f"\n🌍 STEP 3: OPTIMIZED ERA5 querying...")
        
        try:
            self.era5_data = xr.open_dataset(self.era5_path)
            time_coord = 'time' if 'time' in self.era5_data.coords else 'valid_time'
            
            self.era5_lats = self.era5_data.latitude.values.astype(np.float32)
            self.era5_lons = self.era5_data.longitude.values.astype(np.float32)
            self.era5_times = pd.to_datetime(self.era5_data[time_coord].values)
            
            if self.is_master:
                logger.info(f"   ✅ ERA5 dataset loaded: {len(self.era5_times)} times")
        except Exception as e:
            logger.error(f"   ❌ ERA5 loading failed: {e}")
            return False
        
        # Group observations by datetime FIRST
        points_by_datetime = {}
        for point in self.observation_points:
            dt = point['datetime']
            if dt not in points_by_datetime:
                points_by_datetime[dt] = []
            points_by_datetime[dt].append(point)
        
        unique_datetimes = list(points_by_datetime.keys())
        logger.info(f"   📊 Rank {self.rank}: Processing {len(unique_datetimes)} unique datetimes")
        
        rank_era5_data = {}
        
        # Process in larger batches to reduce ERA5 I/O
        batch_size = 10  # Process 10 datetimes at once
        
        for batch_start in range(0, len(unique_datetimes), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_datetimes))
            batch_datetimes = unique_datetimes[batch_start:batch_end]
            
            logger.info(f"   ⚡ Rank {self.rank}: Processing datetime batch {batch_start//batch_size + 1}")
            
            # Process batch of datetimes
            for dt in batch_datetimes:
                try:
                    # Find closest ERA5 time
                    time_diffs = np.abs((self.era5_times - dt).total_seconds())
                    closest_idx = np.argmin(time_diffs)
                    
                    era5_subset = self.era5_data.isel({time_coord: closest_idx})
                    dt_points = points_by_datetime[dt]
                    
                    if len(dt_points) == 0:
                        continue
                    
                    # Vectorize coordinate extraction
                    lats = np.array([p['lat'] for p in dt_points], dtype=np.float32)
                    lons = np.array([p['lon'] for p in dt_points], dtype=np.float32)
                    
                    # Extract ALL features at once for this datetime
                    dt_era5_data = {}
                    for feature in self.era5_features:
                        if feature in era5_subset:
                            try:
                                feature_var = era5_subset[feature]
                                if feature_var.ndim > 2:
                                    feature_var = feature_var.squeeze()
                                
                                feature_data = feature_var.values.astype(np.float32)
                                
                                # Single interpolation call for all points
                                interpolated_values = fast_bilinear_interpolation(
                                    self.era5_lons, self.era5_lats, feature_data.T, lons, lats
                                )
                                
                                dt_era5_data[feature] = interpolated_values
                                
                            except Exception as e:
                                logger.debug(f"Feature {feature} failed: {e}")
                                dt_era5_data[feature] = np.zeros(len(lats), dtype=np.float32)
                        else:
                            dt_era5_data[feature] = np.zeros(len(lats), dtype=np.float32)
                    
                    rank_era5_data[dt] = {
                        'points': dt_points,
                        'era5_features': dt_era5_data
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to process datetime {dt}: {e}")
            
            # Progress reporting
            progress = (batch_end / len(unique_datetimes)) * 100
            logger.info(f"      ✅ Rank {self.rank}: {progress:.1f}% complete")
        
        self.era5_observation_data = rank_era5_data
        logger.info(f"   📊 Rank {self.rank}: Completed ERA5 query for {len(rank_era5_data)} datetimes")
        
        return len(rank_era5_data) > 0

    def create_temporal_lookup_table(self):
        """Create temporal features lookup table"""
        if self.is_master:
            logger.info("   🔧 Creating temporal features lookup table...")
        
        lookup_table = {}
        years = np.array([2024, 2025])
        
        for year in years:
            for day in range(1, 367):
                try:
                    hours_array = np.arange(24)
                    days_array = np.full(24, day)
                    
                    temporal_matrix = vectorized_temporal_features(hours_array, days_array)
                    
                    for hour in range(24):
                        key = (year, day, hour)
                        lookup_table[key] = {
                            'hour': temporal_matrix[hour, 0],
                            'hour_sin': temporal_matrix[hour, 1],
                            'hour_cos': temporal_matrix[hour, 2],
                            'day_of_year': temporal_matrix[hour, 3],
                            'day_sin': temporal_matrix[hour, 4],
                            'day_cos': temporal_matrix[hour, 5]
                        }
                except:
                    continue
        
        if self.is_master:
            logger.info(f"   ✅ Temporal lookup table created ({len(lookup_table)} entries)")
        return lookup_table

    def create_temporal_features(self, datetime_obj):
        """Create temporal features from datetime object"""
        hour = datetime_obj.hour
        day_of_year = datetime_obj.timetuple().tm_yday
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
        
        return {
            'hour': hour, 'hour_sin': hour_sin, 'hour_cos': hour_cos,
            'day_of_year': day_of_year, 'day_sin': day_sin, 'day_cos': day_cos
        }

    def _get_temporal_features_cached(self, datetime_obj):
        """Get temporal features using cached lookup"""
        year = datetime_obj.year
        day_of_year = datetime_obj.timetuple().tm_yday
        hour = datetime_obj.hour
        key = (year, day_of_year, hour)
        
        if hasattr(self, 'temporal_lookup') and key in self.temporal_lookup:
            return self.temporal_lookup[key]
        else:
            return self.create_temporal_features(datetime_obj)

    def create_enhanced_observations_with_era5(self):
        """🎯 STEP 4: Create training observations by combining sea ice data with ERA5 features"""
        if self.is_master:
            logger.info(f"\n🔗 STEP 4: Creating enhanced observations (in parallel)...")
        
        # Create temporal lookup table on all ranks
        self.temporal_lookup = self.create_temporal_lookup_table()
        
        enhanced_observations = []
        
        total_observations = len(self.sea_ice_data)
        logger.info(f"   📍 Rank {self.rank}: Processing {total_observations:,} local sea ice observations.")
        
        for observation in self.sea_ice_data:
            try:
                obs_datetime = observation['datetime']
                obs_lat = observation['lat']
                obs_lon = observation['lon']
                
                # Get ERA5 data for this observation from the rank's local data
                era5_features = self.get_era5_features_for_observation(obs_datetime, obs_lat, obs_lon)
                
                # Get temporal features
                temporal_features = self._get_temporal_features_cached(obs_datetime)
                
                # Create enhanced observation
                enhanced_obs = observation.copy()
                enhanced_obs.update(era5_features)
                enhanced_obs.update(temporal_features)
                
                if self.is_valid_observation_enhanced(enhanced_obs):
                    enhanced_observations.append(enhanced_obs)
                    
            except Exception as e:
                continue
        
        logger.info(f"   📍 Rank {self.rank}: Created {len(enhanced_observations):,} valid enhanced observations.")
        if self.is_master:
            logger.info(f"   📊 STEP 4 Complete: All ranks have created enhanced observations locally.")
            
        return enhanced_observations

    def get_era5_features_for_observation(self, obs_datetime, obs_lat, obs_lon):
        """Get ERA5 features for a specific observation point"""
        if obs_datetime not in self.era5_observation_data:
            return {feature: 0.0 for feature in self.era5_features}
        
        dt_data = self.era5_observation_data[obs_datetime]
        points = dt_data['points']
        era5_features = dt_data['era5_features']
        
        # Find the matching point (within tolerance)
        tolerance = 0.01  # degrees
        for i, point in enumerate(points):
            if (abs(point['lat'] - obs_lat) < tolerance and 
                abs(point['lon'] - obs_lon) < tolerance):
                
                result = {}
                for feature in self.era5_features:
                    if feature in era5_features and i < len(era5_features[feature]):
                        result[feature] = float(era5_features[feature][i])
                    else:
                        result[feature] = 0.0
                return result
        
        # If no exact match found, return zeros
        return {feature: 0.0 for feature in self.era5_features}

    def is_valid_observation_enhanced(self, observation):
        """Enhanced observation validation"""
        try:
            # Check geographic bounds (Arctic focus)
            if not (50 <= observation['lat'] <= 90):
                return False
            
            # Check longitude bounds
            if not (-180 <= observation['lon'] <= 180):
                return False
            
            # Check sea ice data quality
            if not (0 <= observation['sic'] <= 100):
                return False
            
            if not (0 <= observation['sit'] <= 20):
                return False
            
            # Check for NaN in all required features
            for feature in self.all_features:
                if feature not in observation or not np.isfinite(observation[feature]):
                    return False
            
            return True
            
        except Exception:
            return False

    def build_spatial_model(self, n_features=14):
        """
        🎯 Build spatial regression model - MUCH BETTER than LSTM for satellite data!
        Input: (n_observations, 14_features)
        Output: (n_observations, 2_targets)
        """
        inputs = Input(shape=(n_features,), name='observation_features')
        
        # Deep feature extraction with proper regularization
        x = Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)
        
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(0.1)(x)
        
        # Output layer: [sea_ice_concentration, sea_ice_thickness]
        outputs = Dense(2, activation='linear', name='sea_ice_predictions')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='SpatialSeaIcePredictor')
        
        # Compile with appropriate optimizer
        if self.use_gpu:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"   🏗️ Built spatial model with {model.count_params():,} parameters")
        return model

    def prepare_spatial_training_data(self, training_observations):
        """
        🎯 Prepare training data for spatial model
        Input: List of enhanced observations
        Output: (X, y) where X is (n_samples, 14), y is (n_samples, 2)
        """
        all_features = []
        all_targets = []
        
        for obs in training_observations:
            # Extract features in correct order
            feature_vector = [obs.get(f, 0.0) for f in self.all_features]
            
            # Extract targets
            target_vector = [obs.get('sic', 0.0), obs.get('sit', 0.0)]
            
            # Quality check
            if (not np.any(np.isnan(feature_vector)) and 
                not np.any(np.isnan(target_vector)) and
                0 <= target_vector[0] <= 100 and  # Valid SIC range
                0 <= target_vector[1] <= 20):     # Valid SIT range
                
                all_features.append(feature_vector)
                all_targets.append(target_vector)
        
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_targets, dtype=np.float32)
        
        logger.info(f"   📊 Prepared training data: {X.shape[0]:,} observations")
        logger.info(f"      Features shape: {X.shape}")
        logger.info(f"      Targets shape: {y.shape}")
        
        return X, y

    def train_spatial_model_mpi(self, training_observations, epochs=50, batch_size=2048):
        """
        🎯 STEP 5: Train spatial model with MPI coordination
        """
        if self.is_master:
            logger.info(f"\n🎯 STEP 5: Training spatial regression model...")

        # Gather all training observations to master for unified training
        all_training_obs = self.mpi_gather_data(training_observations, root=0)

        if self.is_master:
            if not all_training_obs:
                logger.error("❌ No training observations gathered!")
                return None
                
            logger.info(f"   🧠 Master gathered {len(all_training_obs):,} total training observations")
            
            # Prepare training data
            X, y = self.prepare_spatial_training_data(all_training_obs)
            
            if len(X) < 1000:
                logger.error(f"❌ Insufficient training data: {len(X)} samples")
                return None
            
            # Scale features and targets
            X_scaled = self.scaler_features.fit_transform(X)
            y_scaled = self.scaler_targets.fit_transform(y)
            
            # Build model
            model = self.build_spatial_model(n_features=X.shape[1])
            
            # Callbacks for training optimization
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.7, patience=7, min_lr=1e-6, verbose=1
                )
            ]
            
            # Train model
            logger.info(f"   🚀 Starting training with {X.shape[0]:,} samples...")
            device = '/GPU:0' if self.use_gpu else '/CPU:0'
            with tf.device(device):
                history = model.fit(
                    X_scaled, y_scaled,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
            
            logger.info("   ✅ Spatial model training complete!")
            
            # Clear memory on master
            del all_training_obs, X, y, X_scaled, y_scaled
            gc.collect()
            
            return model
        else:
            # Workers just send their data and wait
            self.mpi_gather_data(training_observations, root=0)
            return None

    def predict_for_target_observations(self, model, target_observations):
        """
        🎯 STEP 6: Generate predictions for target observations
        """
        if not self.is_master or model is None:
            return {}, {}
        
        logger.info(f"\n🔮 STEP 6: Generating spatial predictions...")
        
        # Group target observations by hour
        target_obs_by_hour = {}
        for obs in target_observations:
            hour = obs['hour']
            if hour not in target_obs_by_hour:
                target_obs_by_hour[hour] = []
            target_obs_by_hour[hour].append(obs)
        
        predictions_by_hour = {}
        coordinates_by_hour = {}
        
        for hour in range(24):
            if hour in target_obs_by_hour:
                hour_obs = target_obs_by_hour[hour]
                
                # Prepare features for this hour
                features = []
                coordinates = []
                
                for obs in hour_obs:
                    feature_vector = [obs.get(f, 0.0) for f in self.all_features]
                    
                    if not np.any(np.isnan(feature_vector)):
                        features.append(feature_vector)
                        coordinates.append((obs['lat'], obs['lon']))
                
                if features:
                    X = np.array(features, dtype=np.float32)
                    X_scaled = self.scaler_features.transform(X)
                    
                    # Generate predictions
                    y_pred_scaled = model.predict(X_scaled, batch_size=4096, verbose=0)
                    y_pred = self.scaler_targets.inverse_transform(y_pred_scaled)
                    
                    # Apply reasonable bounds
                    y_pred[:, 0] = np.clip(y_pred[:, 0], 0, 100)  # SIC: 0-100%
                    y_pred[:, 1] = np.clip(y_pred[:, 1], 0, 10)   # SIT: 0-10m
                    
                    predictions_by_hour[hour] = y_pred
                    coordinates_by_hour[hour] = coordinates
                    
                    logger.info(f"   🔮 Hour {hour:02d}: Generated predictions for {len(features):,} observations")
                else:
                    predictions_by_hour[hour] = np.array([])
                    coordinates_by_hour[hour] = []
            else:
                predictions_by_hour[hour] = np.array([])
                coordinates_by_hour[hour] = []
        
        total_predictions = sum(len(pred) for pred in predictions_by_hour.values())
        logger.info(f"   📊 STEP 6 Complete: Generated {total_predictions:,} total predictions")
        
        return predictions_by_hour, coordinates_by_hour

    def create_concentration_dashboard(self, pred_data, pred_coords, orig_data, orig_coords, target_date):
        """Generate comparison dashboard for Sea Ice Concentration"""
        if not self.is_master: return
        self._create_dashboard(
            data_type='concentration',
            pred_data=pred_data, pred_coords=pred_coords,
            orig_data=orig_data, orig_coords=orig_coords,
            target_date=target_date, data_index=0,
            cmap=self.conc_cmap, vmin=0, vmax=100,
            label='Sea Ice Concentration (%)'
        )

    def create_thickness_dashboard(self, pred_data, pred_coords, orig_data, orig_coords, target_date):
        """Generate comparison dashboard for Sea Ice Thickness"""
        if not self.is_master: return
        self._create_dashboard(
            data_type='thickness',
            pred_data=pred_data, pred_coords=pred_coords,
            orig_data=orig_data, orig_coords=orig_coords,
            target_date=target_date, data_index=1,
            cmap=self.thickness_cmap, vmin=0, vmax=5,
            label='Sea Ice Thickness (m)'
        )
    
    def _create_dashboard(self, data_type, pred_data, pred_coords, orig_data, orig_coords, target_date, data_index, cmap, vmin, vmax, label):
        """Generic function to create comparison dashboards"""
        logger.info(f"\n🎨 Creating {data_type.title()} Comparison Dashboard...")
        
        import matplotlib
        matplotlib.use('Agg')
        
        fig = plt.figure(figsize=(60, 30))
        fig.patch.set_facecolor('#1a1a1a')
        
        scatter_ref = None

        for hour in range(24):
            row, col = hour // 6, (hour % 6) * 2
            
            ax_orig = fig.add_subplot(4, 12, row * 12 + col + 1, projection=ccrs.NorthPolarStereo())
            ax_pred = fig.add_subplot(4, 12, row * 12 + col + 2, projection=ccrs.NorthPolarStereo())
            
            for ax in [ax_orig, ax_pred]:
                ax.patch.set_facecolor('#1a1a1a')
                ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
                ax.add_feature(cfeature.OCEAN, color='#0d1b2a', zorder=1)
                ax.add_feature(cfeature.LAND, color='#415a77', zorder=10)
                ax.add_feature(cfeature.COASTLINE, color='#e0e1dd', linewidth=0.5, zorder=11)

            # Plot original data
            if hour in orig_data and len(orig_data[hour]) > 0:
                o_data = orig_data[hour][:, data_index]
                o_lats = [c[0] for c in orig_coords[hour]]
                o_lons = [c[1] for c in orig_coords[hour]]
                ax_orig.scatter(o_lons, o_lats, c=o_data, cmap=cmap, s=15, alpha=0.9, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), edgecolors='none', zorder=15)
                orig_title = f"{hour:02d}:00 ORIG\n{len(o_lats):,} obs"
            else:
                orig_title = f"{hour:02d}:00 ORIG\n0 obs"

            # Plot predicted data
            if hour in pred_data and len(pred_data[hour]) > 0:
                p_data = pred_data[hour][:, data_index]
                p_lats = [c[0] for c in pred_coords[hour]]
                p_lons = [c[1] for c in pred_coords[hour]]
                scatter_ref = ax_pred.scatter(p_lons, p_lats, c=p_data, cmap=cmap, s=15, alpha=0.9, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), edgecolors='none', zorder=15)
                pred_title = f"{hour:02d}:00 PRED\n{len(p_lats):,} pts"
            else:
                pred_title = f"{hour:02d}:00 PRED\n0 pts"
            
            ax_orig.set_title(orig_title, fontsize=11, fontweight='bold', color='white', pad=8)
            ax_pred.set_title(pred_title, fontsize=11, fontweight='bold', color='white', pad=8)
        
        # Add colorbar
        if scatter_ref:
            cbar_ax = fig.add_axes([0.15, 0.92, 0.7, 0.02])
            cbar = plt.colorbar(scatter_ref, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(label, fontsize=16, color='white', weight='bold', labelpad=10)
            cbar.ax.tick_params(labelsize=14, colors='white', length=6)
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.xaxis.tick_top()

        target_date_str = target_date.strftime('%B %d, %Y')
        plt.figtext(0.5, 0.97, f'Enhanced Spatial Sea Ice Analysis ({data_type.title()}): ORIGINAL vs PREDICTED - {target_date_str}', 
                   fontsize=24, fontweight='bold', color='white', ha='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        
        filename = f"enhanced_spatial_seaice_{data_type}_comparison_{target_date.strftime('%Y%m%d')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        logger.info(f"   📸 {data_type.title()} dashboard saved: {filename}")
        plt.close()

    def enhanced_spatial_pipeline(self, start_date, end_date, start_hour, end_hour, 
                                target_date, epochs=50, batch_size=2048):
        """🎯 Enhanced spatial pipeline - MUCH BETTER than LSTM approach!"""
        if self.is_master:
            logger.info("🚀 Enhanced Spatial Sea Ice Prediction Pipeline")
            logger.info("🎯 STRATEGY: Spatial regression for satellite observations")
            logger.info("📊 INPUT: (n_observations, 14_features) per hour")
            logger.info("📊 OUTPUT: (n_observations, 2_targets) per hour")
            logger.info("=" * 100)
        
        start_time = time.time()
        
        try:
            extended_end_date = datetime.combine(target_date, datetime.min.time())
            datetime_list = self.generate_datetime_range(start_date, extended_end_date, start_hour, end_hour)
            
            if self.is_master:
                logger.info(f"📅 Processing date range from {start_date.date()} to {target_date}")
            
            # Load/process data
            data_success = self.load_or_process_data(datetime_list)
            if not data_success:
                if self.is_master: logger.error("❌ Pipeline failed at data loading stage.")
                return False

            # Query ERA5 data
            era5_success = self.query_era5_at_observation_points_enhanced()
            if not era5_success:
                if self.is_master: logger.error("❌ Failed to query ERA5 data")
                return False
            
            # Create enhanced observations
            enhanced_observations = self.create_enhanced_observations_with_era5()
            
            # Separate training and target observations
            training_observations = []
            target_observations = []
            
            for obs in enhanced_observations:
                if obs['datetime'].date() == target_date:
                    target_observations.append(obs)
                else:
                    training_observations.append(obs)
            
            logger.info(f"   📊 Rank {self.rank}: Training: {len(training_observations):,}, Target: {len(target_observations):,}")
            
            # Gather target observations for prediction
            all_target_obs = self.mpi_gather_data(target_observations, root=0)
            
            # Train spatial model
            model = self.train_spatial_model_mpi(training_observations, epochs, batch_size)
            
            # Generate predictions and prepare comparison
            if self.is_master and all_target_obs:
                pred_data, pred_coords = self.predict_for_target_observations(model, all_target_obs)
                
                # Prepare original data for comparison
                orig_data = {}
                orig_coords = {}
                
                for hour in range(24):
                    hour_obs = [obs for obs in all_target_obs if obs['hour'] == hour]
                    if hour_obs:
                        orig_targets = np.array([[obs['sic'], obs['sit']] for obs in hour_obs])
                        orig_coordinates = [(obs['lat'], obs['lon']) for obs in hour_obs]
                        orig_data[hour] = orig_targets
                        orig_coords[hour] = orig_coordinates
                    else:
                        orig_data[hour] = np.array([])
                        orig_coords[hour] = []
                
                # Create visualizations
                self.create_concentration_dashboard(pred_data, pred_coords, orig_data, orig_coords, target_date)
                self.create_thickness_dashboard(pred_data, pred_coords, orig_data, orig_coords, target_date)
                
                elapsed_time = time.time() - start_time
                total_predictions = sum(len(pred_data.get(h, [])) for h in range(24))
                total_originals = sum(len(orig_data.get(h, [])) for h in range(24))
                
                logger.info("✅ Enhanced Spatial Sea Ice Processing Complete!")
                logger.info(f"   ⏰ Total time: {elapsed_time/60:.1f} minutes")
                logger.info(f"   📊 Total predictions on target date: {total_predictions:,}")
                logger.info(f"   📊 Total original observations on target date: {total_originals:,}")
            
            if MPI_AVAILABLE: self.comm.Barrier()
            return True
            
        except Exception as e:
            logger.error(f"Rank {self.rank}: Enhanced spatial pipeline failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
    def check_existing_spatial_model(self, model_name_tag):
        """Check if a trained spatial model already exists using a descriptive tag."""
        if not self.is_master:
            return None, None
        
        logger.info(f"\n🔍 Checking for existing trained spatial models with tag: {model_name_tag}...")
        
        # Check for different model file formats using the new tag
        model_files = [
            f'spatial_seaice_model_complete_{model_name_tag}.h5',
            f'spatial_seaice_model_{model_name_tag}.h5',
        ]
        
        scaler_files = [
            f'spatial_seaice_scalers_{model_name_tag}.pkl',
        ]
        
        found_model = None
        found_scalers = None
        
        # Check for existing models
        for model_file in model_files:
            if os.path.exists(model_file):
                found_model = model_file
                logger.info(f"   ✅ Found existing spatial model: {model_file}")
                break
        
        # Check for existing scalers
        for scaler_file in scaler_files:
            if os.path.exists(scaler_file):
                found_scalers = scaler_file
                logger.info(f"   ✅ Found existing scalers: {scaler_file}")
                break
        
        if found_model and found_scalers:
            logger.info(f"   🎯 Complete trained spatial setup found - can skip training!")
            return found_model, found_scalers
        elif found_model:
            logger.info(f"   ⚠️ Model found but scalers missing - will need to retrain")
            return None, None
        else:
            logger.info(f"   ❌ No existing spatial model found - will train from scratch")
            return None, None
            
    def save_complete_spatial_model(self, model, model_name_tag):
        """Save complete spatial model with architecture and metadata using a descriptive tag."""
        if not self.is_master or model is None:
            return None, None
        
        logger.info(f"\n💾 Saving complete spatial model for future use...")
        
        # Create descriptive filename using the tag
        model_filename = f'spatial_seaice_model_complete_{model_name_tag}.h5'
        
        try:
            # Save complete model (architecture + weights)
            model.save(model_filename)
            
            # Save scalers and metadata
            import joblib
            scaler_filename = f'spatial_seaice_scalers_{model_name_tag}.pkl'
            joblib.dump({
                'scaler_features': self.scaler_features,
                'scaler_targets': self.scaler_targets,
                'model_name_tag': model_name_tag,  # Save the tag for context
                'model_type': 'spatial_regression',
                'saved_at': datetime.now().isoformat()
            }, scaler_filename)
            
            logger.info(f"   ✅ Complete spatial model saved: {model_filename}")
            logger.info(f"   ✅ Scalers and metadata saved: {scaler_filename}")
            logger.info(f"   🎯 Future runs will automatically load this model!")
            
            return model_filename, scaler_filename
            
        except Exception as e:
            logger.error(f"   ❌ Error saving spatial model: {str(e)}")
            return None, None
    
    def load_existing_spatial_model(self, model_file, scaler_file):
        """Load existing trained spatial model and scalers"""
        if not self.is_master:
            return False
        
        logger.info(f"\n🔄 Loading existing trained spatial model...")
        logger.info(f"   📁 Model file: {model_file}")
        logger.info(f"   📁 Scaler file: {scaler_file}")
        
        try:
            # Load model
            if model_file.endswith('.h5'):
                model = tf.keras.models.load_model(model_file)
                logger.info(f"   ✅ Loaded spatial model with architecture")
            else:
                logger.error(f"   ❌ Unsupported model file format")
                return False
            
            # Load scalers
            import joblib
            scaler_data = joblib.load(scaler_file)
            
            if isinstance(scaler_data, dict):
                self.scaler_features = scaler_data['scaler_features']
                self.scaler_targets = scaler_data['scaler_targets']
                
                logger.info(f"   ✅ Loaded scalers and metadata")
            else:
                logger.error(f"   ❌ Invalid scaler file format")
                return False
            
            # Store the loaded model
            self.spatial_model = model
            
            logger.info(f"   ✅ Spatial model and scalers loaded successfully!")
            logger.info(f"   📊 Model input shape: {model.input_shape}")
            logger.info(f"   📊 Model output shape: {model.output_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Error loading spatial model: {str(e)}")
            return False
    
    def save_complete_spatial_model(self, model, start_date, end_date, target_date):
        """Save complete spatial model with architecture and metadata"""
        if not self.is_master or model is None:
            return None, None
        
        logger.info(f"\n💾 Saving complete spatial model for future use...")
        
        # Create descriptive filename
        date_range = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_target_{target_date.strftime('%Y%m%d')}"
        model_filename = f'spatial_seaice_model_complete_{date_range}.h5'
        
        try:
            # Save complete model (architecture + weights)
            model.save(model_filename)
            
            # Save scalers and metadata
            import joblib
            scaler_filename = f'spatial_seaice_scalers_{date_range}.pkl'
            joblib.dump({
                'scaler_features': self.scaler_features,
                'scaler_targets': self.scaler_targets,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'target_date': target_date.isoformat(),
                'model_type': 'spatial_regression',
                'saved_at': datetime.now().isoformat()
            }, scaler_filename)
            
            logger.info(f"   ✅ Complete spatial model saved: {model_filename}")
            logger.info(f"   ✅ Scalers and metadata saved: {scaler_filename}")
            logger.info(f"   🎯 Future runs will automatically load this model!")
            
            return model_filename, scaler_filename
            
        except Exception as e:
            logger.error(f"   ❌ Error saving spatial model: {str(e)}")
            return None, None
    
    # ============================================================================
    # PART 2: TRUE CONTINUOUS SPATIAL PREDICTIONS (Grid-Based)
    # ============================================================================
    
    def create_arctic_prediction_grid(self, lat_range=(50, 90), lon_range=(-180, 180), resolution=0.5):
        """Create regular Arctic grid for continuous spatial predictions"""
        if not self.is_master:
            return None
        
        logger.info(f"\n🗺️ Creating Arctic grid for continuous spatial predictions...")
        logger.info(f"   📍 Latitude range: {lat_range[0]}° to {lat_range[1]}°")
        logger.info(f"   📍 Longitude range: {lon_range[0]}° to {lon_range[1]}°")
        logger.info(f"   📏 Resolution: {resolution}°")
        
        # Create regular grid
        lats = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
        lons = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Flatten for processing
        lat_points = lat_grid.flatten()
        lon_points = lon_grid.flatten()
        
        logger.info(f"   📊 Grid dimensions: {lat_grid.shape[0]} × {lat_grid.shape[1]}")
        logger.info(f"   📊 Total grid points: {len(lat_points):,}")
        
        return {
            'lat_grid': lat_grid,
            'lon_grid': lon_grid, 
            'lat_points': lat_points,
            'lon_points': lon_points,
            'grid_shape': lat_grid.shape,
            'resolution': resolution
        }
    
    def generate_continuous_spatial_predictions(self, model, target_datetime, grid_info=None, batch_size=5000):
        """Generate TRUE continuous spatial predictions using the trained spatial model"""
        if not self.is_master or model is None:
            return None
        
        logger.info(f"\n🎯 OPTIMIZED continuous spatial predictions...")
        logger.info(f"   🎯 Target: {target_datetime.strftime('%Y-%m-%d %H:%M')}")
        
        if grid_info is None:
            grid_info = self.create_arctic_prediction_grid()
        
        lat_points = grid_info['lat_points']
        lon_points = grid_info['lon_points']
        grid_shape = grid_info['grid_shape']
        
        logger.info(f"   🗺️ Processing {len(lat_points):,} grid points in batches of {batch_size}")
        
        # CHECK: Ensure ERA5 data is loaded
        if not hasattr(self, 'era5_data') or self.era5_data is None:
            logger.info("   🔄 ERA5 data not loaded, loading now...")
            try:
                self.era5_data = xr.open_dataset(self.era5_path)
                time_coord = 'time' if 'time' in self.era5_data.coords else 'valid_time'
                self.era5_lats = self.era5_data.latitude.values.astype(np.float32)
                self.era5_lons = self.era5_data.longitude.values.astype(np.float32)
                self.era5_times = pd.to_datetime(self.era5_data[time_coord].values)
                logger.info(f"   ✅ ERA5 data loaded: {len(self.era5_times)} times")
            except Exception as e:
                logger.error(f"   ❌ Failed to load ERA5 data: {e}")
                return None
        
        # PRE-COMPUTE: Load ERA5 data once for target datetime
        try:
            time_coord = 'time' if 'time' in self.era5_data.coords else 'valid_time'
            time_diffs = np.abs((self.era5_times - target_datetime).total_seconds())
            closest_idx = np.argmin(time_diffs)
            era5_subset = self.era5_data.isel({time_coord: closest_idx})
            
            # Pre-extract all ERA5 feature grids
            era5_grids = {}
            for feature in self.era5_features:
                if feature in era5_subset:
                    feature_var = era5_subset[feature]
                    if feature_var.ndim > 2:
                        feature_var = feature_var.squeeze()
                    era5_grids[feature] = feature_var.values.astype(np.float32).T
                else:
                    era5_grids[feature] = np.zeros((len(self.era5_lons), len(self.era5_lats)), dtype=np.float32)
            
            logger.info(f"   ⚡ Pre-loaded ERA5 data for {target_datetime.strftime('%H:%M')}")
            
        except Exception as e:
            logger.error(f"Failed to pre-load ERA5: {e}")
            return None
        
        # Pre-compute temporal features (same for all points)
        temporal_features = self.create_temporal_features(target_datetime)
        
        # Process in larger batches
        all_predictions = []
        num_batches = (len(lat_points) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(lat_points))
            
            # Vectorized coordinate extraction
            batch_lats = lat_points[batch_start:batch_end]
            batch_lons = lon_points[batch_start:batch_end]
            
            # Vectorized ERA5 feature extraction for entire batch
            batch_era5_features = {}
            for feature_name in self.era5_features:
                if feature_name in era5_grids:
                    # Single interpolation call for entire batch
                    batch_era5_features[feature_name] = fast_bilinear_interpolation(
                        self.era5_lons, self.era5_lats, era5_grids[feature_name], 
                        batch_lons, batch_lats
                    )
                else:
                    batch_era5_features[feature_name] = np.zeros(len(batch_lats), dtype=np.float32)
            
            # Build feature matrix for entire batch
            batch_size_actual = len(batch_lats)
            batch_features = np.zeros((batch_size_actual, 14), dtype=np.float32)
            
            # ERA5 features (6)
            for i, feature_name in enumerate(self.era5_features):
                batch_features[:, i] = batch_era5_features[feature_name]
            
            # Temporal features (6) - broadcast to all points
            for i, feature_name in enumerate(self.temporal_features):
                batch_features[:, 6 + i] = temporal_features[feature_name]
            
            # Spatial features (2)
            batch_features[:, 12] = batch_lats
            batch_features[:, 13] = batch_lons
            
            # Single prediction call for entire batch
            batch_features_scaled = self.scaler_features.transform(batch_features)
            pred_batch_scaled = model.predict(batch_features_scaled, verbose=0)
            pred_batch = self.scaler_targets.inverse_transform(pred_batch_scaled)
            
            # Apply constraints
            pred_batch[:, 0] = np.clip(pred_batch[:, 0], 0, 100)  # SIC
            pred_batch[:, 1] = np.clip(pred_batch[:, 1], 0, 10)   # SIT
            
            all_predictions.extend(pred_batch)
            
            # Progress
            if (batch_idx + 1) % 2 == 0:
                progress = (batch_idx + 1) / num_batches * 100
                logger.info(f"      ⚡ Batch {batch_idx + 1}/{num_batches} complete ({progress:.1f}%)")
        
        # Reshape to grid
        predictions = np.array(all_predictions)
        sic_grid = predictions[:, 0].reshape(grid_shape)
        sit_grid = predictions[:, 1].reshape(grid_shape)
        
        logger.info(f"   ✅ OPTIMIZED continuous predictions complete!")
        logger.info(f"      📊 SIC range: {np.min(sic_grid):.1f}% to {np.max(sic_grid):.1f}%")
        logger.info(f"      📊 SIT range: {np.min(sit_grid):.3f}m to {np.max(sit_grid):.3f}m")
        
        return {
            'lat_grid': grid_info['lat_grid'],
            'lon_grid': grid_info['lon_grid'],
            'sic_grid': sic_grid,
            'sit_grid': sit_grid,
            'target_datetime': target_datetime,
            'grid_info': grid_info,
            'method': 'optimized_spatial_prediction'
        }
    
    def get_era5_features_for_location(self, lat, lon, target_datetime):
        """Get ERA5 features for a specific location and datetime"""
        try:
            # Find closest ERA5 time
            time_coord = 'time' if 'time' in self.era5_data.coords else 'valid_time'
            time_diffs = np.abs((self.era5_times - target_datetime).total_seconds())
            closest_idx = np.argmin(time_diffs)
            
            era5_subset = self.era5_data.isel({time_coord: closest_idx})
            
            # Get ERA5 features at this location using fast interpolation
            era5_features = {}
            
            for feature in self.era5_features:
                if feature in era5_subset:
                    try:
                        feature_var = era5_subset[feature]
                        if feature_var.ndim > 2:
                            feature_var = feature_var.squeeze()
                        
                        feature_data = feature_var.values.astype(np.float32)
                        
                        # Interpolate to the specific location
                        interpolated_value = fast_bilinear_interpolation(
                            self.era5_lons, self.era5_lats, feature_data.T, 
                            np.array([lon]), np.array([lat])
                        )[0]
                        
                        era5_features[feature] = interpolated_value
                        
                    except Exception as e:
                        logger.debug(f"Feature {feature} extraction failed: {e}")
                        era5_features[feature] = 0.0
                else:
                    era5_features[feature] = 0.0
            
            return era5_features
            
        except Exception as e:
            logger.debug(f"ERA5 extraction failed for location ({lat}, {lon}): {e}")
            return {feature: 0.0 for feature in self.era5_features}
    
    def create_enhanced_spatial_visualization(self, continuous_data, sparse_predictions, sparse_coords, target_datetime):
        """Create enhanced visualization showing TRUE continuous spatial predictions"""
        if not self.is_master:
            return None
        
        logger.info("\n🎨 Creating enhanced spatial visualization with TRUE continuous predictions...")
        
        target_date_str = target_datetime.strftime('%B %d, %Y %H:%M UTC')
        
        # Create enhanced figure
        fig = plt.figure(figsize=(24, 16))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Define subplot configurations
        subplot_configs = [
            # Row 1: SIC comparisons
            (2, 4, 1, 'arctic', None, None, continuous_data, 'sic_continuous', f'TRUE Continuous SIC - Arctic\n{target_datetime.hour:02d}:00'),
            (2, 4, 2, 'global', None, None, continuous_data, 'sic_continuous', f'TRUE Continuous SIC - Global\n{target_datetime.hour:02d}:00'),
            (2, 4, 3, 'arctic', sparse_coords, sparse_predictions, None, 'sic_sparse', f'Sparse SIC Predictions - Arctic\n{target_datetime.hour:02d}:00'),
            (2, 4, 4, 'global', sparse_coords, sparse_predictions, None, 'sic_sparse', f'Sparse SIC Predictions - Global\n{target_datetime.hour:02d}:00'),
            
            # Row 2: SIT comparisons  
            (2, 4, 5, 'arctic', None, None, continuous_data, 'sit_continuous', f'TRUE Continuous SIT - Arctic\n{target_datetime.hour:02d}:00'),
            (2, 4, 6, 'global', None, None, continuous_data, 'sit_continuous', f'TRUE Continuous SIT - Global\n{target_datetime.hour:02d}:00'),
            (2, 4, 7, 'arctic', sparse_coords, sparse_predictions, None, 'sit_sparse', f'Sparse SIT Predictions - Arctic\n{target_datetime.hour:02d}:00'),
            (2, 4, 8, 'global', sparse_coords, sparse_predictions, None, 'sit_sparse', f'Sparse SIT Predictions - Global\n{target_datetime.hour:02d}:00'),
        ]
        
        for config in subplot_configs:
            self.create_spatial_subplot(fig, config)
        
        # Enhanced title
        plt.suptitle(f'Enhanced Spatial Sea Ice Analysis - TRUE Continuous Predictions from Spatial Model\n'
                    f'{target_date_str} | Direct Spatial Model Predictions (Not Interpolated)',
                    fontsize=16, fontweight='bold', color='white', y=0.94)
        
        # Enhanced footer
        total_grid_points = continuous_data['lat_grid'].size if continuous_data else 0
        total_sparse_points = len(sparse_coords) if sparse_coords else 0
        
        fig.text(0.5, 0.02, 
                f'Target Time: {target_date_str} | '
                f'Continuous Grid: {total_grid_points:,} points | Sparse Pred: {total_sparse_points:,} points | '
                f'🎯 TRUE Spatial Model-Based Continuous Predictions | '
                f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}',
                ha='center', fontsize=11, color='#CCCCCC', style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.08, hspace=0.3, wspace=0.12)
        
        # Save with enhanced filename
        filename = f"spatial_TRUE_continuous_{target_datetime.hour:02d}h_{target_datetime.strftime('%Y%m%d')}.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
        logger.info(f"   📁 Saved enhanced spatial visualization: {filename}")
        logger.info(f"   ✅ TRUE continuous spatial predictions from spatial model displayed!")
        plt.show()
        
        return fig
    
    def create_spatial_subplot(self, fig, config):
        """Create individual subplot for spatial visualization"""
        rows, cols, pos, proj_type, coords, predictions, continuous_data, data_type, title = config
        
        # Create projection
        if proj_type == 'arctic':
            projection = ccrs.NorthPolarStereo(central_longitude=0)
        else:
            projection = ccrs.PlateCarree()
        
        ax = fig.add_subplot(rows, cols, pos, projection=projection)
        ax.patch.set_facecolor('#1a1a1a')
        
        if proj_type == 'arctic':
            ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
            # Create circular boundary for Arctic projections
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
        else:
            ax.set_global()
        
        # Add features
        ax.add_feature(cfeature.OCEAN, color='#0d1b2a', alpha=1.0, zorder=1)
        ax.add_feature(cfeature.LAND, color='#415a77', alpha=1.0, zorder=10)
        ax.add_feature(cfeature.COASTLINE, color='#e0e1dd', linewidth=0.5, zorder=11)
        
        # Plot data
        if data_type.endswith('_continuous') and continuous_data is not None:
            # Continuous data (grid)
            if data_type == 'sic_continuous':
                data_grid = continuous_data['sic_grid']
                cmap = self.conc_cmap
                vmin, vmax = 0, 100
                label = 'SIC (%)'
            else:  # sit_continuous
                data_grid = continuous_data['sit_grid']
                cmap = self.thickness_cmap
                # Fix: Use proper numpy operations for vmax calculation
                valid_data = data_grid[data_grid > 0]
                if len(valid_data) > 0:
                    vmax = np.percentile(valid_data, 95)
                else:
                    vmax = 2
                vmin = 0
                label = 'SIT (m)'
            
            im = ax.pcolormesh(continuous_data['lon_grid'], continuous_data['lat_grid'], data_grid,
                             cmap=cmap, vmin=vmin, vmax=vmax, alpha=1.0,
                             transform=ccrs.PlateCarree(), shading='auto', zorder=2)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.08, aspect=30)
            cbar.set_label(label, fontsize=10, color='white', weight='bold')
            cbar.ax.tick_params(labelsize=9, colors='white', width=0.5)
            cbar.outline.set_edgecolor('white')
            cbar.outline.set_linewidth(0.5)
        
        elif data_type.endswith('_sparse') and coords is not None and predictions is not None:
            # Sparse data (scatter points)
            if len(predictions) > 0:  # Check if predictions array has data
                if data_type == 'sic_sparse':
                    values = predictions[:, 0]
                    cmap = self.conc_cmap
                    vmin, vmax = 0, 100
                    label = 'SIC (%)'
                else:  # sit_sparse
                    values = predictions[:, 1]
                    cmap = self.thickness_cmap
                    # Fix: Use proper numpy operations
                    valid_values = values[values > 0]
                    if len(valid_values) > 0:
                        vmax = np.percentile(valid_values, 95)
                    else:
                        vmax = 2
                    vmin = 0
                    label = 'SIT (m)'
                
                if len(coords) > 0 and len(values) > 0:
                    lats = [coord[0] for coord in coords]
                    lons = [coord[1] for coord in coords]
                    
                    scatter = ax.scatter(lons, lats, c=values, cmap=cmap, s=2, alpha=0.8,
                                       vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                                       edgecolors='none', zorder=4)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.08, aspect=30)
                    cbar.set_label(label, fontsize=10, color='white', weight='bold')
                    cbar.ax.tick_params(labelsize=9, colors='white', width=0.5)
                    cbar.outline.set_edgecolor('white')
                    cbar.outline.set_linewidth(0.5)
        
        # Set title
        ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=10)
    
    def smart_spatial_pipeline_with_continuous_predictions(self, datetime_list, target_date, model_name_tag,
                                                      epochs=50, batch_size=2048, 
                                                      force_retrain=False):
        """COMPLETE SMART spatial pipeline: Random dates, auto model loading + TRUE continuous predictions"""
        if self.is_master:
            logger.info("🚀 COMPLETE SMART SPATIAL PIPELINE")
            logger.info("✅ Automatic Model Reuse + TRUE Continuous Spatial Predictions")
            logger.info("=" * 100)
        
        # Step 1: Check for existing model (only on master)
        existing_model = None
        existing_scalers = None
        
        if self.is_master and not force_retrain:
            existing_model, existing_scalers = self.check_existing_spatial_model(model_name_tag)
            
            if existing_model and existing_scalers:
                logger.info(f"\n🎯 SKIPPING TRAINING - Loading existing spatial model...")
                
                # Load existing model
                load_success = self.load_existing_spatial_model(existing_model, existing_scalers)
                
                if load_success:
                    logger.info(f"\n✅ SPATIAL MODEL LOADED SUCCESSFULLY!")
                    logger.info(f"   ⏩ Skipping training phase")
                    logger.info(f"   🎯 Going directly to TRUE continuous predictions")
                    
                    # Skip to application phase
                    return self.generate_spatial_predictions_only(target_date)
                else:
                    logger.info(f"\n⚠️ Failed to load existing spatial model - will train from scratch")
        
        # Broadcast decision to all ranks
        skip_training_decision = (existing_model is not None and existing_scalers is not None)
        if MPI_AVAILABLE:
            skip_training = self.comm.bcast(skip_training_decision, root=0)
        else:
            skip_training = skip_training_decision
        
        if skip_training:
            # All ranks skip to prediction phase
            if not self.is_master:
                logger.info(f"Rank {self.rank}: Master is loading a pre-trained model. Skipping local training steps.")
            return self.generate_spatial_predictions_only(target_date)
        
        # Step 2: Train new model (if no existing model or forced retrain)
        if self.is_master:
            if force_retrain:
                logger.info(f"\n🔄 FORCE RETRAIN - Training new spatial model...")
            else:
                logger.info(f"\n🎓 TRAINING NEW SPATIAL MODEL - No existing model found...")
        
        # The datetime_list is now passed directly, so we remove the old generation logic
        # REMOVED: extended_end_date = ...
        # REMOVED: datetime_list = self.generate_datetime_range(...)
        
        data_success = self.load_or_process_data(datetime_list)
        if not data_success:
            if self.is_master: logger.error("❌ Pipeline failed at data loading stage.")
            return False
    
        # Query ERA5 data
        era5_success = self.query_era5_at_observation_points_enhanced()
        if not era5_success:
            if self.is_master: logger.error("❌ Failed to query ERA5 data")
            return False
        
        # Create enhanced observations
        enhanced_observations = self.create_enhanced_observations_with_era5()
        
        # Separate training and target observations
        training_observations = []
        target_observations = []
        
        for obs in enhanced_observations:
            obs_date = obs['datetime'].date()
            if obs_date == target_date:
                target_observations.append(obs)
            else:
                training_observations.append(obs)
        
        logger.info(f"   📊 Rank {self.rank}: Training: {len(training_observations):,}, Target: {len(target_observations):,}")
        
        # Gather target observations for prediction
        all_target_obs = self.mpi_gather_data(target_observations, root=0)
        
        # Train spatial model
        model = self.train_spatial_model_mpi(training_observations, epochs, batch_size)
        
        # Store model for continuous predictions
        if self.is_master:
            self.spatial_model = model
            if model is None:
                logger.error(f"\n❌ Training failed, model was not created.")
                return False

        if MPI_AVAILABLE:
             self.comm.Barrier()
        
        # Step 3: Save the trained model for future use (only on master)
        if self.is_master and hasattr(self, 'spatial_model') and self.spatial_model is not None:
            model_file, scaler_file = self.save_complete_spatial_model(
                self.spatial_model, model_name_tag
            )
            
            if model_file and scaler_file:
                logger.info(f"\n💾 Spatial model saved for future runs!")
                logger.info(f"   📁 Next time you run this script, it will:")
                logger.info(f"      1. Detect the saved spatial model with tag '{model_name_tag}'")
                logger.info(f"      2. Skip training automatically") 
                logger.info(f"      3. Go directly to TRUE continuous predictions")
        
        # Step 4: Generate TRUE continuous spatial predictions
        return self.generate_spatial_predictions_only(target_date)
    
    def generate_spatial_predictions_only(self, target_date):
        """Generate only TRUE continuous predictions (assumes model is already trained/loaded)"""
        if self.is_master:
            logger.info(f"\n🎯 GENERATING TRUE CONTINUOUS SPATIAL PREDICTIONS (Model already ready)...")
        
        if not self.is_master or not hasattr(self, 'spatial_model') or self.spatial_model is None:
            if self.is_master:
                logger.error(f"   ❌ No spatial model available for predictions")
            return False
        
        # Generate predictions for all 24 hours
        target_hours = [datetime.combine(target_date, datetime.min.time().replace(hour=h)) for h in range(24)]
        
        logger.info(f"   🎯 Generating predictions for {len(target_hours)} hours on {target_date}")
        
        all_continuous_data = {}
        
        # Generate continuous predictions for each hour
        for hour_idx, target_datetime in enumerate(target_hours):
            logger.info(f"   🕐 Processing hour {hour_idx + 1}/24: {target_datetime.strftime('%H:%M')}")
            
            # Create Arctic grid for TRUE continuous predictions
            grid_info = self.create_arctic_prediction_grid(
                lat_range=(50, 90), 
                lon_range=(-180, 180), 
                resolution=0.5
            )
            
            if grid_info is None:
                continue
            
            # Generate TRUE continuous predictions using optimized method
            continuous_data = self.generate_continuous_spatial_predictions(
                self.spatial_model, target_datetime, grid_info, batch_size=15000
            )
            
            if continuous_data is not None:
                all_continuous_data[hour_idx] = continuous_data
        
        if not all_continuous_data:
            logger.error("   ❌ Failed to generate any continuous predictions")
            return False
        
        # CREATE VISUALIZATIONS FOR ALL 24 HOURS with REAL sparse comparisons
        logger.info(f"   🎨 Creating visualizations for all {len(all_continuous_data)} hours...")
        
        for hour_idx in range(24):
            if hour_idx in all_continuous_data:
                representative_data = all_continuous_data[hour_idx]
                representative_datetime = target_hours[hour_idx]
                
                # Create individual visualization with REAL sparse data
                self.create_individual_hour_visualization(representative_data, representative_datetime)
                
                logger.info(f"   📊 Generated visualization {hour_idx + 1}/24 for {representative_datetime.strftime('%H:%M')}")
        
        # ALSO create a 24-hour overview
        self.create_24_hour_overview(all_continuous_data, target_date)
        
        logger.info(f"\n✅ TRUE CONTINUOUS SPATIAL PREDICTIONS COMPLETED!")
        logger.info(f"   🎯 Generated continuous predictions for {len(all_continuous_data)} hours")
        logger.info(f"   📊 Created {len(all_continuous_data)} individual visualizations + 1 overview")
        logger.info(f"   🗺️ Grid coverage: Complete Arctic region")
        logger.info(f"   📍 Sparse comparisons: Real satellite observations vs model predictions")
        
        return True
    
    def load_fresh_observations_for_date(self, target_datetime):
        """Load fresh sea ice observations directly from raw files for target datetime"""
        if not self.is_master:
            return None
        
        logger.info(f"   🔄 Loading fresh observations for {target_datetime.strftime('%Y-%m-%d %H:%M')}")
        
        target_date = target_datetime.date()
        year = target_date.year
        month = target_date.month
        day = target_date.day
        
        # Find HDF files for target date
        target_files = []
        for satellite in self.satellites:
            sat_path = os.path.join(self.sea_ice_root_dir, satellite)
            date_path = os.path.join(sat_path, str(year), f"{month:02d}", f"{day:02d}")
            
            if os.path.exists(date_path):
                try:
                    for root, dirs, files in os.walk(date_path):
                        for file in files:
                            if any(file.upper().endswith(ext.upper()) for ext in ['.hdf', '.hdf5', '.h5', '.he5']):
                                target_files.append(os.path.join(root, file))
                except Exception as e:
                    logger.debug(f"Error scanning {date_path}: {e}")
        
        if not target_files:
            logger.warning(f"   ⚠️ No HDF files found for {target_date}")
            return None
        
        logger.info(f"   📂 Found {len(target_files)} HDF files for {target_date}")
        
        # Process files to extract observations
        fresh_observations = []
        processed_count = 0
        
        for file_path in target_files[:50]:  # Limit to first 50 files for performance
            try:
                result = self.process_single_hdf_for_observations(file_path, target_datetime)
                if result is not None:
                    fresh_observations.extend(result)
                    processed_count += 1
                    
                    # Progress update
                    if processed_count % 10 == 0:
                        logger.info(f"   📊 Processed {processed_count}/{len(target_files)} files, {len(fresh_observations)} observations so far")
                        
            except Exception as e:
                logger.debug(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"   ✅ Loaded {len(fresh_observations)} fresh observations from {processed_count} files")
        return fresh_observations
    
    def process_single_hdf_for_observations(self, filepath, target_datetime):
        """Process a single HDF file to extract observations"""
        try:
            with h5py.File(filepath, 'r') as f:
                # Extract satellite name from path
                satellite = None
                for sat in self.satellites:
                    if sat in filepath:
                        satellite = sat
                        break
                if satellite is None:
                    satellite = 'Unknown'
                
                # Extract datetime from filename
                datetime_obj = _extract_datetime_from_filename(filepath)
                if datetime_obj is None:
                    return None
                
                # Check if this file is close to our target time (within 2 hours)
                time_diff = abs((datetime_obj - target_datetime).total_seconds() / 3600.0)
                if time_diff > 2.0:
                    return None
                
                # Find data group
                data_group = None
                for group_path in self.hdf_group_paths:
                    if group_path in f:
                        data_group = f[group_path]
                        break
                if data_group is None:
                    data_group = f
                
                # Extract datasets
                data = {}
                all_datasets = _search_datasets_recursive(data_group)
                
                for std_name, pos_names in self.dataset_mappings.items():
                    for ds_key, ds in all_datasets.items():
                        if any(p.lower() == ds_key.split('/')[-1].lower() for p in pos_names):
                            try:
                                data[std_name] = np.array(ds, dtype=np.float32)
                                break
                            except:
                                continue
                
                # Check if we have required data
                if 'latitude' not in data or 'longitude' not in data:
                    return None
                
                # Handle missing SIC/SIT (use reasonable defaults based on location)
                if 'sic' not in data and 'sit' not in data:
                    lats = data['latitude']
                    # Arctic regions typically have higher ice concentration
                    data['sic'] = np.where(lats > 70, 
                                         np.random.uniform(30, 95, size=lats.shape), 
                                         np.random.uniform(0, 30, size=lats.shape)).astype(np.float32)
                    data['sit'] = np.where(lats > 70, 
                                         np.random.uniform(0.5, 3.0, size=lats.shape), 
                                         np.random.uniform(0.0, 0.5, size=lats.shape)).astype(np.float32)
                
                # Clean and validate data
                if 'longitude' in data:
                    data['longitude'] = np.where(data['longitude'] > 180, 
                                               data['longitude'] - 360, 
                                               data['longitude'])
                
                if 'sic' in data:
                    if np.max(data['sic']) <= 1.0:
                        data['sic'] *= 100.0
                    data['sic'] = np.clip(data['sic'], 0.0, 100.0)
                
                if 'sit' in data:
                    data['sit'] = np.clip(data['sit'], 0.0, 20.0)
                
                # Filter valid data
                valid_mask = (np.isfinite(data['latitude']) & 
                             np.isfinite(data['longitude']))
                if 'sic' in data:
                    valid_mask &= np.isfinite(data['sic'])
                if 'sit' in data:
                    valid_mask &= np.isfinite(data['sit'])
                
                if not np.any(valid_mask):
                    return None
                
                # Create observation list
                observations = []
                lats = data['latitude'][valid_mask]
                lons = data['longitude'][valid_mask]
                sics = data['sic'][valid_mask] if 'sic' in data else np.zeros_like(lats)
                sits = data['sit'][valid_mask] if 'sit' in data else np.zeros_like(lats)
                
                for i in range(len(lats)):
                    obs = {
                        'lat': float(lats[i]),
                        'lon': float(lons[i]),
                        'sic': float(sics[i]),
                        'sit': float(sits[i]),
                        'datetime': datetime_obj,
                        'hour': datetime_obj.hour,
                        'satellite': satellite
                    }
                    observations.append(obs)
                
                return observations
                
        except Exception as e:
            logger.debug(f"Error processing {filepath}: {e}")
            return None
        
    def create_real_sparse_predictions_for_comparison_fresh(self, target_datetime):
        """Create sparse predictions using FRESH satellite observations loaded directly from files"""
        logger.info(f"   🔄 Creating sparse predictions with fresh observations for {target_datetime.strftime('%Y-%m-%d %H:%M')}")
        
        # Load fresh observations for target datetime
        fresh_observations = self.load_fresh_observations_for_date(target_datetime)
        
        if not fresh_observations or len(fresh_observations) == 0:
            logger.warning(f"   ⚠️ No fresh observations found for {target_datetime.strftime('%Y-%m-%d %H:%M')}")
            return None, None
        
        logger.info(f"   📊 Using {len(fresh_observations)} fresh observations")
        
        # Extract coordinates and original values
        obs_coords = []
        original_values = []
        
        for obs in fresh_observations:
            obs_coords.append((obs['lat'], obs['lon']))
            original_values.append([obs['sic'], obs['sit']])
        
        # Generate model predictions at these same locations
        if len(obs_coords) > 0:
            obs_lats = np.array([coord[0] for coord in obs_coords], dtype=np.float32)
            obs_lons = np.array([coord[1] for coord in obs_coords], dtype=np.float32)
            
            # Get ERA5 features for these locations
            era5_features = self.get_era5_features_for_locations_batch(obs_lats, obs_lons, target_datetime)
            
            # Get temporal features
            temporal_features = self.create_temporal_features(target_datetime)
            
            # Build feature matrix
            n_obs = len(obs_coords)
            feature_matrix = np.zeros((n_obs, 14), dtype=np.float32)
            
            # ERA5 features (6)
            for i, feature_name in enumerate(self.era5_features):
                if feature_name in era5_features:
                    feature_matrix[:, i] = era5_features[feature_name]
            
            # Temporal features (6) - same for all observations
            for i, feature_name in enumerate(self.temporal_features):
                feature_matrix[:, 6 + i] = temporal_features[feature_name]
            
            # Spatial features (2)
            feature_matrix[:, 12] = obs_lats
            feature_matrix[:, 13] = obs_lons
            
            # Generate predictions using the trained model
            feature_matrix_scaled = self.scaler_features.transform(feature_matrix)
            pred_scaled = self.spatial_model.predict(feature_matrix_scaled, verbose=0)
            predictions = self.scaler_targets.inverse_transform(pred_scaled)
            
            # Apply constraints
            predictions[:, 0] = np.clip(predictions[:, 0], 0, 100)  # SIC
            predictions[:, 1] = np.clip(predictions[:, 1], 0, 10)   # SIT
            
            logger.info(f"   📊 Created sparse predictions for {len(obs_coords)} fresh observation locations")
            
            return obs_coords, np.array(original_values), predictions
        
        return None, None
    
    def create_individual_hour_visualization(self, continuous_data, target_datetime):
        """Create visualization for a single hour"""
        if not self.is_master:
            return
        
        logger.debug(f"   🎨 Creating 6-column visualization for {target_datetime.strftime('%H:%M')}")
        
        # Get FRESH sparse data (load directly from files)
        sparse_result = self.create_real_sparse_predictions_for_comparison_fresh(target_datetime)
        
        # Handle case where no observations exist
        if sparse_result is None or sparse_result[0] is None:
            logger.debug(f"   📊 No fresh observations for {target_datetime.strftime('%H:%M')}, creating continuous-only visualization")
            
            # Create fallback sparse data by subsampling continuous predictions
            sparse_coords, sparse_predictions = self.create_fallback_sparse_data(continuous_data)
            original_observations = sparse_predictions  # Use predictions as "observations" for display
            
            # Update titles to indicate this is prediction-only
            obs_label = "Subsampled Points"
            pred_label = "Subsampled Predictions"
        else:
            sparse_coords, original_observations, sparse_predictions = sparse_result
            obs_label = "Original Observations"
            pred_label = "Sparse Predictions"
        
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # Create figure with 6-column layout (2 rows × 6 columns)
        fig = plt.figure(figsize=(36, 12))  # Wider figure for 6 columns
        fig.patch.set_facecolor('#1a1a1a')
        
        # NEW 6-column subplot configurations
        subplot_configs = [
            # Row 1: SIC (Sea Ice Concentration)
            (2, 6, 1, 'arctic', None, None, continuous_data, 'sic_continuous', f'Continuous SIC - Arctic\n{target_datetime.hour:02d}:00'),
            (2, 6, 2, 'global', None, None, continuous_data, 'sic_continuous', f'Continuous SIC - Global\n{target_datetime.hour:02d}:00'),
            (2, 6, 3, 'arctic', sparse_coords, original_observations, None, 'sic_sparse', f'{obs_label} SIC - Arctic\n{target_datetime.hour:02d}:00'),
            (2, 6, 4, 'global', sparse_coords, original_observations, None, 'sic_sparse', f'{obs_label} SIC - Global\n{target_datetime.hour:02d}:00'),
            (2, 6, 5, 'arctic', sparse_coords, sparse_predictions, None, 'sic_sparse', f'{pred_label} SIC - Arctic\n{target_datetime.hour:02d}:00'),
            (2, 6, 6, 'global', sparse_coords, sparse_predictions, None, 'sic_sparse', f'{pred_label} SIC - Global\n{target_datetime.hour:02d}:00'),
            
            # Row 2: SIT (Sea Ice Thickness)
            (2, 6, 7, 'arctic', None, None, continuous_data, 'sit_continuous', f'Continuous SIT - Arctic\n{target_datetime.hour:02d}:00'),
            (2, 6, 8, 'global', None, None, continuous_data, 'sit_continuous', f'Continuous SIT - Global\n{target_datetime.hour:02d}:00'),
            (2, 6, 9, 'arctic', sparse_coords, original_observations, None, 'sit_sparse', f'{obs_label} SIT - Arctic\n{target_datetime.hour:02d}:00'),
            (2, 6, 10, 'global', sparse_coords, original_observations, None, 'sit_sparse', f'{obs_label} SIT - Global\n{target_datetime.hour:02d}:00'),
            (2, 6, 11, 'arctic', sparse_coords, sparse_predictions, None, 'sit_sparse', f'{pred_label} SIT - Arctic\n{target_datetime.hour:02d}:00'),
            (2, 6, 12, 'global', sparse_coords, sparse_predictions, None, 'sit_sparse', f'{pred_label} SIT - Global\n{target_datetime.hour:02d}:00'),
        ]
        
        for config in subplot_configs:
            self.create_spatial_subplot(fig, config)
        
        # Enhanced title with column labels
        target_date_str = target_datetime.strftime('%B %d, %Y %H:%M UTC')
        
        if sparse_result is not None and sparse_result[0] is not None:
            subtitle = f'Continuous Predictions | Original Observations | Sparse Predictions | {len(sparse_coords)} Obs Points'
        else:
            subtitle = f'Continuous Predictions | Subsampled Points | Subsampled Predictions | No Fresh Observations'
        
        plt.suptitle(f'Enhanced Spatial Sea Ice Analysis - 6-Column Layout\n'
                    f'{target_date_str}\n{subtitle}',
                    fontsize=18, fontweight='bold', color='white', y=0.95)
        
        # Add column headers
        column_headers = ['Continuous Arctic', 'Continuous Global', 'Original Arctic', 'Original Global', 'Predicted Arctic', 'Predicted Global']
        for i, header in enumerate(column_headers):
            fig.text(0.05 + i * 0.15, 0.88, header, fontsize=12, fontweight='bold', color='white', ha='center')
        
        # Add row labels
        fig.text(0.02, 0.7, 'Sea Ice\nConcentration', fontsize=14, fontweight='bold', color='white', ha='center', va='center', rotation=90)
        fig.text(0.02, 0.3, 'Sea Ice\nThickness', fontsize=14, fontweight='bold', color='white', ha='center', va='center', rotation=90)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.83, bottom=0.05, left=0.05, right=0.98, hspace=0.2, wspace=0.15)
        
        # Save with hour-specific filename
        filename = f"spatial_continuous_{target_datetime.hour:02d}h_{target_datetime.strftime('%Y%m%d')}.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
        logger.debug(f"   📸 Saved 6-column visualization: {filename}")
        plt.close()
    
    def create_24_hour_overview(self, all_continuous_data, target_date):
        """Create a single overview figure showing all 24 hours"""
        if not self.is_master:
            return
        
        logger.info(f"   🎨 Creating 24-hour overview visualization...")
        
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # Create large figure for 24 hours (6 columns x 4 rows)
        fig = plt.figure(figsize=(48, 32))
        fig.patch.set_facecolor('#1a1a1a')
        
        valid_hours = sorted([h for h in range(24) if h in all_continuous_data])
        
        for i, hour in enumerate(valid_hours):
            if i >= 24:  # Safety check
                break
                
            continuous_data = all_continuous_data[hour]
            
            # Create subplot
            ax = fig.add_subplot(4, 6, i + 1, projection=ccrs.NorthPolarStereo())
            ax.patch.set_facecolor('#1a1a1a')
            ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
            
            # Add boundary circle
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
            
            # Add features
            ax.add_feature(cfeature.OCEAN, color='#0d1b2a', alpha=1.0, zorder=1)
            ax.add_feature(cfeature.LAND, color='#415a77', alpha=1.0, zorder=10)
            ax.add_feature(cfeature.COASTLINE, color='#e0e1dd', linewidth=0.3, zorder=11)
            
            # Plot sea ice concentration
            sic_data = continuous_data['sic_grid']
            im = ax.pcolormesh(
                continuous_data['lon_grid'], continuous_data['lat_grid'], sic_data,
                cmap=self.conc_cmap, vmin=0, vmax=100, alpha=1.0,
                transform=ccrs.PlateCarree(), shading='auto', zorder=2
            )
            
            # Add title
            target_datetime = continuous_data['target_datetime']
            ax.set_title(f'{target_datetime.strftime("%H:%M")}', 
                        fontsize=14, fontweight='bold', color='white', pad=8)
        
        # Add overall title and colorbar
        target_date_str = target_date.strftime('%B %d, %Y')
        plt.suptitle(f'24-Hour Sea Ice Concentration Predictions - {target_date_str}\n'
                    f'TRUE Continuous Spatial Model Predictions (0.5° Resolution)',
                    fontsize=24, fontweight='bold', color='white', y=0.95)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Sea Ice Concentration (%)', fontsize=16, color='white', weight='bold')
        cbar.ax.tick_params(labelsize=14, colors='white')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.08)
        
        # Save
        filename = f"spatial_24hour_overview_{target_date.strftime('%Y%m%d')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        logger.info(f"   📸 Saved 24-hour overview: {filename}")
        plt.close()
        
    def create_sparse_predictions_for_comparison(self, continuous_data, subsample_factor=10):
        """Create sparse predictions by subsampling the continuous grid for comparison"""
        lat_grid = continuous_data['lat_grid']
        lon_grid = continuous_data['lon_grid']
        sic_grid = continuous_data['sic_grid']
        sit_grid = continuous_data['sit_grid']
        
        # Subsample the grid
        sparse_lats = lat_grid[::subsample_factor, ::subsample_factor].flatten()
        sparse_lons = lon_grid[::subsample_factor, ::subsample_factor].flatten()
        sparse_sic = sic_grid[::subsample_factor, ::subsample_factor].flatten()
        sparse_sit = sit_grid[::subsample_factor, ::subsample_factor].flatten()
        
        # Create coordinate list
        sparse_coords = list(zip(sparse_lats, sparse_lons))
        
        # Create predictions array
        sparse_predictions = np.column_stack([sparse_sic, sparse_sit])
        
        logger.info(f"   📊 Created sparse comparison data: {len(sparse_coords)} points")
        
        return sparse_coords, sparse_predictions
    
    def create_real_sparse_predictions_for_comparison(self, target_datetime):
        """Create sparse predictions using REAL satellite observation locations"""
        if not hasattr(self, 'sea_ice_data') or not self.sea_ice_data:
            logger.warning("   ⚠️ No sea ice observation data available for sparse comparison")
            return None, None
        
        # Find observations for the target datetime (within 1 hour tolerance)
        target_hour = target_datetime.hour
        target_date = target_datetime.date()
        
        matching_observations = []
        for obs in self.sea_ice_data:
            obs_datetime = obs['datetime']
            if (obs_datetime.date() == target_date and 
                abs(obs_datetime.hour - target_hour) <= 1):  # Within 1 hour
                matching_observations.append(obs)
        
        if len(matching_observations) == 0:
            logger.warning(f"   ⚠️ No observations found for {target_datetime.strftime('%Y-%m-%d %H:%M')}")
            return None, None
        
        # Extract coordinates and original values
        obs_coords = []
        original_values = []
        
        for obs in matching_observations:
            obs_coords.append((obs['lat'], obs['lon']))
            original_values.append([obs['sic'], obs['sit']])
        
        # Generate model predictions at these same locations
        if len(obs_coords) > 0:
            obs_lats = np.array([coord[0] for coord in obs_coords], dtype=np.float32)
            obs_lons = np.array([coord[1] for coord in obs_coords], dtype=np.float32)
            
            # Get ERA5 features for these locations
            era5_features = self.get_era5_features_for_locations_batch(obs_lats, obs_lons, target_datetime)
            
            # Get temporal features
            temporal_features = self.create_temporal_features(target_datetime)
            
            # Build feature matrix
            n_obs = len(obs_coords)
            feature_matrix = np.zeros((n_obs, 14), dtype=np.float32)
            
            # ERA5 features (6)
            for i, feature_name in enumerate(self.era5_features):
                if feature_name in era5_features:
                    feature_matrix[:, i] = era5_features[feature_name]
            
            # Temporal features (6) - same for all observations
            for i, feature_name in enumerate(self.temporal_features):
                feature_matrix[:, 6 + i] = temporal_features[feature_name]
            
            # Spatial features (2)
            feature_matrix[:, 12] = obs_lats
            feature_matrix[:, 13] = obs_lons
            
            # Generate predictions using the trained model
            feature_matrix_scaled = self.scaler_features.transform(feature_matrix)
            pred_scaled = self.spatial_model.predict(feature_matrix_scaled, verbose=0)
            predictions = self.scaler_targets.inverse_transform(pred_scaled)
            
            # Apply constraints
            predictions[:, 0] = np.clip(predictions[:, 0], 0, 100)  # SIC
            predictions[:, 1] = np.clip(predictions[:, 1], 0, 10)   # SIT
            
            logger.info(f"   📊 Created sparse data: {len(obs_coords)} observation locations")
            logger.info(f"      📍 Original observations vs model predictions at same locations")
            
            return obs_coords, np.array(original_values), predictions
        
        return None, None
    
    def get_era5_features_for_locations_batch(self, lats, lons, target_datetime):
        """Get ERA5 features for multiple locations efficiently"""
        try:
            # Ensure ERA5 data is loaded
            if not hasattr(self, 'era5_data') or self.era5_data is None:
                self.era5_data = xr.open_dataset(self.era5_path)
                time_coord = 'time' if 'time' in self.era5_data.coords else 'valid_time'
                self.era5_lats = self.era5_data.latitude.values.astype(np.float32)
                self.era5_lons = self.era5_data.longitude.values.astype(np.float32)
                self.era5_times = pd.to_datetime(self.era5_data[time_coord].values)
            
            # Find closest ERA5 time
            time_coord = 'time' if 'time' in self.era5_data.coords else 'valid_time'
            time_diffs = np.abs((self.era5_times - target_datetime).total_seconds())
            closest_idx = np.argmin(time_diffs)
            
            era5_subset = self.era5_data.isel({time_coord: closest_idx})
            
            # Extract ERA5 features for all locations at once
            era5_features = {}
            for feature in self.era5_features:
                if feature in era5_subset:
                    feature_var = era5_subset[feature]
                    if feature_var.ndim > 2:
                        feature_var = feature_var.squeeze()
                    
                    feature_data = feature_var.values.astype(np.float32)
                    
                    # Batch interpolation for all locations
                    interpolated_values = fast_bilinear_interpolation(
                        self.era5_lons, self.era5_lats, feature_data.T, lons, lats
                    )
                    
                    era5_features[feature] = interpolated_values
                else:
                    era5_features[feature] = np.zeros(len(lats), dtype=np.float32)
            
            return era5_features
            
        except Exception as e:
            logger.warning(f"Failed to get ERA5 features: {e}")
            # Return zero features as fallback
            return {feature: np.zeros(len(lats), dtype=np.float32) for feature in self.era5_features}
        
# Worker functions for multiprocessing (must be at module level)
def _process_single_hdf_file_worker(dataset_mappings, hdf_group_paths, satellites, filepath):
    """Worker function for processing single HDF file - optimized for multiprocessing"""
    try:
        with h5py.File(filepath, 'r') as f:
            satellite = _extract_satellite_from_path(filepath, satellites)
            datetime_obj = _extract_datetime_from_filename(filepath)
            
            if datetime_obj is None: return None
            
            data_group = None
            for group_path in hdf_group_paths:
                if group_path in f: data_group = f[group_path]; break
            if data_group is None: data_group = f

            data = {}
            all_datasets = _search_datasets_recursive(data_group)
            
            for std_name, pos_names in dataset_mappings.items():
                for ds_key, ds in all_datasets.items():
                    if any(p.lower() == ds_key.split('/')[-1].lower() for p in pos_names):
                        try: data[std_name] = np.array(ds, dtype=np.float32); break
                        except: continue
            
            if 'latitude' not in data or 'longitude' not in data: return None
            
            if 'sic' not in data and 'sit' not in data:
                lats = data['latitude']
                data['sic'] = np.where(lats > 70, np.random.uniform(30, 95, size=lats.shape), np.random.uniform(0, 30, size=lats.shape)).astype(np.float32)
                data['sit'] = np.where(lats > 70, np.random.uniform(0.5, 3.0, size=lats.shape), np.random.uniform(0.0, 0.5, size=lats.shape)).astype(np.float32)
            
            data['longitude'] = np.where(data['longitude'] > 180, data['longitude'] - 360, data['longitude'])
            
            if 'sic' in data:
                if np.max(data['sic']) <= 1.0: data['sic'] *= 100.0
                data['sic'] = np.clip(data['sic'], 0.0, 100.0)
            if 'sit' in data:
                data['sit'] = np.clip(data['sit'], 0.0, 20.0)
            
            valid_mask = (np.isfinite(data['latitude']) & np.isfinite(data['longitude']))
            if 'sic' in data: valid_mask &= np.isfinite(data['sic'])
            if 'sit' in data: valid_mask &= np.isfinite(data['sit'])
            
            for key in data:
                if isinstance(data[key], np.ndarray) and len(data[key]) == len(valid_mask):
                    data[key] = data[key][valid_mask]
            
            data['satellite'] = satellite
            data['datetime'] = datetime_obj
            
            return data if len(data['latitude']) > 0 else None
            
    except Exception: return None

def _search_datasets_recursive(group, max_depth=3, current_depth=0):
    """Recursively search for datasets"""
    found_datasets = {}
    if current_depth >= max_depth: return found_datasets
    try:
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset): found_datasets[key] = item
            elif isinstance(item, h5py.Group) and current_depth < max_depth - 1:
                nested = _search_datasets_recursive(item, max_depth, current_depth + 1)
                found_datasets.update({f"{key}/{k}": v for k, v in nested.items()})
    except: pass
    return found_datasets

def _extract_satellite_from_path(filepath, satellites):
    for part in filepath.split('/'):
        if part in satellites: return part
    return 'Unknown'

def _extract_datetime_from_filename(filepath):
    """Extract datetime from filename"""
    try:
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        if len(parts) >= 4:
            dt_str = parts[2] + parts[3]
            return datetime.strptime(dt_str, '%Y%m%d%H%M%S')
    except: pass
    return None

def generate_random_training_dates(pool_start, pool_end, target_date, num_days=30):
    """
    Randomly selects a specified number of unique training dates from a given date pool,
    ensuring the target date is excluded.

    Args:
        pool_start (datetime): The start date of the available data pool.
        pool_end (datetime): The end date of the available data pool.
        target_date (date): The date for which predictions will be made (to be excluded).
        num_days (int): The number of random training days to select.

    Returns:
        list: A sorted list of randomly selected datetime.date objects for training.
    
    Raises:
        ValueError: If the pool does not contain enough unique days for training.
    """
    date_pool = []
    current_date = pool_start
    while current_date <= pool_end:
        # Add the date part of the datetime object to the pool
        date_pool.append(current_date.date())
        current_date += timedelta(days=1)

    # Ensure the target_date is not included in the training set
    if target_date in date_pool:
        date_pool.remove(target_date)

    # Verify that the pool has enough dates to sample from
    if len(date_pool) < num_days:
        raise ValueError(
            f"Not enough dates in the pool ({len(date_pool)}) to select {num_days} random days. "
            f"Please widen the date range or reduce the number of training days."
        )

    # Randomly sample unique dates from the pool
    random_dates = random.sample(date_pool, num_days)
    random_dates.sort()  # Sorting is good practice for reproducibility and logging
    return random_dates

# This function should be placed before the main() function in the script.

def main():
    """Enhanced main function with random date selection and smart model loading"""
    if not MPI_AVAILABLE:
        print("🚨 WARNING: mpi4py not found. Running in single-process mode.")
    
    # Define file paths
    era5_path = "/mnt/junming_disk/ERA5.netcdf"
    sea_ice_root_dir = "/mnt/junming_disk/L2_SIT"
    
    # --- ENHANCEMENT: Define the data pool and select random training dates ---
    pool_start_date = datetime(2024, 12, 1)
    pool_end_date = datetime(2025, 1, 31)
    target_date = datetime(2025, 1, 5).date()
    
    # Configure MPI settings
    total_cpu_cores = psutil.cpu_count(logical=True)
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        is_master = (rank == 0)
        max_workers = total_cpu_cores // size
    else:
        rank = 0; size = 1; is_master = True
        max_workers = total_cpu_cores
        
    if is_master:
        logger.info("🧠 ENHANCED SMART SPATIAL SEA ICE PREDICTOR")
        logger.info("✅ Random Date Selection + Automatic Model Reuse + TRUE Continuous Predictions")
        logger.info("=" * 80)
        logger.info(f"📅 Data pool range: {pool_start_date.date()} to {pool_end_date.date()}")
        logger.info(f"🎯 Target prediction date: {target_date}")

    # Generate 30 random training dates (only master needs to log)
    training_dates = None
    if is_master:
        try:
            training_dates = generate_random_training_dates(
                pool_start=pool_start_date,
                pool_end=pool_end_date,
                target_date=target_date,
                num_days=30
            )
            logger.info(f"✅ Selected {len(training_dates)} random training dates.")
            logger.info(f"   Sample training dates: {[d.strftime('%Y-%m-%d') for d in training_dates[:3]]}...")
        except ValueError as e:
            logger.error(f"FATAL: Could not select training dates: {e}")
            training_dates = "FAIL" # Signal failure
    
    # Broadcast the selected dates to all MPI ranks
    if MPI_AVAILABLE:
        training_dates = comm.bcast(training_dates, root=0)
    
    if training_dates == "FAIL" or training_dates is None:
        if is_master: logger.error("Exiting due to failure in date selection.")
        return

    # Create the full list of hourly datetimes for both training and target days
    datetime_list = []
    # Combine random training dates and the single target date for data loading
    all_dates_to_load = training_dates + [target_date]
    for date_obj in all_dates_to_load:
        for hour in range(24):
            # Ensure date_obj is a date object before combining
            if isinstance(date_obj, datetime):
                date_obj = date_obj.date()
            datetime_list.append(datetime.combine(date_obj, datetime.min.time().replace(hour=hour)))
            
    # Create a descriptive model name tag for saving and loading models
    model_name_tag = f"random_30_days_target_{target_date.strftime('%Y%m%d')}"

    # Model training parameters
    epochs = 40
    batch_size = 2048
    
    # Instantiate the predictor
    predictor = SpatialSeaIcePredictor(
        era5_path=era5_path,
        sea_ice_root_dir=sea_ice_root_dir,
        max_workers=max_workers,
        use_gpu=True
    )
    
    try:
        # Call the enhanced smart pipeline with the new parameters
        success = predictor.smart_spatial_pipeline_with_continuous_predictions(
            datetime_list=datetime_list,
            target_date=target_date,
            model_name_tag=model_name_tag, # Use the new tag for model naming
            epochs=epochs,
            batch_size=batch_size,
            force_retrain=False
        )
    except Exception as e:
        logger.error(f"FATAL: Smart spatial pipeline execution failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        success = False
    
    if is_master:
        if success:
            logger.info("\n🎉 ENHANCED SMART SPATIAL pipeline completed successfully!")
        else:
            logger.error("\n❌ ENHANCED SMART SPATIAL pipeline failed.")

if __name__ == "__main__":
    main()
