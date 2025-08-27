A high-performance machine learning system for predicting Arctic sea ice concentration and thickness using ERA5 climate data. This system employs spatial regression models.
Features
The system uses deep neural networks optimized for spatial prediction tasks with MPI parallelization for distributed processing across multiple nodes. Intelligent model saving and loading capabilities avoid unnecessary retraining, while continuous spatial predictions generate true continuous outputs on regular Arctic grids. Advanced visualization creates detailed comparison dashboards and 24-hour overview displays.
System Requirements
Core dependencies include numpy, pandas, xarray, scipy, scikit-learn, and tensorflow for scientific computing and machine learning. Geospatial visualization requires cartopy and matplotlib, while data handling uses h5py and netcdf4. Performance optimization relies on numba and joblib, with optional mpi4py for parallel processing. Hardware requirements include a minimum of 16 GB RAM and 4 CPU cores, though 64+ GB RAM, 16+ CPU cores, and NVIDIA GPU with 8+ GB VRAM are recommended. Approximately 100 GB of storage is needed for processed data cache.
Installation
Clone the repository and install dependencies using pip with the provided requirements file. For parallel processing, install MPI using your system package manager (mpich on Ubuntu/Debian, mpich-devel on CentOS/RHEL) followed by mpi4py installation via pip.
Data Requirements
The system requires ERA5 climate data as a single NetCDF file containing variables cc, pv, r, t, u, v with latitude, longitude, and time coordinates. Sea ice data should be organized in HDF5 files. The system automatically handles data quality filtering, coordinate standardization, missing value imputation, and temporal alignment between datasets.
Usage
Basic usage involves initializing the SpatialSeaIcePredictor with paths to ERA5 and sea ice data, then calling the enhanced_spatial_pipeline method with date ranges and target prediction date. For MPI parallel execution, use mpirun with the desired number of processes. The smart pipeline automatically detects and reuses previously trained models, skipping training on subsequent runs with the same model configuration.
Configuration
Key parameters include training epochs (default 40-50), batch size (default 2048), CPU workers per MPI rank (auto-detected), GPU acceleration (enabled by default), and grid resolution for continuous predictions (default 0.5 degrees). The spatial regression model uses 14 input features comprising 6 ERA5 variables, 6 temporal features, and 2 spatial coordinates, outputting sea ice concentration and thickness predictions through dense layers with batch normalization and dropout regularization.
Output
The system generates complete trained models with feature scalers saved as HDF5 and pickle files respectively. Processed sea ice observations are cached for future use. Visualizations include individual hourly prediction maps, complete 24-hour overview displays, and detailed comparison dashboards. Predictions cover sea ice concentration (0-100% coverage) and thickness (0-10 meters) at 0.5-degree resolution across the Arctic region (50°N-90°N) with hourly temporal resolution.
Performance
Memory management processes data in chunks with automatic garbage collection between steps, scaling usage with MPI process count. Computational efficiency utilizes Numba JIT compilation, vectorized interpolation, batch ERA5 processing, and GPU mixed precision training when available. Runtime scales from 2-4 hours for one month of data on 4 processes to 8+ hours for six months on 16+ processes.
Troubleshooting
Memory errors can be resolved by reducing batch size, increasing MPI processes, or enabling data chunking. GPU issues may require disabling GPU acceleration, checking CUDA compatibility, or monitoring memory usage. File access problems often relate to directory permissions, HDF5 file integrity, or insufficient disk space. The system provides detailed logging with separate files for each MPI rank.
Citation
If you use this code in your research, please cite appropriately with author names, publication year, and repository URL.
Support
For questions and issues, create GitHub issues for bugs, consult the troubleshooting section, and review detailed log files for error information.
