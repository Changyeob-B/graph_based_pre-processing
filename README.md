# graph_based_pre-processing
Python implementation of a graph conversion-based method for pre-processing active travel GNSS trajectories. The method uses node degrees to identify weakly connected trajectories and preserves the original geometry of retained trajectories.

Usage'''
At the top of trajectory_preprocessing.py, set INPUT_FILE to your input GeoPackage and OUTPUT_FILE to where you want to save the retained trajectories.
INPUT_FILE = Path(r"data/input.gpkg")
OUTPUT_FILE = Path(r"results/output.gpkg")

The input should contain line trajectories in a projected coordinate reference system with metre units.
Install the required packages and run the script.
python -m pip install geopandas networkx pyogrio
python trajectory_preprocessing.py
