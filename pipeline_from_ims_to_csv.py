"""
1) Extract tiff series from the Imaris file
2) register to the 10um Allen atlas
3) detect cells (cellfinder + deepblink + dbscan)
4) classify cells using the base model or a fine-tuned ResNet model
5) remove background detections (optional)
6) map spots to the atlas space, create a spreadsheet
7) combine with metadata, save resulting csv

requirements:

download and extract pre-packed conda environments (as described in the README.md)
OR
create them manually (not recommended):

######### main env ##########
conda create -y -n morphine_cfos_paper_main python=3.8
conda activate morphine_cfos_paper_main
pip install cellfinder==0.4.21
pip install --upgrade cellfinder-core==0.3.0
pip install --upgrade --upgrade-strategy only-if-needed dask==2022.5.0
pip install --upgrade --upgrade-strategy only-if-needed imaris_ims_file_reader
pip install --upgrade --upgrade-strategy only-if-needed ulid-py==1.1.0

######### deepblink env ##########
conda create -y -n morphine_cfos_paper_deepblink python=3.8
conda activate morphine_cfos_paper_deepblink
conda install -y tensorflow-gpu=2.4.1
pip install --upgrade --upgrade-strategy only-if-needed deepblink==0.1.4

######### fastai env ##########
conda create -y -n morphine_cfos_paper_fastai python=3.9
conda activate morphine_cfos_paper_fastai
conda install -y pytorch=1.12.1=gpu_cuda113py39h19ae3d8_1
pip install --upgrade --upgrade-strategy only-if-needed fastai==2.7.10
pip install --upgrade --upgrade-strategy only-if-needed imaris_ims_file_reader
"""

import json
import os
import re
import subprocess
import sys
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile

# # Brainglobe deps
import bg_space as bgs
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from cellfinder.analyse.analyse import transform_points_to_downsampled_space
from cellfinder.main import get_downsampled_space
from imlib.IO.cells import get_cells

import dask
import dask.array as da
import ulid
from imaris_ims_file_reader import ims

from scipy.ndimage import zoom
from numpy.lib.stride_tricks import as_strided


ims_file_path = sys.argv[1]
if not os.path.exists(ims_file_path):
    ims_file_path = os.path.join(os.path.dirname(__file__), ims_file_path)

# read settings for processing
settings_file = os.path.join(os.path.dirname(ims_file_path), 'settings.json')
with open(settings_file, 'r') as f:
    options = json.load(f)

out_name = os.path.join(os.path.dirname(ims_file_path), 'analysis')

PYTORCH_MODEL_NAME = os.path.basename(options["model_path"])
PYTORCH_MODEL_VERSION = options["model_version"]
DEEPBLINK_MODEL_PATH = os.path.join(os.path.dirname(ims_file_path), 'deepblink_particle.h5')
DEEPBLINK_CHUNK_SIZE = (40, 1700, 3500)
LOW_MEMORY = True
DEEPBLINK_ENV_NAME = "morphine_cfos_paper_deepblink"
DEEPBLINK_ENV_PATH = os.path.join(os.getcwd(), 'morphine_cfos_paper_deepblink')
FASTAI_ENV_NAME = "morphine_cfos_paper_fastai"
FASTAI_ENV_PATH = os.path.join(os.getcwd(), 'morphine_cfos_paper_fastai')


if not os.path.exists(out_name):
    os.makedirs(out_name)


# ===========================================
# Extract tiff series from the Imaris file
# ===========================================
def ensure_tiffs_extracted(options, channel=0):
    parent_folder = os.path.join(out_name, "resolution_level_x")
    tiff_series_dir = os.path.join(parent_folder, f"channel_{channel}")
    extracted_tiffs = glob(os.path.join(tiff_series_dir, '*.tif'))
    if len(extracted_tiffs) < options['shape'][0]:
        ims_file = ims(ims_file_path)
        ims_file.save_Tiff_Series(
            location=tiff_series_dir,
            channels=(channel,),
            resolutionLevel=options["resolution_level"],
            overwrite=True
        )


# ===========================================
# Register to the 10um Allen atlas
# ===========================================
def register_brain_one_channel(options, channel=0, folder_prefix=None):
    """
    Register specified channel.

    :param options, dict
      example: options = {
        "out_name": "/path/to/analysis/ims_file_name_folder/",
        "allen_resolution": 10,
        "resolution": [10, 10, 10],
        "orientation": "sal",
    }
    """
    def launch_registration(input_folder, output_folder, atlas, resolution, orientation, run_in_background=False):
        #  Run brainreg
        print('Starting registration CL tool ...')

        cmd = [
            'brainreg', input_folder, output_folder,
            '-v', str(resolution[0]), str(resolution[1]), str(resolution[2]),
            '--orientation', orientation,
            '--atlas', atlas,
            '--save-original-orientation'
        ]

        if run_in_background:
            subprocess.Popen(cmd)
            return

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print('Registration failed')
            return

        print('Finished registration')

    print(f"Registering channel {channel}")
    analysis_dir_this_brain = out_name
    atlas = f'allen_mouse_{options["allen_resolution"]}um'
    orientation = options["orientation"]
    out_directory = os.path.join(analysis_dir_this_brain, 'resolution_level_x')

    if folder_prefix:
        # registering pre-processed data
        brainreg_input = os.path.join(out_directory, folder_prefix)
        brainreg_output_folder = os.path.join(out_directory, f"registration_{folder_prefix}")
        print(f"Registering pre-processed data at {brainreg_output_folder}")
        if not os.path.exists(os.path.join(brainreg_output_folder, 'registered_atlas.tiff')):
            launch_registration(brainreg_input, brainreg_output_folder, atlas, options["resolution"], orientation)
        else:
            print(f'{brainreg_input}: Registered atlas exists. Skipping registration.')

    # registering original (not pre-processed) data
    brainreg_output_folder = os.path.join(out_directory, f'registration_channel_{channel}')
    brainreg_input = os.path.join(out_directory, f"channel_{channel}")

    # Extract tiff series
    ensure_tiffs_extracted(options, channel=channel)

    if not os.path.exists(os.path.join(brainreg_output_folder, 'registered_atlas.tiff')):
        launch_registration(brainreg_input, brainreg_output_folder, atlas, options["resolution"], orientation)
    else:
        print(f'{brainreg_input}: Registered atlas exists. Skipping registration.')


# ===========================================
# Detect cells
# ===========================================
def detect_cells(options, channel=0):
    def launch_cell_finder(
            signal_folders,
            background_folder,
            output_folder,
            resolution,
            orientation,
            atlas,
            threshold=6,
            soma_diameter=10,
            align=False,
            classify=False,
            path_to_model=None
    ):
        """
        Run cellfinder via subprocess.

        All parameters are described here: https://docs.brainglobe.info/cellfinder/user-guide/command-line
        """
        print(f"Launching cellfinder. Saving to {output_folder}")
        cmd = [
            'cellfinder',
            '-s', *signal_folders, '-b', background_folder, '-o', output_folder,
            '-v', str(resolution[0]), str(resolution[1]), str(resolution[2]),
            '--orientation', orientation,
            '--atlas', atlas.atlas_name,
            '--no-analyse', '--no-figures',
            '--threshold', str(threshold), '--soma-diameter', str(soma_diameter), '--artifact-keep'
        ]
        if not align:
            cmd.append('--no-register')

        if not classify:
            cmd.append('--no-classification')
        elif path_to_model is not None:
            cmd.append('--trained-model')
            cmd.append(str(path_to_model).strip())

        print(f"Cellfinder CMD: {cmd}")

        # TODO: how to run the 'ulimit -n 60000' command before cellfinder?
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            if ret.stdout:
                print(ret.stdout.decode())
            print('Cell detection failed')
            return
        print("Cellfinder: success")

    print('Starting cell detection CL tool...')
    analysis_dir_this_brain = out_name

    atlas = BrainGlobeAtlas(f'allen_mouse_{options["allen_resolution"]}um')
    soma_diameter = options.get('cell_size', 10)  # um
    threshold = options.get('threshold', 6)  # sigmas above mean

    out_directory = os.path.join(analysis_dir_this_brain, 'resolution_level_x')

    signal_channels = [channel]
    signal_folders = []
    for signal_channel in signal_channels:
        signal_folders.append(os.path.join(out_directory, f"channel_{signal_channel}"))
        ensure_tiffs_extracted(options, channel=signal_channel)

    background_folder = signal_folders[0]
    cellfinder_output_folder = os.path.join(out_directory, 'output_full')

    success = launch_cell_finder(
        signal_folders,
        background_folder,
        cellfinder_output_folder,
        options["resolution"],
        options["orientation"],
        atlas,
        threshold=threshold,
        soma_diameter=soma_diameter,
        align=False,
        classify=False
    )

    if not success:
        return

    if len(signal_channels) == 1:
        cellfinder_output_folders = [cellfinder_output_folder]
    else:
        cellfinder_output_folders = []
        for signal_channel in range(len(signal_channels)):
            cellfinder_output_folders.append(os.path.join(cellfinder_output_folder, f"channel_{signal_channel}"))

    for cellfinder_output_folder in cellfinder_output_folders:
        # get all detected spots into one array
        if options.get('classify', False):
            all_detections = get_cells(os.path.join(cellfinder_output_folder, 'points', 'cell_classification.xml'), cells_only=True)
        else:
            all_detections = get_cells(os.path.join(cellfinder_output_folder, 'points', 'cells.xml'))

        all_detected_spots = []

        for cell in all_detections:
            all_detected_spots.append([cell.z, cell.y, cell.x])

        all_detected_spots = np.asarray(all_detected_spots)
        np.save(os.path.join(cellfinder_output_folder, 'all_detected_spots.npy'), all_detected_spots)

    print('Finished detecting cells')


def detect_cells_nn(options, channel=0):
    def launch_deepblink(model_path, img_path):
        cmd = [
            'conda', 'run', '-p', DEEPBLINK_ENV_PATH, 'deepblink', 'predict', '--model', model_path, '--input', img_path
        ]
        ret = subprocess.run(cmd)

    def detect_cells_deepblink(chunks_folder):
        print("Splitting stack into chunks")
        files = sorted(glob(os.path.join(chunks_folder, '*.tif')))
        import random
        random.shuffle(files)
        print("Starting detection")
        for file in files:
            if os.path.exists(os.path.join(chunks_folder, os.path.basename(file).replace('tif', 'csv'))):
                print(f"Skipping chunk {os.path.basename(file).replace('tif', 'csv')}")
                continue

            print(f"Detecting cells in chunk {os.path.basename(file)}")
            tstl = datetime.now()
            launch_deepblink(DEEPBLINK_MODEL_PATH, file)
            print("Chunk finished in", datetime.now() - tstl)

    def convert_to_napari_format(chunks_folder):
        """
        Change columns in csv file to make it readable with napari.

        :param chunks_folder:
        :return:
        """
        csv_files = sorted(glob(os.path.join(chunks_folder, 'chunk*.csv')))
        for csv_file in csv_files:
            if csv_file.startswith('napari'):
                continue
            napari_csv_file_path = os.path.join(os.path.dirname(csv_file), f"napari_{os.path.basename(csv_file)}")
            if os.path.exists(napari_csv_file_path):
                continue
            try:
                df = pd.read_csv(csv_file)
            except pd.errors.EmptyDataError as e:
                print("Warning: ", e)
                continue
            df2 = pd.DataFrame()
            df2['index'] = list(range(df.shape[0]))
            try:
                zvals = df['z'].tolist()
                yvals = df['y [px]'].tolist()
                xvals = df['x [px]'].tolist()
            except KeyError as e:
                print(e)
                continue

            df2['axis-0'] = zvals
            df2['axis-1'] = xvals
            df2['axis-2'] = yvals
            df2.to_csv(napari_csv_file_path)

    def get_origin_coords(ndim, patchify_chunks_shape, chunk_size):
        """
        Get coordinates of each chunk origin.

        TODO: only 3D now, make compatible with 2D

        :param ndim:
        :param chunk_shape:
        :param patches_shape:
        :return:
        """
        coords_shape = list(patchify_chunks_shape[:ndim]) + [ndim]
        coords = np.empty(coords_shape, dtype=np.uint16)
        for z in range(coords.shape[0]):
            for y in range(coords.shape[1]):
                for x in range(coords.shape[2]):
                    coords[z, y, x, :] = np.array((
                        z * chunk_size[0],
                        y * chunk_size[1],
                        x * chunk_size[2]
                    ))
        coords = np.reshape(coords, (np.prod(coords.shape[:ndim]), ndim))
        return coords

    def merge_df(chunks_folder, origin_coords):
        """
        Transform coordinates from chunk space to raw image space.

        :return:
        """

        df_column_names = ['index', 'axis-0', 'axis-1', 'axis-2']  # TODO: ndim
        df = pd.DataFrame(columns=df_column_names)
        csv_files = sorted(glob(os.path.join(chunks_folder, 'napari*.csv')))
        for chunk_file in csv_files:
            current_chunk = int(re.findall(r"\d+", os.path.basename(chunk_file))[-1])
            chunk_df = pd.read_csv(chunk_file)
            if chunk_df.empty:
                continue
            chunk_df_corrected = pd.DataFrame()
            z_values = chunk_df[['axis-0']].to_numpy()
            y_values = chunk_df[['axis-1']].to_numpy()
            x_values = chunk_df[['axis-2']].to_numpy()
            z_values += origin_coords[current_chunk, 0]
            y_values += origin_coords[current_chunk, 1]
            x_values += origin_coords[current_chunk, 2]
            chunk_df_corrected['index'] = list(range(chunk_df.shape[0]))
            chunk_df_corrected['axis-0'] = z_values
            chunk_df_corrected['axis-1'] = y_values
            chunk_df_corrected['axis-2'] = x_values
            df = pd.concat([df, chunk_df_corrected])

        df['index'] = list(range(df.shape[0]))
        return df

    def get_chunk_indices(origin_coords, chunk_size):
        indices = []
        for origin in list(origin_coords):
            indices.append([
                slice(origin[0], origin[0] + chunk_size[0], 1),
                slice(origin[1], origin[1] + chunk_size[1], 1),
                slice(origin[2], origin[2] + chunk_size[2], 1)
            ])
        return indices

    def dask_chunks_to_tiffs(lazy_tiff_stack, chunk_indices, output_folder):
        for ind, slices in enumerate(chunk_indices):
            chunk_file = os.path.join(output_folder, f"chunk_{str(ind).zfill(5)}.tif")
            if not os.path.exists(chunk_file):
                chunk = lazy_tiff_stack[tuple(slices)]
                print(chunk.shape)
                chunk = chunk.compute()
                if chunk.shape != DEEPBLINK_CHUNK_SIZE:
                    chunk_padded = np.zeros(DEEPBLINK_CHUNK_SIZE, dtype='uint16')
                    chunk_padded[:chunk.shape[0], :chunk.shape[1], :chunk.shape[2]] = chunk
                    chunk = chunk_padded
                tifffile.imwrite(chunk_file, chunk.astype('uint16'))

    def patchify_fn(arr_in, window_shape, step):
        """
        Function borrowed from patchify, modified to accept n-dimensional step parameter.

        :param arr_in:
        :param window_shape:
        :param step:
        :return:
        """
        arr_shape = np.array(arr_in.shape)
        window_shape = np.array(window_shape, dtype=arr_shape.dtype)

        # -- build rolling window view
        slices = tuple(slice(None, None, st) for st in step)
        window_strides = np.array(arr_in.strides)

        indexing_strides = arr_in[slices].strides

        win_indices_shape = (
                                    (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
                            ) + 1

        new_shape = tuple(list(win_indices_shape) + list(window_shape))
        strides = tuple(list(indexing_strides) + list(window_strides))

        arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
        return arr_out

    def tiff_series_to_ram(folder, img_shape, img_dtype):
        print("Reading tiff series to RAM")
        files = sorted(glob(os.path.join(folder, '*.tif')))
        if len(files) < img_shape[0]:
            print(f"Incomplete tiff series at {folder}. Skipping.")
            return
        tiffstack = np.empty(img_shape, dtype=img_dtype)

        def read_img(img):
            # print("Reading", img)
            return tifffile.imread(img)

        def save_img(z, img):
            # print("Saving", z)
            tiffstack[z, :, :] = img
            return True

        imgs = [dask.delayed(read_img)(i) for i in files]
        saved = [dask.delayed(save_img)(z, i) for z, i in enumerate(imgs)]
        saved = dask.compute(saved)
        return tiffstack

    def split_in_chunks_nd_no_overlap(img, chunk_shape):
        ndim = len(img.shape)
        ratios = (np.array(img.shape) / np.array(chunk_shape)).astype(int)
        ratios += 1
        larger_img = np.zeros((ratios * chunk_shape), img.dtype)
        if ndim == 2:  # TODO rewrite
            larger_img[:img.shape[0], :img.shape[1]] = img
        elif ndim == 3:
            larger_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        patches = patchify_fn(larger_img, chunk_shape, step=chunk_shape)
        patches_shape = np.array(patches.shape)
        new_patches_shape = (np.prod(patches_shape[:ndim]), *patches_shape[ndim:])
        return patches.reshape(new_patches_shape)

    def chunks_to_tiffs(chunks, output_folder):
        """
        Save chunks (result of patchify) to disk.

        :param chunks:
        :param output_folder:
        :return:
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for ind, chunk in enumerate(list(chunks)):
            chunk_file = os.path.join(output_folder, f"chunk_{str(ind).zfill(5)}.tif")
            if not os.path.exists(chunk_file):
                tifffile.imwrite(chunk_file, chunk)

    print("Starting NN cell detection ...")
    tst = datetime.now()
    analysis_dir_this_brain = out_name
    out_directory = os.path.join(analysis_dir_this_brain, 'resolution_level_x')
    signal_channels = [channel]
    signal_folders = []
    for signal_channel in signal_channels:
        signal_folders.append(os.path.join(out_directory, f"channel_{signal_channel}"))

    deepblink_folder = os.path.join(out_directory, 'output_deepblink')
    if not os.path.exists(deepblink_folder):
        os.makedirs(deepblink_folder)

    for ind, signal_channel in enumerate(signal_channels):
        out_csv_path = os.path.join(deepblink_folder, f'channel_{signal_channel}_cells.csv')
        if os.path.exists(out_csv_path):
            print(f"Output dataframe already exists at {out_csv_path}. Skipping.")
            continue
        ratios = (np.array(options['shape']) / np.array(DEEPBLINK_CHUNK_SIZE)).astype('int') + 1
        patchify_chunks_shape = (*list(ratios), *DEEPBLINK_CHUNK_SIZE)
        origin_coords = get_origin_coords(3, patchify_chunks_shape, DEEPBLINK_CHUNK_SIZE)
        chunks_folder = os.path.join(out_directory, f"channel_{signal_channel}_chunks")
        if not os.path.exists(chunks_folder):
            os.makedirs(chunks_folder)
        np.save(
            os.path.join(chunks_folder, "origin_coords.npy"),
            origin_coords
        )

        if LOW_MEMORY:
            print("Reading chunks from Imaris")
            ims_file = ims(ims_file_path, ResolutionLevelLock=options['resolution_level'])
            ims_file_dask = da.array(ims_file)
            tiffstack = ims_file_dask[0,signal_channel,:,:,:]
            chunk_indices = get_chunk_indices(origin_coords, DEEPBLINK_CHUNK_SIZE)
            dask_chunks_to_tiffs(tiffstack, chunk_indices, chunks_folder)
        else:
            tiffstack = tiff_series_to_ram(signal_folders[ind], options['shape'], np.uint16)
            if tiffstack is None:
                continue
            chunks = split_in_chunks_nd_no_overlap(tiffstack, DEEPBLINK_CHUNK_SIZE)
            chunks_to_tiffs(chunks, chunks_folder)

        detect_cells_deepblink(chunks_folder)
        convert_to_napari_format(chunks_folder)
        df = merge_df(chunks_folder, origin_coords)
        df.to_csv(out_csv_path)

    tfi = datetime.now()
    print("Total time spent on detection", tfi - tst)


def merge_detected_cells(options, channel=0):
    """
    Combines deepblink and cellfinder output
    Then runs dbscan on combined output, to remove duplication
    :return:
    """

    def run_dbscan_on_chunk(df):
        print("Clustering detections...")
        from sklearn.cluster import DBSCAN
        points = df[["axis-0", "axis-1", "axis-2"]].to_numpy()

        eps = 3
        min_samples = 2

        # create a DBSCAN object
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        # fit the points to the model
        dbscan.fit(points)

        # get the cluster assignments for each point
        labels = dbscan.labels_

        # get the unique cluster labels
        cluster_labels = np.unique(labels)
        print("Total clusters:", len(cluster_labels))

        # calculate the centroid of each cluster
        def get_cluster_centroid(label):
            points_in_cluster = points[labels == label]
            centroid = np.mean(points_in_cluster, axis=0)
            return centroid

        centroids = [dask.delayed(get_cluster_centroid)(x) for x in cluster_labels]
        cluster_centroids = dask.compute(centroids)[0]
        print(type(cluster_centroids))
        print(len(cluster_centroids))

        cluster_centroids = np.array(cluster_centroids)
        print(cluster_centroids.shape)
        print(cluster_centroids.dtype)
        df = pd.DataFrame()
        df['axis-0'] = cluster_centroids[:, 0]
        df['axis-1'] = cluster_centroids[:, 1]
        df['axis-2'] = cluster_centroids[:, 2]
        return df

    analysis_dir_this_brain = out_name
    signal_channels = [channel]
    signal_channel = options.get("signal_channel", signal_channels[0])
    resolution_level_dir = os.path.join(analysis_dir_this_brain, 'resolution_level_x')
    cellfinder_output_dir = os.path.join(resolution_level_dir, "output_full")
    cellfinder_output_folders = [cellfinder_output_dir]

    deepblink_csv = os.path.join(resolution_level_dir, "output_deepblink", f"channel_{signal_channel}_cells.csv")
    cellfinder_npy = os.path.join(cellfinder_output_dir, "all_detected_spots.npy")
    merged_csv_path = os.path.join(cellfinder_output_dir, "points", "combined_cells_cellfinder_deepblink.csv")
    dbscan_csv_path = os.path.join(cellfinder_output_dir, "points", "combined_cells_cellfinder_deepblink_dbscan_3_2.csv")
    if os.path.exists(dbscan_csv_path):
        print("Merged and clustered cell DF already exists")
        return

    def restore_cellfinder_points():
        points_dir = cellfinder_output_folders[signal_channels.index(signal_channel)]
        all_detections = get_cells(os.path.join(points_dir, 'points', 'cells.xml'))
        all_detected_spots = []

        for cell in all_detections:
            all_detected_spots.append([cell.z, cell.y, cell.x])

        all_detected_spots = np.asarray(all_detected_spots)
        np.save(os.path.join(points_dir, 'all_detected_spots.npy'), all_detected_spots)
        np.save(cellfinder_npy, all_detected_spots)

    print("Restoring cellfinder points")
    restore_cellfinder_points()
    df = pd.read_csv(deepblink_csv)
    deepblink_points = df[['axis-0', 'axis-1', 'axis-2']].to_numpy()
    cellfinder_points = np.load(cellfinder_npy)
    print("Merging deepblink + cellfinder points")
    merged_points = np.concatenate([deepblink_points, cellfinder_points])

    merged_df = pd.DataFrame()
    merged_df['axis-0'] = merged_points[:,0]
    merged_df['axis-1'] = merged_points[:,1]
    merged_df['axis-2'] = merged_points[:,2]
    merged_df.to_csv(merged_csv_path)

    dbscan_df = run_dbscan_on_chunk(merged_df)
    print("Saving clustered points")
    dbscan_df.to_csv(dbscan_csv_path)
    try:
        os.rename(cellfinder_npy, os.path.join(os.path.dirname(cellfinder_npy), f"cellfinder_{os.path.basename(cellfinder_npy)}"))
    except:
        pass
    points = dbscan_df[['axis-0', 'axis-1', 'axis-2']].to_numpy()
    np.save(cellfinder_npy, points)


def classify_cells_fastai(options, channel=0):
    """
    Run cell classification via fastai.

    :param options: dict
    :return: None
    """
    print("Starting cell classification... (might take an hour)")
    cmd = ['conda', 'run', '-p', FASTAI_ENV_PATH, 'python', 'classify_cells_fastai.py', os.path.abspath(settings_file)]
    subprocess.run(cmd)


def remove_background_detections(options):
    """
    Remove false positive cell detections in the background.

    Runs on the classification results (TODO: run before classification, to save time)

    :param options:
    :return:
    """
    print("Removing background detections...")
    cellfinder_output_folder = os.path.join(
        out_name,
        'resolution_level_x',
        "output_full"
    )
    path_to_classification_df = glob(
        os.path.join(
            cellfinder_output_folder,
            'points',
            f'predictions_*_*.csv'
        )
    )[-1]
    df = pd.read_csv(path_to_classification_df)
    combined_filtered_cells_df = []
    combined_filtered_non_cells_df = []
    mask_folder = os.path.join(out_name, 'resolution_level_x', 'bg_fg_mask_apoc')
    save_results_to = os.path.join(cellfinder_output_folder, 'points')

    def process(df_part):
        points = df_part[['axis-1', 'axis-2']].to_numpy()
        detected_cells_np = np.floor(points).astype(int)
        # create a binary image where points are represented as pixels with value 1, the rest of the image is 0
        cells_binary = np.zeros(options['shape'][1:], dtype=np.uint8)
        np.put(cells_binary, np.ravel_multi_index(detected_cells_np.T, options['shape'][1:]), 1)
        # read the mask
        # multiply cells image by mask
        cells_filtered = cells_binary * mask
        # get indices of points that get removed, "unravel" them
        # convert binary image back to points
        nz = np.nonzero(cells_filtered)
        zipped_nz = list(zip(*nz))
        filtered_cells_np = np.asarray(zipped_nz)
        # save points as dataframe
        filtered_cells_df = pd.DataFrame()
        if filtered_cells_np.shape[0]:
            filtered_cells_df['axis-0'] = [z] * filtered_cells_np.shape[0]
            filtered_cells_df['axis-1'] = list(filtered_cells_np[:, 0])
            filtered_cells_df['axis-2'] = list(filtered_cells_np[:, 1])
        return filtered_cells_df

    # loop through z
    for z in range(options['shape'][0]):
        # select all points at current z
        partial_df = df[df["axis-0"] == z]
        partial_df_cells = partial_df[partial_df["nn_decoded"] == 'cell']
        partial_df_non_cells = partial_df[partial_df["nn_decoded"] == 'non_cell']
        mask = tifffile.imread(os.path.join(mask_folder, f"{str(z).zfill(5)}.tif"))

        filtered_cells_df = pd.DataFrame()
        filtered_non_cells_df = pd.DataFrame()
        if partial_df_cells.shape[0]:
            filtered_cells_df = process(partial_df_cells)
        if partial_df_non_cells.shape[0]:
            filtered_non_cells_df = process(partial_df_non_cells)

        # append remaining points to a new dataframe
        combined_filtered_cells_df.append(filtered_cells_df)
        combined_filtered_non_cells_df.append(filtered_non_cells_df)

    print("Saving dfs to csv")
    combined_filtered_cells_df = pd.concat(combined_filtered_cells_df, ignore_index=True)
    combined_filtered_cells_df.to_csv(
        os.path.join(
            save_results_to,
            f'predicted_cells_{PYTORCH_MODEL_NAME}_{PYTORCH_MODEL_VERSION}_bg_removed.csv'
        )
    )
    combined_filtered_non_cells_df = pd.concat(combined_filtered_non_cells_df, ignore_index=True)
    combined_filtered_non_cells_df.to_csv(
        os.path.join(
            save_results_to,
            f'predicted_non_cells_{PYTORCH_MODEL_NAME}_{PYTORCH_MODEL_VERSION}_bg_removed.csv'
        )
    )
    combined_filtered_cells_df['nn_decoded'] = ['cell'] * combined_filtered_cells_df.shape[0]
    combined_filtered_non_cells_df['nn_decoded'] = ['non_cell'] * combined_filtered_non_cells_df.shape[0]
    combined_cells_noncells = pd.concat([combined_filtered_cells_df, combined_filtered_non_cells_df], ignore_index=True)
    combined_cells_noncells.to_csv(
        os.path.join(
            cellfinder_output_folder,
            'points',
            f'predictions_{PYTORCH_MODEL_NAME}_{PYTORCH_MODEL_VERSION}_bg_removed.csv'
        )
    )


# ==================================================
# Map spots to the atlas space, create a spreadsheet
# ==================================================
def get_df(analysis_dir, path_to_classification_df, registration_folder, metadata_id):
    def transform_points_downsampled_to_atlas_space(
            downsampled_points, atlas, deformation_field_paths, output_filename=None
    ):
        field_scales = [int(1000 / resolution) for resolution in atlas.resolution]
        points = [[], [], []]
        for axis, deformation_field_path in enumerate(deformation_field_paths):
            deformation_field = tifffile.imread(deformation_field_path)
            for point in downsampled_points:
                try:
                    point = [int(round(p)) for p in point]
                    points[axis].append(
                        int(
                            round(
                                field_scales[axis]
                                * deformation_field[point[0], point[1], point[2]]
                            )
                        )
                    )
                except IndexError:
                    print(
                        f'IndexError when transforming point ({point[0]},{point[1]},{point[2]}) from downsampled to atlas space.'
                    )
        transformed_points = np.array(points).T

        if output_filename is not None:
            df = pd.DataFrame(transformed_points)
            df.to_hdf(output_filename, key="df", mode="w")

        return transformed_points

    deformation_field_paths = [
        os.path.join(registration_folder, 'deformation_field_0.tiff'),
        os.path.join(registration_folder, 'deformation_field_1.tiff'),
        os.path.join(registration_folder, 'deformation_field_2.tiff')
    ]

    atlas = BrainGlobeAtlas('allen_mouse_{}um'.format(options['allen_resolution']))

    source_space = bgs.AnatomicalSpace(
        options['orientation'],
        shape=options['shape'],
        resolution=options['resolution'],
    )

    downsampled_space = get_downsampled_space(
        atlas,
        os.path.join(registration_folder, 'boundaries.tiff')
    )

    classification_df = pd.read_csv(path_to_classification_df)
    classification_df_coords = classification_df[["axis-0", "axis-1", "axis-2"]]
    all_detected_spots = classification_df_coords.to_numpy()
    all_detected_spots_downsampled = transform_points_to_downsampled_space(
        all_detected_spots, downsampled_space, source_space
    )
    all_detected_spots_transformed = transform_points_downsampled_to_atlas_space(
        all_detected_spots_downsampled, atlas, deformation_field_paths
    )

    all_detected_spots_downsampled = np.round(all_detected_spots_downsampled).astype(int)
    # For each point, get atlas label
    label_ids = []
    empty_points = []
    error_points = []
    good_points = []
    df_data = []

    signal_channels = [1]
    signal_channel = signal_channels[0]

    for ind in range(all_detected_spots_transformed.shape[0]):
        label_id = 0
        structure_code = ''
        structure_name = ''

        if np.any(all_detected_spots_transformed[ind] < 0):
            error_points.append([
                all_detected_spots_transformed[ind, 0],
                all_detected_spots_transformed[ind, 1],
                all_detected_spots_transformed[ind, 2]
            ])
            print("Point with negative coordinates")
            continue

        try:  # some points are outside atlas
            atlas_value = atlas.annotation[
                all_detected_spots_transformed[ind, 0],
                all_detected_spots_transformed[ind, 1],
                all_detected_spots_transformed[ind, 2]
            ]
        except IndexError as e:
            error_points.append([
                all_detected_spots_transformed[ind, 0],
                all_detected_spots_transformed[ind, 1],
                all_detected_spots_transformed[ind, 2]
            ])
        else:  # if we didn't get exception
            df_row = atlas.lookup_df.index[atlas.lookup_df['id'] == atlas_value]
            if not df_row.empty:  # such atlas_value exists
                row_values = atlas.lookup_df.iloc[df_row]
                structure_name = row_values['name'].values[0]
                structure_code = row_values['acronym'].values[0]
                label_id = atlas_value
                good_points.append([
                    all_detected_spots_transformed[ind, 0],
                    all_detected_spots_transformed[ind, 1],
                    all_detected_spots_transformed[ind, 2]
                ])
            else:  # no such atlas_value (it's usually 0 in this case)
                empty_points.append([
                    all_detected_spots_transformed[ind, 0],
                    all_detected_spots_transformed[ind, 1],
                    all_detected_spots_transformed[ind, 2]
                ])
        finally:  # it will run either way
            if any([x < 0 for x in all_detected_spots_transformed[ind, :]]):
                label_id = 0
                structure_code = ''
                structure_name = ''

            label_ids.append(label_id)
            data_entry = [
                ulid.new(),  # uuid  # TODO: create when saving imaris points?
                0,  # time_point
                signal_channel,  # channel
                all_detected_spots[ind, 0] * options['resolution'][0],
                # 'z_raw'  # TODO take from original imaris points?
                all_detected_spots[ind, 1] * options['resolution'][1],  # 'y_raw',
                all_detected_spots[ind, 2] * options['resolution'][2],  # 'x_raw',
                'um',  # 'raw_coord_units'
                int(round(all_detected_spots[ind, 0] * options['resolution'][0] / options['full_resolution'][0])),
                # 'z_raw_px'  # TODO take from original imaris points?
                int(round(all_detected_spots[ind, 1] * options['resolution'][1] / options['full_resolution'][1])),
                # 'y_raw_px'
                int(round(all_detected_spots[ind, 2] * options['resolution'][2] / options['full_resolution'][2])),
                # 'x_raw_px'
                1,   # 'is_cell',
                '',  # 'type',
                atlas.atlas_name,  # 'atlas_name',
                options["allen_resolution"],  # 'atlas_resolution',
                all_detected_spots_downsampled[ind, 0],  # 'z_downsampled',
                all_detected_spots_downsampled[ind, 1],  # 'y_downsampled',
                all_detected_spots_downsampled[ind, 2],  # 'x_downsampled',
                all_detected_spots_transformed[ind, 0] * atlas.resolution[0],  # 'z_transformed',
                all_detected_spots_transformed[ind, 1] * atlas.resolution[1],  # 'y_transformed',
                all_detected_spots_transformed[ind, 2] * atlas.resolution[2],  # 'x_transformed',
                'um',  # transformed_coord_units
                int(round(all_detected_spots_transformed[ind, 0])),  # 'z_transformed_px',
                int(round(all_detected_spots_transformed[ind, 1])),  # 'y_transformed_px',
                int(round(all_detected_spots_transformed[ind, 2])),  # 'x_transformed_px',
                structure_name,  # 'atlas_structure_name',
                structure_code,  # 'atlas_structure_acronym',
                label_id,  # 'atlas_structure_number',
                metadata_id,  # 'metadata'
            ]
            df_data.append(data_entry)

    # create a DataFrame
    df_column_names = [
        'uuid',
        'time_point',
        'channel',
        'z_raw',
        'y_raw',
        'x_raw',
        'raw_coord_units',
        'z_raw_px',
        'y_raw_px',
        'x_raw_px',
        'is_cell',
        'type',
        'atlas_name',
        'atlas_resolution',
        'z_downsampled',
        'y_downsampled',
        'x_downsampled',
        'z_transformed',
        'y_transformed',
        'x_transformed',
        'transformed_coord_units',
        'z_transformed_px',
        'y_transformed_px',
        'x_transformed_px',
        'atlas_structure_name',
        'atlas_structure_acronym',
        'atlas_structure_number',
        'metadata'
    ]

    df = pd.DataFrame(df_data, columns=df_column_names)
    output_csv_file_path = os.path.join(analysis_dir, f"df_for_dashboard.csv")

    df = df.astype(
        {"uuid": str, "z_raw": float, "y_raw": float, "x_raw": float, "raw_coord_units": str, "z_raw_px": int,
         "y_raw_px": int, "x_raw_px": int, "is_cell": int, "type": str, "atlas_name": str, "atlas_resolution": str,
         "z_downsampled": int, "y_downsampled": int, "x_downsampled": int, "z_transformed": float,
         "y_transformed": float, "x_transformed": float, "transformed_coord_units": str, "z_transformed_px": int,
         "y_transformed_px": int, "x_transformed_px": int, "atlas_structure_name": str, "atlas_structure_acronym": str,
         "atlas_structure_number": int, "metadata": int}
    )
    df.to_csv(output_csv_file_path)
    return output_csv_file_path


# =====================
# Combine with metadata
# =====================
def join_with_metadata(cells, metadata, metadata_shift=0):
    print("Joining with metadata...")
    df_cells = pd.read_csv(
        cells,
        usecols=[
            "uuid",
            "z_raw",
            "y_raw",
            "x_raw",
            "z_raw_px",
            "y_raw_px",
            "x_raw_px",
            "atlas_name",
            "atlas_resolution",
            "z_downsampled",
            "y_downsampled",
            "x_downsampled",
            "z_transformed",
            "y_transformed",
            "x_transformed",
            "z_transformed_px",
            "y_transformed_px",
            "x_transformed_px",
            "atlas_structure_acronym",
            "atlas_structure_number",
            "metadata",
        ],
    )
    df_cells = df_cells.rename(columns={"uuid": "uuid_cell"})
    dtype_dict = {"time_point": float}
    df_metadata = pd.read_csv(
        metadata,
        usecols=["id", "uuid", "file_path", "treatment", "time_point", "sex"],
        dtype=dtype_dict,
    )
    df_metadata = df_metadata.rename(columns={"uuid": "uuid_brain"})
    df_merged = pd.merge(
        left=df_metadata, right=df_cells, how="inner", left_on="id", right_on="metadata"
    ).drop(["id"], axis=1)
    joint_file = cells.replace(".csv", "_with_metadata.csv")
    df_merged["metadata"] += metadata_shift
    df_merged.to_csv(joint_file, sep="\t", index=False)
    return joint_file


if __name__ == "__main__":
    # # Extract tiff series from the Imaris file
    ensure_tiffs_extracted(options, channel=options['signal_channel'])

    # # register to the 10um Allen atlas
    register_brain_one_channel(options, channel=options['signal_channel'])

    # # detect cells (cellfinder + deepblink + dbscan)
    detect_cells(options, channel=options['signal_channel'])
    detect_cells_nn(options, channel=options['signal_channel'])
    merge_detected_cells(options, channel=options['signal_channel'])

    # # classify cells using fastai ResNet-50
    classify_cells_fastai(options, channel=options['signal_channel'])

    # # remove background detections (optional)
    remove_background_detections(options)

    # # map spots to the atlas space, create a spreadsheet
    brain_params = [
        out_name,
        os.path.join(out_name, 'resolution_level_x', 'output_full', 'points', f'predicted_cells_{PYTORCH_MODEL_NAME}_{PYTORCH_MODEL_VERSION}_bg_removed.csv'),
        os.path.join(out_name, 'resolution_level_x', 'registration_channel_1'),
        options["metadata_id"]
    ]
    df_path = get_df(*brain_params)

    # # combine with metadata, save resulting csv
    metadata = os.path.join(os.path.dirname(settings_file), options['metadata'])
    joint_file = join_with_metadata(df_path, metadata, metadata_shift=options["metadata_shift"])
    print("THE END!")
    print("Your output csv is at", os.path.abspath(joint_file))
