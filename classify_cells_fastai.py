import os
import json
import sys
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile
from fastai.data.all import CategoryBlock, DataBlock, RandomSplitter, TensorImage, TransformBlock, Tuple, tensor, nn
from fastai.vision.all import vision_learner, resnet50
from fastai.metrics import error_rate
from imaris_ims_file_reader import ims
from scipy.ndimage import zoom

print("Classifying cells...")
settings_file = sys.argv[1]
output_folder = os.path.join(os.path.dirname(settings_file), 'analysis')
f = open(settings_file, 'r')
options = json.load(f)

model_path = os.path.join(os.path.dirname(settings_file), options["model_path"])
model_name = os.path.basename(model_path)
model_version = options["model_version"]

cellfinder_output_folder = os.path.join(
    output_folder,
    'resolution_level_x',
    "output_full"
)
signal_channels = [options['signal_channel']]

data_dirs_root = os.path.join(
    output_folder,
    'resolution_level_x',
)
signal_folders = []
for signal_channel in signal_channels:
    signal_folders.append(os.path.join(data_dirs_root, f"channel_{signal_channel}"))

points_to_classify_npy = os.path.join(output_folder, 'resolution_level_x', 'output_full', 'all_detected_spots.npy')
save_results_to = os.path.join(cellfinder_output_folder, 'points')
if os.path.exists(
        os.path.join(
            save_results_to,
            f'predicted_non_cells_{model_name}_{model_version}.csv'
        )
) and os.path.exists(
    os.path.join(
        save_results_to,
        f'predicted_cells_{model_name}_{model_version}.csv'
    )
):
    sys.exit(0)

tst = datetime.now()
# Transform npy file to napari-compatible csv
points = np.load(points_to_classify_npy)
points_df = pd.DataFrame()
z_values = np.round(points[:, 0]).astype(int)
y_values = np.round(points[:, 1]).astype(int)
x_values = np.round(points[:, 2]).astype(int)
points_df['index'] = list(range(points.shape[0]))
points_df['axis-0'] = z_values
points_df['axis-1'] = y_values
points_df['axis-2'] = x_values
points_csv_name = os.path.join(os.path.dirname(points_to_classify_npy),
                               os.path.basename(points_to_classify_npy).replace('.npy', '_napari.csv'))
points_df.to_csv(points_csv_name)
n_cubes = points_df.shape[0]

df_inference = points_df.copy()
df_inference.rename(columns={'axis-0': 'axis_0', 'axis-1': 'axis_1', 'axis-2': 'axis_2'}, inplace=True)
df_inference['ann'] = ['unknown1'] * (n_cubes // 2) + ['unknown2'] * (n_cubes - n_cubes // 2)

resolution_level = options['resolution_level']
ims_file = ims(os.path.join(os.path.dirname(settings_file), options['ims_file_path']), ResolutionLevelLock=resolution_level)

# Extract cubes the same way it is done in cellfinder
cube_shape2 = (20, 25, 25)
img_shape = ims_file.metaData[resolution_level, 0, 0, 'shape'][-3:]


def get_cube_slicing2(z, y, x):
    """
    Extract cube like cellfinder does. Subsequent zoom needed.
    """
    z_left = max([z - cube_shape2[0] // 2, 0])
    z_right = min([z + cube_shape2[0] // 2, img_shape[0]])
    y_left = max([y - cube_shape2[1] // 2, 0])
    y_right = min([y_left + cube_shape2[1], img_shape[1]])
    x_left = max([x - cube_shape2[2] // 2, 0])
    x_right = min([x_left + cube_shape2[2], img_shape[2]])
    return slice(z_left, z_right, None), slice(y_left, y_right, None), slice(x_left, x_right, None)


# create data loader
print("Creating data loader")


def get_x(r):
    """
    Extract cubes as cellfinder does, with zoom
    """
    slice_z, slice_y, slice_x = get_cube_slicing2(r.axis_0, r.axis_1, r.axis_2)
    raw_cube = np.array(ims_file[resolution_level, 0, 0, slice_z, slice_y, slice_x])
    zoomed_cube = zoom(raw_cube, [1, 2, 2], order=2)
    return zoomed_cube


def get_y(r):
    return r['ann']


def int2float(o: TensorImage):
    return o.float().div_(2 ** 16)


class ImageND(Tuple):
    @classmethod
    def create(cls, nd_image):
        nd_image = nd_image.astype(float)
        nd_image = tensor(nd_image)
        return nd_image


def ImageNDBlock():
    return TransformBlock(type_tfms=ImageND.create, batch_tfms=int2float)


dblock = DataBlock(blocks=(ImageNDBlock, CategoryBlock),
                   get_x=get_x,
                   get_y=get_y,
                   splitter=RandomSplitter(valid_pct=0.1))

dls = dblock.dataloaders(df_inference)

# create a learner
print("Loading model")
learn = vision_learner(dls, resnet50, metrics=error_rate)
nChannels = 20
learn.model[0][0] = nn.Conv2d(nChannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# load model
learn.load(model_path)


def read_tiff_stack(filenames, z_start, z_end):
    """
    Read a stack of TIFF files into a 3D numpy array.
    """
    img = np.expand_dims(tifffile.imread(filenames[z_start:z_end][0]), axis=0)
    for filename in filenames[z_start:z_end][1:]:
        img = np.concatenate((img, np.expand_dims(tifffile.imread(filename), axis=0)), axis=0)
    return img


def read_tiff_plane(filenames, z):
    return np.expand_dims(tifffile.imread(filenames[z]), axis=0)


def extract_boxes(img, partial_df, z_start):
    """
    Extract boxes of a specified size from the 3D numpy array.
    """

    def get_x(r):
        """
        Extract cubes as cellfinder does, with zoom
        """
        slice_z, slice_y, slice_x = get_cube_slicing2(r['axis-0'] - z_start, r['axis-1'],
                                                      r['axis-2'])  # todo: just 10 instead of r.axis_0 - z_start ?
        raw_cube = img[slice_z, slice_y, slice_x]  # todo: img[:, slice_y, slice_x] ?
        zoomed_cube = zoom(raw_cube, [1, 2, 2], order=2)
        return zoomed_cube

    test_files = [get_x(x) for i, x in partial_df.iterrows()]

    return test_files


def sliding_window(filenames, df, box_size):
    """
    Implement the sliding window approach to read the TIFF stack and extract boxes.
    """
    print("df", df.shape)
    out_df = []
    img = None
    total_rows = 0
    for z_start in range(0, len(filenames) - box_size[0] + 1):
        z_end = z_start + box_size[0]
        z_center = z_start + box_size[0] // 2
        partial_df = df[df["axis-0"] == z_center]
        partial_df_complete = partial_df.loc[
            (partial_df['axis-1'] >= box_size[1] // 2)
            & (partial_df['axis-1'] <= img_shape[1] - box_size[1] // 2 - 1)
            & (partial_df['axis-2'] >= box_size[2] // 2)
            & (partial_df['axis-2'] <= img_shape[2] - box_size[2] // 2 - 1)
            ].copy()
        partial_df_incomplete = partial_df.loc[
            (partial_df['axis-1'] < box_size[1] // 2)
            | (partial_df['axis-1'] > img_shape[1] - box_size[1] // 2 - 1)
            | (partial_df['axis-2'] < box_size[2] // 2)
            | (partial_df['axis-2'] > img_shape[2] - box_size[2] // 2 - 1)
            ].copy()
        if img is None:
            img = read_tiff_stack(filenames, z_start, z_end)
        else:
            new_img = read_tiff_plane(filenames, z_end - 1)
            img = np.concatenate((img[1:, :, :], new_img), axis=0)
        boxes = extract_boxes(img, partial_df_complete, z_start)
        if len(boxes):
            test_dl = learn.dls.test_dl(boxes)
            preds, _, decoded_values = learn.get_preds(dl=test_dl, with_decoded=True)
            probabilities = [float(x[0]) for x in preds]
            v = ["cell", "non_cell"]
            decoded_values = [v[x] for x in decoded_values]
            partial_df_complete['nn_decoded'] = decoded_values
            partial_df_complete['prob'] = probabilities
            partial_df_complete['incomplete'] = [False] * partial_df_complete.shape[0]

        partial_df_incomplete['nn_decoded'] = [''] * partial_df_incomplete.shape[0]
        partial_df_incomplete['prob'] = [np.nan] * partial_df_incomplete.shape[0]
        partial_df_incomplete['incomplete'] = [True] * partial_df_incomplete.shape[0]
        partial_df = pd.concat([partial_df_complete, partial_df_incomplete], ignore_index=True)
        out_df.append(partial_df)
        total_rows += partial_df.shape[0]

    out_df = pd.concat(out_df, ignore_index=True)
    return out_df


df_low_z = points_df[points_df['axis-0'] < cube_shape2[0] // 2].copy()
df_low_z['nn_decoded'] = [''] * df_low_z.shape[0]
df_low_z['prob'] = [np.nan] * df_low_z.shape[0]
df_low_z['incomplete'] = [True] * df_low_z.shape[0]

df_high_z = points_df[points_df['axis-0'] > img_shape[0] - cube_shape2[0] // 2].copy()
df_high_z['nn_decoded'] = [''] * df_high_z.shape[0]
df_high_z['prob'] = [np.nan] * df_high_z.shape[0]
df_high_z['incomplete'] = [True] * df_high_z.shape[0]

filenames = sorted(
    glob(os.path.join(output_folder, 'resolution_level_x', f'channel_{signal_channels[0]}', '*.tif')))

df_inference_complete = sliding_window(filenames, points_df, cube_shape2)

df_inference_complete = pd.concat([df_low_z, df_inference_complete, df_high_z], ignore_index=True)

print("Saving outputs")
df_inference_complete.to_csv(
    os.path.join(
        save_results_to,
        f'predictions_{model_name}_{model_version}.csv'
    )
)

df_inference_cells = df_inference_complete.loc[df_inference_complete["nn_decoded"] == "cell"]
df_inference_non_cells = df_inference_complete.loc[df_inference_complete["nn_decoded"] == "non_cell"]

# save classification results to napari compatible csv
cells_df = pd.DataFrame()
cells_df['index'] = list(range(df_inference_cells.shape[0]))
cells_df['axis-0'] = df_inference_cells['axis-0'].to_list()
cells_df['axis-1'] = df_inference_cells['axis-1'].to_list()
cells_df['axis-2'] = df_inference_cells['axis-2'].to_list()
cells_df.to_csv(
    os.path.join(
        save_results_to,
        f'predicted_cells_{model_name}_{model_version}.csv'
    )
)

non_cells_df = pd.DataFrame()
non_cells_df['index'] = list(range(df_inference_non_cells.shape[0]))
non_cells_df['axis-0'] = df_inference_non_cells['axis-0'].to_list()
non_cells_df['axis-1'] = df_inference_non_cells['axis-1'].to_list()
non_cells_df['axis-2'] = df_inference_non_cells['axis-2'].to_list()
non_cells_df.to_csv(
    os.path.join(
        save_results_to,
        f'predicted_non_cells_{model_name}_{model_version}.csv'
    )
)
tfi = datetime.now()
print(f"------------ Classification took {tfi - tst} -------------")

