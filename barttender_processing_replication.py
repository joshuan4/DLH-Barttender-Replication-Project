# This file replicates processing for the main barttender experiment, appending bars to images
from barttender.mimic_constants import *
import sys
import os
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from pathlib import Path
from PIL import Image
from skimage.io import imread
from skimage.io import imsave, imshow
from skimage.transform import resize

# Import fails due to relative file pathing discrepancy - functions are manually copied in code below
# from barttender.mimic_add_bar_preprocess import preprocess_mimic_df, preprocess_only_bars, preprocess_and_append_bars, preprocess_no_bars

# every path returned from this file MUST be a pathlib.Path object
# Updated paths for user environment
home_out_dir = Path('physionet.org/outputs/cardiomegaly')
scratch_dir  = Path('physionet.org/files/mimic-cxr-jpg/2.1.0')  # MIMIC-CXR-JPG images
nb_group_dir = Path('physionet.org/files/mimic-cxr-jpg/2.1.0/preprocessed')  # Preprocessed images directory

# Import cleaned master dataframe
feature_folder = Path('physionet.org/Mimic_features/')

# Updating functions with new file paths
def get_master_df(idp=False):
    if idp:
        # (2662, 104)
        return pd.read_pickle('barttender/CardiomegalyBiomarkers/Cardiomegaly_Classification/MIMIC_features_OG/MIMIC_features.pkl')
    return pd.read_pickle('physionet.org/Mimic_features/MIMIC_features_v3.pkl')

def get_cardiomegaly_df(idp=False):
    df = None
    if idp:
        df = pd.read_pickle('barttender/CardiomegalyBiomarkers/Cardiomegaly_Classification/MIMIC_features_OG/MIMIC_features.pkl')
    else:
        df = pd.read_pickle('physionet.org/Mimic_features/MIMIC_features_v3.pkl')
    return df[df['Cardiomegaly'].isin([0, 1])]

# Ensure the directory exists
os.makedirs(nb_group_dir, exist_ok=True)
print(f"Ensured directory exists: {nb_group_dir}")


### Copying functions from barttender.mimic_add_bar_preprocess

def npy_bar(data, colormap, img_w=500, img_h=100, add_label=False, add_colormap=False, verbose=False):
    variables = data.columns[1:]  # Exclude ID
    variables = variables[::-1]  # Bars generated bottom to top

    for index, row in data.iterrows():
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        bar_height = 1.0  # Height of each bar -- keep it to 1.0 to ensure no gap
        if verbose:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig, ax = plt.subplots(figsize=(img_w*px, img_h*px))

        # Create horizontal bar plot
        for i, var in enumerate(variables):
            value = row[var]
            if np.isnan(value):
                color = 'r'  # i.e. 'red'
            else:
                color = colormap(value)
            ax.barh(i, 1, color=color, height=bar_height, edgecolor='none')

        # Set y-ticks and labels
        if add_label:
            ax.set_yticks(np.arange(len(variables)))
            ax.set_yticklabels(variables)
        else:
            ax.set_yticks([])

        # Remove grid and axes
        ax.grid(False)
        ax.set_xticks([])
        ax.set_xlim(0, 1)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add colorbar
        if add_colormap:
            sm = plt.cm.ScalarMappable(
                cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
            cbar.set_label('Normalized feature value')

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        if add_label or add_colormap:
            plt.show()
        else:
            bar_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            bar_img = bar_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            if img_h != 224:
                # slicing removes whitespace from bars; need to redo if bar size is changed
                bar_img = bar_img[3:-2, :, :]  # HWC
            plt.close(fig)
        
            if verbose:
                print(bar_img.shape)
                imshow(bar_img.astype(np.uint8))

            return bar_img
        
def construct_preproc_path(og_path: str, root_dir: Path, preproc_dir=None):
    split = og_path.split("/")
    preproc_filename = f"{split[2]}_{split[3]}_{split[4]}"
    if preproc_dir:
        return root_dir / preproc_dir / preproc_filename
    else:
        return root_dir / preproc_filename

def preprocess_mimic_df(idp=False, order=None, bar_vars=significant_variables, label='Cardiomegaly', verbose=False):
    variables = np.array(bar_vars)
    order, suffix = get_barcode_order_info(order, bar_vars)
    
    df_master = get_master_df(idp=idp)
    if not idp:
        df_master = df_master[df_master[label].isin([0, 1])]
        study_year = np.floor(df_master['StudyDate'] / 10000)
        delta_years = study_year - df_master['anchor_year']
        df_master['age'] = df_master['anchor_age'] + delta_years
    else:
        df_master['age'] = df_master['anchor_age']

    
    # normalize age as in chexpert (0-100 to 0-1)
    df_master['age_val'] = df_master['age'].apply(lambda x: min(x / 100, 1))
    # quantile normalize the rest
    for var in bar_vars[1:]:
        transformer = QuantileTransformer(output_distribution='uniform')
        df_master.loc[:, var] = transformer.fit_transform(df_master[[var]])

    df = df_master[['path'] + bar_vars]

    if verbose:
        print(df.head())
        print(variables[order].tolist())
    
    return df[['path'] + variables[order].tolist()[0]]

def preprocess_and_append_bars(df, image_type, img_data_dir, idp=False, order=None, bar_vars=significant_variables, colormap=plt.colormaps['binary'], verbose=False):
    df_cxr = df.copy()
    out_dir = img_data_dir

    order, suffix = get_barcode_order_info(order)
    preproc_dir = get_preproc_subpath(image_type, suffix)

    print(f'Image Type: {image_type}')
    print(f"Barcode Order: {suffix.replace('_', ', ')}")
    print(f'preproc_dir: {out_dir / preproc_dir}')
    
    bars_h = int(0.2 * 224) # 44
    whitespace_offset = 5 + 1 # num of pixel rows removed from bar output + 1 (row to remove because resizing artifact)

    if not verbose:
        os.makedirs(out_dir/ preproc_dir, exist_ok=True)

    for idx, p in enumerate(tqdm(df_cxr['path'])):
        out_path = construct_preproc_path(p, out_dir, preproc_dir)
        if (not os.path.exists(out_path)) or verbose:
            height = 224 + whitespace_offset - bars_h; width = 224
            if image_type == 'xray':
                image = imread(scratch_dir / p)  # og chexpert imagery is on scratch
                image = resize(image, output_shape=(height, width), preserve_range=True)
                image = np.expand_dims(image, axis=2).repeat(3, axis=2)
            elif image_type == 'noise':
                # Generate random noise
                image = np.random.randint(low=0, high=255, size=(height, width)) # 1-channel, B/W noise
                image = np.expand_dims(image, axis=2).repeat(3, axis=2)  # make it 3-channel
            elif image_type == 'blank':
                image = np.zeros(shape=(height, width, 3))  # all black image

            image = image[:-1,:,:]  # remove last img row (gets rid of resizing artifacts)

            # Generate the barcode and append it to the bottom of the image
            bar_img = npy_bar(df_cxr.iloc[[idx]], colormap, img_h=bars_h, img_w=width)
            
            combined_img = np.concatenate((image, bar_img), axis=0)
            
            if verbose:
                imshow(combined_img.astype(np.uint8))
            else: 
                out_path.parent.mkdir(parents=True, exist_ok=True)
                #out_path = Path('C:\\Users\\Josh\\Desktop\\DLH Barttender Replication Project\\physionet.org\\files\\mimic-cxr-jpg\\2.1.0\\preprocessed\\preproc_224x224_xray_age_chloride_rr_urea_nitrogren_magnesium_glucose_phosphate_hematocrit\\test.jpg')
                imsave(out_path, combined_img.astype(np.uint8))

def preprocess_no_bars(df, out_dir, verbose=False):
    df_cxr = df.copy()
    image_type = 'xray'
    suffix = 'no_bars'
    df_cxr['path_preproc'] = df_cxr['path']

    preproc_dir = get_preproc_subpath(image_type, suffix)

    print(f'Image Type: {image_type}s with no barcodes')
    print(f'preproc_dir: {out_dir / preproc_dir}')
    
    if not verbose:
        os.makedirs(out_dir/ preproc_dir, exist_ok=True)
        

    for idx, p in enumerate(tqdm(df_cxr['path'])):
        out_path = construct_preproc_path(p, out_dir, preproc_dir)
        
        if (not os.path.exists(out_path)) or verbose:
            height = 224; width = 224
            image = imread(scratch_dir / p)  # og chexpert imagery is on scratch
            image = resize(image, output_shape=(height, width), preserve_range=True)
            image = np.expand_dims(image, axis=2).repeat(3, axis=2)

            if verbose:
                imshow(image.astype(np.uint8))
            else:
                imsave(out_path, image.astype(np.uint8))


# Preprocess all with 8 significant features
df_cxr = preprocess_mimic_df(idp=True, verbose=False, bar_vars=significant_variables)

# If you are on windows and getting a "File Not Found" error when saving images, your file path may be too long
print('Beginning Image Barttender Processing (xray + bars)')
# Image Barttender processing (xray + bars) 
preprocess_and_append_bars(
    df_cxr,
    image_type='xray',
    img_data_dir=nb_group_dir,
    idp=True,
    bar_vars=significant_variables,
    verbose=False # IMPORTANT: it won't save the images if you run in verbose mode
)

print('Beginning Blank Barttender Processing (black image + bars)')
# Blank barttender (black image + bars)
preprocess_and_append_bars(
    df_cxr,
    image_type='blank',
    img_data_dir=nb_group_dir,
    idp=True,
    order=None,
    bar_vars=significant_variables,
    verbose=False
)

print('Beginning Image Only Processing (image only - no bars)')
# Image only
preprocess_no_bars(
    df_cxr,
    out_dir=nb_group_dir,
    verbose=False
)
