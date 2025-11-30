# This file goes through all data preprocessing steps,
# ultimately creating the cleaned final dataset

# Code is adapted from barttender/CardiomegalyBiomarkers/Cardiomegaly_Classification/DataPipeline.ipynb

# Imports
import numpy as np
import pandas as pd
from barttender.CardiomegalyBiomarkers.Cardiomegaly_Classification.src.data_pipeline_functions import filter_pd_read_chunkwise
from barttender.CardiomegalyBiomarkers.Cardiomegaly_Classification.src.utils.pandas_utils import filter_df_isin
from tqdm import tqdm
from typing import List, Dict
import os
from barttender.CardiomegalyBiomarkers.Cardiomegaly_Classification.src.data_pipeline_functions import (x_ray_dataframe_generator_v2, icu_xray_matcher_v2)
from barttender.CardiomegalyBiomarkers.Cardiomegaly_Classification.src.data_pipeline_functions import dfCleaningNoIDP, SignalTableGeneratorNoIDP

# File Paths
mimic_iv_path = 'physionet.org/files/mimiciv/3.1/'
icu_stays_path = mimic_iv_path + 'icu/icustays.csv.gz'
chart_events_path = mimic_iv_path + 'icu/chartevents.csv.gz'
patients_table_path = mimic_iv_path + 'hosp/patients.csv.gz'
admissions_table_path = mimic_iv_path + 'hosp/admissions.csv.gz'
lab_events_path = mimic_iv_path + 'hosp/labevents.csv.gz'

mimic_cxr_path = 'physionet.org/files/mimic-cxr-jpg/2.1.0/'
cxr_records_path = mimic_cxr_path + 'cxr-record-list.csv.gz'
cxr_metadata_path = mimic_cxr_path + 'mimic-cxr-2.0.0-metadata.csv.gz'
df_split_path = mimic_cxr_path + 'mimic-cxr-2.0.0-split.csv.gz'
negbio_path = mimic_cxr_path + 'mimic-cxr-2.0.0-negbio.csv.gz'
chexpert_path = mimic_cxr_path + 'mimic-cxr-2.0.0-chexpert.csv.gz'

# Features intermediate
feature_folder = 'physionet.org/Mimic_features/'

# MIMIC intermediate
relevant_chart_events_save_path = feature_folder + 'RelevantChartEvents.pkl'
relevant_lab_events_save_path = feature_folder + 'RelevantLabEvents.pkl'
df_icu_xray_path =  feature_folder + 'IcuXrayMatched.pkl'

# Final cleaned features
features_path = feature_folder + 'MIMIC_features_v3.pkl'

# General Parameters
label = 'Cardiomegaly'  # Define label of target disease ('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices')
chunk_size = 10**7      # When extracting relevant lab and chart events we iterate through the original files in chunks of size 'chunk_size'.

# MIMIC-CXR (imaging) Parameters
view = 'PA'     # Choose the X-ray view position you're interested in, AP or PA

# MIMIC-IV (non-imaging) Parameters
MIMIC_IV_version = 3            # Version of MIMIC-IV downloaded
days_before_icu = 365           # The number of days before ICU admission that we look for x-rays in
xray_gap_after_icu = 0          # You can choose to include a 'gap' after ICU discharge in which you don't look for any X-rays
xray_max_time_after_icu = 90    # If you don't want a gap, xray_max_time_after_icu is just the number of days after ICU discharge that we look for x-rays in. We look for x-rays which are between Gap and Gap + xray_max_time_after_icu days after out-time
average_by = 'Stay'             # 'Hourly' to average readings every hour and have one hour per row; 'Stay', to average chart and lab values across a stay
filter_col = 'itemid'   	    # Define features to use for time-series prep

# Labels of desiered non-imaging features

# Lables
chart_labels_mean = {
    220045: 'HR_mean',
    220277: 'SpO2_mean',
    223761: 'Temp(F)_mean',
    220210: 'RR_mean',
    220052: 'ABPm_mean',
    220051: 'ABPd_mean',
    220050: 'ABPs_mean',
    220180: 'NBPd_mean',
    220181: 'NBPm_mean',
    220179: 'NBPs_mean',
    223835: 'FiO2_mean',
    220274: 'PH_mean',
    220235: 'PCO2_mean',
    220227: 'SaO2_mean',
    227457: 'PlateletCount_mean',
    227456: 'Albumin_mean',
    220603: 'Cholesterol_mean',
    220645: 'Sodium_mean',
    220224: 'PO2_mean',
}

chart_labels_max = {
    220045: 'HR_max',
    220210: 'RR_max',
    220052: 'ABPm_max',
    220051: 'ABPd_max',
    220050: 'ABPs_max',
    220180: 'NBPd_max',
    220181: 'NBPm_max',
    220179: 'NBPs_max',
    223835: 'FiO2_max',
    220235: 'PCO2_max',
    220645: 'Sodium_max',
}

chart_labels_min = {
    220045: 'HR_min',
    220277: 'SpO2_min',
    220210: 'RR_min',
    220052: 'ABPm_min',
    220051: 'ABPd_min',
    220050: 'ABPs_min',
    220180: 'NBPd_min',
    220181: 'NBPm_min',
    220179: 'NBPs_min',
    220235: 'PCO2_min',
    220645: 'Sodium_min',
}

lab_labels_mean = {
    50826: 'Tidal_Volume_mean',
    51006: 'Urea_Nitrogren_mean',
    50863: 'Alkaline_Phosphatase_mean',
    50893: 'Calcium_Total_mean',
    50902: 'Chloride_mean',
    50931: 'Glucose_mean',
    50813: 'Lactate_mean',
    50960: 'Magnesium_mean',
    50970: 'Phosphate_mean',
    50971: 'Potassium_mean',
    50885: 'Bilirubin',
    51003: 'Troponin-T_mean',
    51221: 'Hematocrit_mean',
    50811: 'Hemoglobin_mean',
    50861: 'ALT_mean',
    50912: 'Creatinine_mean',
    51275: 'PTT_mean',
    51516: 'WBC_mean',
    51214: 'Fibrinogen',
}

lab_labels_max = {
    50971: 'Potassium_max',
    51003: 'Troponin-T_max',
    50811: 'Hemoglobin_max',
    51516: 'WBC_max',
}

lab_labels_min = {
    50971: 'Potassium_min',
    50811: 'Hemoglobin_min',
    51516: 'WBC_min',
}

# Aggregation of all laboratory items into LabItems
LabItems = dict(lab_labels_mean)
LabItems.update(lab_labels_max)
LabItems.update(lab_labels_min)

# Aggregation of the vital signs / chart items into ChartItems
ChartItems = dict(chart_labels_mean)
ChartItems.update(chart_labels_max)
ChartItems.update(chart_labels_min)

# Adapted filter_pd_read_chunkwise function to show progress bar
def filter_pd_read_chunkwise_v2(
    file_path: str,
    filter_col: str,
    filter_list: List[str],
    chunksize: float,
    dtype: dict = None,
) -> pd.DataFrame:

    chunk_iter = pd.read_csv(file_path, chunksize=chunksize, dtype=dtype)

    filtered_chunks = []

    # Wrap the iterator in tqdm for a progress bar
    for chunk in tqdm(chunk_iter, desc="Reading CSV in chunks"):
        filtered_chunks.append(filter_df_isin(chunk, filter_col, filter_list))

    return pd.concat(filtered_chunks)

os.makedirs(feature_folder, exist_ok = True)
if not os.path.exists(relevant_chart_events_save_path):
    # MIMIC-IV: Extract necessary features chunkwise
    # This will process 44 chunks
    df_icu_timeseries = filter_pd_read_chunkwise_v2(
        file_path=chart_events_path,
        filter_col=filter_col,
        filter_list=ChartItems.keys(),
        chunksize=chunk_size,
    )

    df_icu_timeseries.to_pickle(relevant_chart_events_save_path)
else:
    print(f'{relevant_chart_events_save_path} already exists.')

if not os.path.exists(relevant_lab_events_save_path):
    # This will process 16 chunks
    df_icu_lab = filter_pd_read_chunkwise_v2(
        file_path=lab_events_path,
        filter_col=filter_col,
        filter_list=LabItems.keys(),
        chunksize=chunk_size,
    )

    df_icu_lab.to_pickle(relevant_lab_events_save_path)
else:
    print(f'{relevant_lab_events_save_path} already exists.')

# Matching MIMIC-IV and MIMIC-CXR Data
# This cell takes ~10 minutes to run
df_split = pd.read_csv(df_split_path)
df_metadata = pd.read_csv(cxr_metadata_path, header=0, sep=',')
df_cxr_records = pd.read_csv(cxr_records_path, header=0, sep=',')
df_nb = pd.read_csv(negbio_path)
df_cx = pd.read_csv(chexpert_path)

# MIMIC-CXR: Create X-Ray dataframes (the table will only contain the paths to the actual pictures)
labels = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices']
if not os.path.exists(df_icu_xray_path):
    df_xray_v2 = x_ray_dataframe_generator_v2(
        # TEST: just use cardiomegaly label like OG does
        labels=labels,
        df_cxr_records=df_cxr_records,
        df_nb=df_nb,
        df_cx=df_cx,
        df_cxr_meta_data=df_metadata,
        df_split=df_split)

    # Link X-Ray to ICU stays if in certain time window defined by days_before_icu, xray_gap_after_icu, and xray_max_time_after_icu
    df_icu_stays = pd.read_csv(icu_stays_path)

    df_icu_xray_v2 = icu_xray_matcher_v2(
        labels=labels,
        days_before_icu=days_before_icu,
        xray_gap_after_icu=xray_gap_after_icu,
        xray_max_time_after_icu=xray_max_time_after_icu,
        df_xray=df_xray_v2,
        df_icu_stays=df_icu_stays)
    df_icu_xray_v2.to_pickle(df_icu_xray_path)
else:
    print(f'{df_icu_xray_path} already exists.')

df_patients = pd.read_csv(patients_table_path)
df_admissions = pd.read_csv(admissions_table_path)
df_icu_xray = pd.read_pickle(df_icu_xray_path)
df_icu_lab = pd.read_pickle(relevant_lab_events_save_path)
df_icu_timeseries = pd.read_pickle(relevant_chart_events_save_path)

# edit name of df_admissions column if data taken from versions after MIMIC-IV v1.0 as 'ethnicity' column was renamed 'race' in following version (v2.0)
if MIMIC_IV_version != 1:
    df_admissions.rename(columns={'race':'ethnicity'}, inplace=True)

if not os.path.exists(features_path):
    # collate all features (MIMIC-IV feautres, MIMIC-CXR file paths, biomarker values) into one master table
    df_master = SignalTableGeneratorNoIDP(df_icu_xray, 
                                    df_icu_timeseries=df_icu_timeseries, 
                                    df_icu_lab=df_icu_lab, 
                                    df_patients=df_patients, 
                                    df_admissions=df_admissions, 
                                    chart_labels_mean=chart_labels_mean, 
                                    chart_labels_max=chart_labels_max, 
                                    chart_labels_min=chart_labels_min, 
                                    lab_labels_mean=lab_labels_mean, 
                                    lab_labels_max=lab_labels_max, 
                                    lab_labels_min=lab_labels_min, 
                                    average_by=average_by)

    df_master_cleaned = dfCleaningNoIDP(df_master)
    df_master_cleaned.to_pickle(features_path)
else:
    print(f'{features_path} already exists.')