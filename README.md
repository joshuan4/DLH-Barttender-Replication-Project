# DLH-Barttender-Replication-Project
**CS:598 Deep Learning for Healthcare - Barttender Replication Project**

This repositroy stores all of our code for our project replicating and extending the paper
*Barttender: An approachable & interpretable way to compare medical imaging and non-imaging data*.

This repository builds off code and utilizes scripts / functions from two repositories:

1. [https://github.com/singlaayush/barttender](https://github.com/singlaayush/barttender)

2. [https://github.com/singlaayush/CardiomegalyBiomarkers/tree/a134e8a9af335076be811f3cedb82b63adff72b2](https://github.com/singlaayush/CardiomegalyBiomarkers)

A script has been included in our repository for cloning the above two repositories with the correct relative file pathing. Please run clone_repos.sh as a bash script inside the DLH-Barttender-Replication-Project folder.


**Data Download Instructions**

Our project uses data from the MIMIC-IV and MIMIC-CXR-JPG databases. The links to these data sources, as well as instructions for downloading the necessary data, can be found here:
1. https://physionet.org/content/mimiciv/3.1/
2. https://physionet.org/content/mimic-cxr-jpg/2.1.0/

**Assumed File Structure**

Once the above repositories are cloned and the data is downloaded, our code assumes the following file structure to successfully run all of our scripts:
```
project/
├── barttender/
│   ├── CardiomegalyBiomarkers/
│   │   └── <CardiomegalyBiomarkers files>
│   └── <barttender files>
├── physionet.org/files/
│   │   └── mimic-cxr-jpg/2.1.0/
│   │       └── files/
│   │           └── <cxr image files>         
│   │       └── cxr-record-list.csv.gz
│   │       └── mimic-cxr-2.0.0-chexpert.csv.gz
│   │       └── mimic-cxr-2.0.0-metadata.csv.gz
│   │       └── mimic-cxr-2.0.0-negbio.csv.gz
│   │       └── mimic-cxr-2.0.0-split.csv.gz
│   │   └── mimiciv/3.1/
│   │       └── hosp/
│   │           └── admissions.csv.gz
│   │           └── labevents.csv.gz
│   │           └── patients.csv.gz
│   │       └── icu/
│   │           └── chartevents.csv.gz
│   │           └── icustays.csv.gz
└── <project files>
```

**Script Order**

In order to run all files from our project end-to-end, please run scripts in the following order:

1. clone_repos.sh
2. preprocessing.py
3. baseline_replication.py
4. barttender_processing_replication.py
5. barttender_experiment_replication.py

**Replication Notebook**

The full workflow was execute in a Google Colab notebook environment - see the execution of various preprocessing, training and evaluation steps in BarttenderReplication.ipynb

