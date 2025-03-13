### Requirements:

**general**:<br>
- Linux (tested on Ubuntu 22.04+)<br>
- git<br>
- curl<br>
- conda (from Anaconda, Miniconda or Miniforge)<br>

**for the heavy lifting pipeline**:<br>
- NVidia GPU (optional but highly recommended)<br>
- Enough disk space (>1.5Tb per file) to download the raw data

*Note:*
*Implementation of floating-point operations slightly differs between the versions of CUDA, other software and hardware. If you try to reproduce our results, please keep in mind that even using the exact same versions of software, the outputs of the deep-learning based methods can be slightly different form ours.*


## To run the notebooks that reproduce figures and tables:

**1) clone the repository**

```
git clone https://github.com/CBI-PITT/cfos_morphine_pipeline.git
cd cfos_morphine_pipeline
```

**2) Download python environment to run the notebooks**

```
curl "https://brain-api.cbi.pitt.edu/browser/world/opoid_paper/morphine_cfos_paper_jupyter.tar.gz" --output morphine_cfos_paper_jupyter.tar.gz
```

**3) Extract the archive**

```
mkdir morphine_cfos_paper_jupyter
tar -xf morphine_cfos_paper_jupyter.tar.gz -C morphine_cfos_paper_jupyter
```

**4) Use conda-unpack to fix the environment**

```
morphine_cfos_paper_jupyter/bin/conda-unpack
```

**5) Activate the environment**

```
source morphine_cfos_paper_jupyter/bin/activate
```

**6) Download the data (high-level data .csv files)**

```
curl "https://brain-api.cbi.pitt.edu/browser/world/opoid_paper/csv_files.tar.gz" --output csv_files.tar.gz
```

the archive should be saved in the same directory as notebooks

**7) Unpack the data**

```
tar -xvf csv_files.tar.gz
```

This will unpack the files into a csv_files folder next to the code.

**8) Start the notebook server**

```
jupyter notebook
```

This should open the current directory view in your web browser

**9) Open the notebook**

(cfos_paper_figures.ipynb or cfos_paper_tables_and_numbers.ipynb)

**10) Run all cells**

In the main menu select Cell > Run All

Sometimes the execution of notebooks gets stuck and they need to be run twice. If the notebook seems stuck, select Kernel > Restart and then select Cell > Run All again.


## To run the heavy lifting pipeline:

The heavy lifting pipeline generates high-level data from raw terabyte-sized multiscale Imaris files.

In the most simple scenario, it consists of:
- extraction of low-resolution data (.tif) from the multiscale Imaris (.ims) format
- registration to 10-um Allen adult mouse brain atlas
- cell detection (conventional + deep learning based)
- removal of artifacts (deep learning based)
- mapping of the remaining true cells to brain regions
- export of the mapped cell location data (.csv)

*Warning: Downloading our raw Imaris files would take a long time and would use up terabytes of space. They will be available for web-based visualization in Neuroglancer at the Brain Image Library upon publication of the manuscript.
Reproduction of analysis of all data might take days.*

We provide an example using one representative Imaris file and a simplified version of the workflow, which can be run on a single machine. A high-performance, user friendly and more generic version of the workflow will be a subject of a separate publication.

1) Clone the repository

```
git clone https://github.com/CBI-PITT/cfos_morphine_pipeline.git
cd cfos_morphine_pipeline
```
2) Download and unzip the Imaris file and models

This step can take a long time as the Imaris file alone is > 1 Terabyte

```
curl "https://brain-api.cbi.pitt.edu/browser/world/opoid_paper/f5_brain.tar.gz" --output f5_brain.tar.gz
```

```
tar -xf f5_brain.tar.gz
```

3) Download python environments to run the code

Three separate environments are used because of their complexity and conflicting dependencies:
- main environment - morphine_cfos_paper_main

```
curl "https://brain-api.cbi.pitt.edu/browser/world/opoid_paper/morphine_cfos_paper.tar.gz" --output morphine_cfos_paper_main.tar.gz
```

- tensorflow environment - morphine_cfos_paper_deepblink

```
curl "https://brain-api.cbi.pitt.edu/browser/world/opoid_paper/morphine_cfos_paper_deepblink_tf_2.8.tar.gz" --output morphine_cfos_paper_deepblink.tar.gz
```

- pytorch environment - morphine_cfos_paper_fastai

```
curl "https://brain-api.cbi.pitt.edu/browser/world/opoid_paper/morphine_cfos_paper_fastai.tar.gz" --output morphine_cfos_paper_fastai.tar.gz
```

This should save 3 .tar.gz archives to the same directory where the code is.

4) Extract the archives

```
mkdir morphine_cfos_paper_main
tar -xf morphine_cfos_paper_main.tar.gz -C morphine_cfos_paper_main
```
```
mkdir morphine_cfos_paper_deepblink
tar -xf morphine_cfos_paper_deepblink.tar.gz -C morphine_cfos_paper_deepblink
```
```
mkdir morphine_cfos_paper_fastai
tar -xf morphine_cfos_paper_fastai.tar.gz -C morphine_cfos_paper_fastai
```

5) Use conda-unpack to fix the environments

```
morphine_cfos_paper_main/bin/conda-unpack
morphine_cfos_paper_deepblink/bin/conda-unpack
morphine_cfos_paper_fastai/bin/conda-unpack
```

6) Activate the main environment 

```
source morphine_cfos_paper_main/bin/activate
```

7) Run the pipeline

```
python pipeline_from_ims_to_csv.py f5_brain/F5.ims
```










