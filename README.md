# Traffic Transformer
A pytorch implementation of Traffic Transformer for traffic forecasting.
*Paper(https://ieeexplore.ieee.org/document/9520129)
<img width="800" height="400" src=fig2.png>

## Acknowledgement
Thank to the authors of [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet) and [DCRNN](https://github.com/liyaguang/DCRNN).
My work stands on their basic code and data.

## Requirements
- python 3
- see `requirements.txt`

## Data

### Step1: Download METR-LA data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN). 

### Step2: Process raw data 

```
# Create data directories
mkdir -p data/{METR-LA}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

```
## Train

```
python train.py
```

## Test
```
python test.py
```


