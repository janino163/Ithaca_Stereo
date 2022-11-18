1. Set up conda env 
```
conda create -n stereo python=3.8
```

```
conda activate stereo
```
```
conda install -c anaconda ipykernel
```

```
python -m ipykernel install --user --name=stereo
```
Install either the gpu version of pytorch

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```
or cpu version

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

```
pip install -r requirements.txt
```

2. download data
https://cornell.box.com/s/6g9y5zo1wnxa791sdi5s0k9pcw4b1ock
unzip into drives/

3. Go through tutorial for using the data
Run data_tutorial.ipynb  
