# **GNTau Workflow**

## General Steps:
1. entering the cluster (example: ssh zivka@192.114.102.138)
2. source /usr/local/anaconda/3.8u/etc/profile.d/conda.sh 


## Preparing the samples
root --> h5 - In this step we would like to converte the filse from root to h5 format.


**1.** converter: 
```
python converter.py -c config.yaml
```
*the output file format will be output_JZ_?_.h5 - notice the second point on things you might need to modify*

converter.py, config.ymal - under the folder preparing_samples

Things you might need to modify:
1. the path of the different jetjet samples folders
2. the output file name (for each JZ slice)
3. n_jets - in the config file

*tautau samples are ready (R22 - after all stages), and can be found here:*  
/storage/agrp/zivka/umami_Tau/UPP/umami-preprocessing/upp/ntuples/tau

**2.** merging: 
```
python merge_ds.py -c merging.yaml
```
merge_ds.py, merging.yaml - under the folder preparing_samples


Things you might need to modify:
1. output file name, in merging.yaml at the top of the page
2. total_size: 40e6
3. fraction of each JZ slice - notice that you might need to change the path of the JZ slices in this file


**3.** labeling
```
python label.py
```
label.py - under the folder preparing_samples

Things you might need to modify:
1. line 173: datasets, you can choose between jet/tau samples.
   * (if you created new tau samples you will need to do this stage twice and change the dataset)
2. line 182: for tau samples, you can also modify total_jets number.


**Congrats!** now you have ready to go ntuples 

## UPP - preprocessing 

You can use the UPP tutorial: https://github.com/umami-hep/umami-preprocessing  
**Please** add these 2 files to the folder:

tau-variables.yaml, tau.yaml - under the folder umami

Things that are not written and important to mention:  
1. notice that you activate UPP
2. you should have a ntuples folder that should be seperated as follows:
* ntuples
    * qcd - contains jetjet samples after labeling
    * tau - contains tautau samples after labeling 
3. to create all train, val, and test files together you should run the following:
```
preprocess --config configs/tau.yaml --split all
```
**Congrats!** now you are ready to train

## Training 

training is done by salt, the tutorial can be found in this folder under tutorial-salt

*Do only once - Replace your predictionwriter.py with the file predictionwriter.py - under the folder salt*

1. You might need to run all or some of the following, you will see it the the tutorial :
* screen / submit a job
* conda activate salt 
* python -m pip install -e .
2. Go to the run directory and launch:
```
salt fit -c GN2TauA.yaml --force
```
*A* stands for all, you can modify the config file as you wish  

GN2TauA.yaml, GNTau.ymal, GNTauJ.ymal, GNTauC.ymal - under the folder salt

Things you might need to modify:
1. train, val, test, norm_dict, class_dict paths - to the output files of UPP
2. write_tracks: True - this determines if we will have the output of the aux task or not (the tracks GNTau prediction)
3. variables that we want to train on:
    * Comment the variables
    * Modify the numbers as necessary, for example if you want to change the track variables, you should change line 141, input_size: 51, to the total number of jets and the new track variables.
4. number of epochs, according to Dmitrii for the latest training we overfitted, 20 epochs should have been enough.


**Congrats!** now you have a trained model

## Evaluation

The evaluation part is also done by salt, you should run the following:  
```
salt test --config logs/GN2Tau_20230726-T175600/config.yaml --data.test_file /storage/agrp/zivka/umami_Tau/UPP/umami-preprocessing/upp/output/pp_output_test.h5
```

config file: can be found in the folder of the training output under logs  
data.test_file file:  can be the path path of the test_file that was used during training, (UPP output) 

## Output Plots

**1.** ROC and the eff/rej plots
You should copy the notebook to your directory and play with the parameters:
clean_plot_tau_ziv.ipynb - under the folder notebooks

this is the old one, it contains more things that are not really needed to create the ROC plots, but if necessary you can also take a look into this one: 
plot_tau_ziv.ipynb - under the folder notebooks

Things you might need to modify:
1. test_file should be the exact file you evaluated on
2. modify the networks at the top of the notebook
3. at the last part of the eff/rej plots, you can modify the network, now it checks only RNN 

**2.** CM
1. RNN - confusion_matrix.ipynb - under the folder notebooks
2. GNTau - confusion_matrix_NOT_RNN.ipynb - under the folder notebooks
    * if we have the output of the aux task

**3.** JZ slices 
exploration.ipynb - you should use only the last part, can be found under the folder notebooks

**4.** Variables Plots
If you wish to plot variable distributions you can take a look in this notebook and modify as necessary:
compare_ntuples.ipynb - under the folder notebooks
