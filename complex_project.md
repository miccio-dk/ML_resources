# Complex project
The objectives of this project are:

1. Find out whether input representation of sound which doesn't discard phase information enables the DL model to perform        better than the usual magnitude spectrogram.

2. Find out how well an audio model that uses complex numbers performs compared to the standard solution which uses only real    numbers.

Results: https://goo.gl/PjM1MQ



## Framework on the AAU server based in CPH
Working directory:   /home/tomas/complex (You might need to use sudo to execute commands)

Data directory:      /home/tomas/datasets/sc012



## Requirements
Python 3.6

keras-gpu==2.0.8

tensorflow-gpu==1.4

pandas

librosa

scikit-learn


### Installation
Install Miniconda (environment manager) on Linux https://goo.gl/o15sAs

Create conda environment
```
conda create -n <name> python=3.6
```
   
Activate your environment
```
source activate <name>
```

Install packages
```
conda install -c c3i_test2 keras-gpu
conda install -c cjj3779 tensorflow-gpu
conda install -c conda-forge librosa pandas scikit-learn
```


## Usage
1. a) Implement pre-processing function in data_generator.py that will turn a complex spectrogram into a desired input               representation (similarly as the function compute_log_mel_s does).

   b) Use it as another pre-processing option in __data_generation (follow the example given after statement - if                   self.pre_processing == 'log_mel':)
   

2. a) Create your shell script based on train_log_mel_baseline.sh. 

   b) Change pre-processing, model_path and log_dir parameters.
   
   c) If necessary, change n_rows, n_cols and n_channels parameters, which should correspond to the size of your input                 representation.
   
   d) If necessary, implement a new model architecture in train.py and change the model parameter in the shell script               correspondingly. 
   
   e) Don't change the other training hyper-parameters, so we compare only input representations and               architectures.
   
Brief description of all the parameters can be found in the end of the train.py script.

   
3. a) Execute your shell script (e.g. ./train_log_mel_baseline.sh), which will train, save and test your model. 

   b) Write down the testing accuracy and loss of the model as well as the total number of parameters into the following sheet: https://goo.gl/PjM1MQ
   
   c) Describe the input representation and possibly the architecture.
   

Email any questions to tgajar17@student.aau.dk



## Dataset
Subset of the Speech Commands dataset (https://goo.gl/tf3diA), which contains three spoken digits: 0, 1 and 2 (around 2200 recordings of each).

+ 1 second long

+ 16kHz



## Basic pre-processing
Pad/truncate to 16k samples corresponding to 1 second of audio.

Short-time Fourier transform with 1024 samples long Hann windows and hop-size of 512 samples.

Resulting shape: (513, 32)

Data type: complex64

Code in: /home/tomas/datasets/sc012/compute_spectrograms.ipynb



## Baseline
Input representation: standard log Mel power spectrogram with 128 Mel bins -> (128, 32) -> (128, 32, 1)

Architecture: Five consecutive 2D convolutional layers with 3*3 kernels/filters, each followed by batch normalization, ReLU acivation function and 2D MaxPooling.

