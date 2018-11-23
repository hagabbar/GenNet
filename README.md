# GAN for GW Parameter Estimation

This is a repository which is used for generating posterior estimates on a binary black hole (BBH) waveform burried in noise. Other waveform models will be added at a later date. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Keras, tensorflow, scipy, matplotlib, numpy, PIL, time, pandas, pickle, gwpy, h5py, sympy, cuda, cudnn

```
pip install --user Keras tensorflow scipy matplotlib numpy PIL time pandas pickle gwpy h5py sympy
```

Don't forget to install CUDA and CUDNN on your machine.

### Installing

Just simply git clone this repository like so.

```
git clone https://github.com/hagabbar/GenNet.git
```

In order to run the code, you will need to produce your own lalinference waveform and posterior estimates on the event of interest. See Eric Thrane's useful guide on how to do this [guide](http://users.monash.edu.au/~erict/Resources/lal/). I will add my own guide on how to do this at a later date.

## Running the scripts

Running of the whole pipeline is done in a few simple steps

### 1.) Lalinference

Run lalinference_pipe and submit run to condor in order to get lalinference posterior estimates on injection waveform. This will require that you have both an injection.xml and an injection.ini file. 

This assumes you have access to the LIGO caltech clusters.
```
source /home/jveitch/lalsuites/master/etc/lalsuiterc

lalinference_pipe -r injection_run_mass-time-varry_gw150914_srate-2048/ -I injection_gw150914.xml injection_gw150914.ini

condor_submit_dag /home/hunter.gabbard/parameter_estimation/john_bayesian_tutorial/injection_run_mass-time-varry_gw150914_srate-2048/multidag.dag
```

Make sure to point all the following scripts the lalinference_nest/engine directory of your lalinference run.

### 2.) Convert lalinference parameters

We will be doing parameter estimation on the chirp mass and the inverse mass ratio. Unfortunately, lalinference does not give these parameters explicitly, rather they have to be derived from the output of the posterior. We do this by using the following script.

```
python get_lalinf_pars.py
```  

You will need to choose what parameters you would like to produce in the get_lalinf_pars file.

### 3.) Making templates for training of CNN and GAN

Now that we have the parameters for our lalinference posterior which we will be comparing our GAN/CNN results to, we now need to make training waveforms/parameters. This can be done by running the following script.

```
python gw_template_maker.py
```

This will produce GW template time series files (denoted with ts in the filename) and their corresponding parameters (files denoted with pars).

Some editing in the global parameters portion of the file will need to be done in order to specifiy output directory, lalinference parameters, and lalinference waveforms.

### 4.) CNN sanity check waveforms

Given lalinference parameters, we will now produce simulated waveforms from those parameters. This is done so that we can test how well the CNN can perform if given ideal estimates on a noise-free waveform.
```
python lalinf_post_waveform_maker.py
```

This will output a file with cnn_sanity check in the name. It will contain an array of time series with their corresponding lalinference parameters.

### 5.) Train CNN point estimator and GAN waveform estimator

First train the CNN model by setting the variable do_only_old_pe_model in bbhMahoGANy.py to False. Models are saved every 5000 epochs. Once the CNN has been trained to your satisfaction, switch do_only_old_pe_model to True. Now run bbhMahoGANy.py again in order to train GAN to make noise-free waveform estimates on the event of interest.

```
python bbhMahoGANy.py
```

Good luck!

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Hunter Gabbard** - *lead developer* - [github account](https://github.com/hagabbar)
* **Chris Messenger**
* **Micheal Williams**
* **Jordan McGinn**

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
