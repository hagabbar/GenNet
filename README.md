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

run lalinference_pipe and submit run to condor in order to get lalinference posterior estimates on injection waveform. This will require that you have both an injection.xml and an injection.ini file. 

This assumes you have access to the LIGO caltech clusters.
```
source /home/jveitch/lalsuites/master/etc/lalsuiterc

lalinference_pipe -r injection_run_mass-time-varry_gw150914_srate-2048/ -I injection_gw150914.xml injection_gw150914.ini

condor_submit_dag /home/hunter.gabbard/parameter_estimation/john_bayesian_tutorial/injection_run_mass-time-varry_gw150914_srate-2048/multidag.dag
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```
## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Hunter Gabbard** - *lead developer* - [github account](https://github.com/hagabbar)
* **Chris Messenger**

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
