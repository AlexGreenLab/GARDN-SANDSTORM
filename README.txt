README

##### Sequence and Structure of RNA Molecule Predictive CNNs & Generative Adversarial RNA Design Networks #####

Doi:https://zenodo.org/records/15058435


Code associated with Generative and Predictive Neural Networks for the Design of Functional RNA Molecules

* Randomized training of predictive Models for Toehold Switches, 5' UTRs, CRISPR gRNAs and RBS sites can be found in the SANDSTORM folder. Example GARDN model training and Activation-maximization routines can be found in the GARDN folder. Associated figures from these domains can be created from 'sequence_composition_analysis.ipynb', 'Toehold_Timer.ipynb', 'Array_Comparison.ipynb', and 'Fig 5.ipynb'

* Scripts for RBS generation and GARDN model optimization can be found in the RBS folder

* Scripts for aptaswtich 100-fold cross validation, N-gene tiling, and GARDN generation can be found within the aptamer folder
* All 100 of our trained aptaswitch predictor models as well as the saved optimization experiments for each of the 100 optimization experiments are in the aptaswitch/models folder as well as the aptaswitch/stat_tests folder


The developemnt of this code was conducted on an apple M1, M1 max, and M1 Ultra chip


#####Installation#####

1.  Create a new conda environment from the .yml files
        'conda env create -f environment. yml'
        
2. Agree to the NUPACK license and install NUPACK from https://docs.nupack.org/start/ using pip

3. Optionally for recreation of overlaid 2nd structure plots in 'Fig 5.ipynb', viennarna and forgi must be installed separately


#Installation Notes
The user must agree to the NUPACK license and download the package from the developers website to recreate many of the figures and analysis
NUPACK https://docs.nupack.org/start/
* NUPACK cannot be installed from conda at the time of development

The release of Forgi used in this repo has come with an outdated 'from collections' statement that required manual adjustment to 'from collections.abc' in the threedee.model folder. This dependency has been isolated to the file 'Fig 5.ipynb' which creates the overlaid secondary structure plots of different design paradigms. The calculations of the structures demonstrated in this plot are conducted in the 'sequence_composition_analysis.ipynb' file along with the quanitative comparison of structural agreement; forgi is only necessary to reproduce the qualitative view of the overlaid structures. 
Forgi https://viennarna.github.io/forgi/


#Notes for retraining on a new dataset

1. The primary step will be to train a SANDSTORM model in the new predictive domain. The definition for the base structure arrays can be found in src/GA_util.prototype_ppms_fast. Any one-hot encoding function should do, src/util.one_hot_encode will allow for encoding from a pandas dataframe where one column is the sequences. The training scripts can be adapted to the new domain, or the model definition (src/GA_util.create_SANDSTORM) can be taken.

2. A suitable generator must be trained within this sequence domain. All of our generator and discriminator model definitions live in src/GARDN_util.py. If your target sequences must adhere to a specific secondary structure, we suggest beginning with the toehold generator. If your targets do not need to adhere to a specific structure, the UTR or RBS models are suitable starting points. The adjsuted model definitions can be plugged into the GARDN/GARDN_toehold or GARDN/GARDN_UTR scripts. 

* An important note is that our current GAN training functions utilize tensorflow @function decorators that dramatically speed-up runtime. The downside of this is that adjusting the parameters of the model or restarting training will throw an error when calling decorated functions that can be solved by refreshing the kernel within the jupyter notebook (i.e. the kernel for these notebooks must be refreshed if more than one iteration of training is to be conducted). We suggest that in a prototype or development environment, commenting out these decorators will prevent this error from being thrown and should allow repetitive training runs without the need to restart the kernel.

3. Once a generator and predictor have been identified, the GARDN/ActMax scripts provide a shell that can be rapidly adjusted to optimize new generators with new predictors. The parameters of this optimization include the number of sequences, optimization steps per sequence, and the gradient update multiplier.
