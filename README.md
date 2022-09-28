# SoSe2022_Reinforcement_Learning

Repository for study group at THM Gießen/Friedberg in the course "Reinforcement Learning" in the summer semester of 2022,
by Professor Nicolas Stein 

# Setting up the venv with conda
To install all requirements,open your terminal in this repository,  create a new conda with 

    "conda env create -f environment.yml"

After that, activate the environment as the default interpreter for this main and everything should be able to run,
at least in terms of dependencies and such.

#Orientation with table of contents:

* This repository contains the results of our assignment, Nr.3: Dueling  DQN.
* The whole list of assignments can be consulted in the documentation directory, as well the explanatory paper for Dueling DQN
* The log directory contains the raw logging data from our main training episode on which our results are based. Those can alre
* Those were then converted into jsons and plotted, the code for that can be found in the visualize method of main.py
* The other two methods of main are a bundling of tests for both models which are saved as .h5 data in the 'models'directory,
as well as the train method which trains both models again with the parameters configured in Assignment3.py.

#Programming

The dueling DQN was implemented as a copy of the vanilla-agent which was given to us by defining three instead of 1 layer
for calculating the q-value: 

One Tensor for the value of the state, one for the q-value of all actions and a third calculating the difference.
This leads the agent to consider the quality of its' state and not just the qualities of actions it can take, 
leading to a higher learning rate as bad states are no longer as heavily trained as before.
The architecture of the neural net consists of two layers with 64 and 128 neuron respectively, both with simple relu 
functions and optimized by the adam algorithm with the metric of mean square error. 
The other hyperparameter concerning the reinforcement agent training itself were manually explored. The configuration 
with which the main data presented was gained is still saved as strings in the configuration dictionary of the RLRunner 
in Assignment3.py
The other parts of training were not altered, as we did not complete the bonus task given in our assignment. 

