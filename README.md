# BayesCNNSurrogate
This repository hosts codes and data for a Bayesian convolutional neural network surrogate model that predicts mechanical properties from microstructure data of ceramic aerogels, with quantified uncertainties.

## Manuscript
**Title:** *Stochastic Deep Learning Surrogate Models for Uncertainty Propagation in Microstructure-Properties of Ceramic Aerogels*

**Authors:**  
- Md Azharul Islam 
- Dwyer Deighan   
- Shayan Bhattacharjee  
- Daniel Tantalo  
- Pratyush Kumar Singh  
- David Salac 
- Danial Faghihi


## Summary
Deep learning surrogate models have become pivotal in enabling model-driven materials discovery to achieve exceptional properties. However, ensuring the accuracy and reliability of predictions from these models, trained on limited and sparse material datasets remains a significant challenge.
This study introduces an integrated deep learning framework for predicting the synthesis, microstructure, and mechanical properties of ceramic aerogels, leveraging physics-based models such as Lattice Boltzmann simulations for microstructure formation and stochastic finite element methods for mechanical property calculations.
To address the computational demands of repeated physics-based simulations required for experimental calibration and material design, a linked surrogate model is developed, leveraging Convolutional Neural Networks (CNNs) for stochastic microstructure generation and microstructure-to-mechanical property mapping. To overcome challenges associated with limited training datasets from expensive physical modeling, CNN training is formulated within a Bayesian inference framework, enabling robust uncertainty quantification in predictions.
Numerical results highlight the strengths and limitations of the linked surrogate framework, demonstrating its effectiveness in predicting properties of aerogels with pore sizes and morphologies similar to the training data (in-distribution) and its ability to interpolate to new microstructural features between training data (out-of-distribution).

## Credits
This software uses the following open source packages: `LBfoam` [(github)] (https://github.com/mehdiataei/LBfoam) is an open-source CFD solver based on the lattice Boltzmann method for foaming simulations. The solver is an extended version of the Palabos library. 

`PyTorch-BayesianCNN` library by Kumar Shridhar [(github)](https://github.com/kumar-shridhar/PyTorch-BayesianCNN/tree/master) was used to implement bayeisan CNN. Bayesian convolutional neural networks is coupled with variational inference, a variant of convolutional neural networks (CNNs), in which the intractable posterior probability distributions over weights are inferred by Bayes by Backprop.

## Acknowledgement
This work is supported by the Research Foundation of The State University of New York (SUNY) under grant number 1191358.
