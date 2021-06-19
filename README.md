# Robustness of Sparse Neural Networks
Author: Neil Kichler

The code for SET was modified from an alternate Numba based version of the official [SET implementation](https://github.com/SelimaC/Tutorial-SCADS-Summer-School-2020-Scalable-Deep-Learning)

##### Dependencies
To run the program, please make sure that the following requirements are met
(if not - install with ```pip install <package>``` or anaconda etc.):

python 3.7.6, numpy 1.17.2, scipy 1.4.1, numba 0.48.0, scikit-learn 0.24.2,
matplotlib 3.4.1, pandas 1.2.4, psutil 5.8.0

######  Abstract of associated Paper:
Deep Neural Networks have seen great success yet require increasingly higher dimensional data to be applied successfully.
To reduce the ever-increasing computational, energy and memory requirements, the concept of sparsity has emerged as a leading approach.
Sparse-to-sparse training methods allow training and inference on more resource-limited devices.
It has been hinted in previous work like SET [1], that such methods could be applied to feature selection since they may implicitly encode the input neuron strengths during training. However, a proper investigation of this potential idea has not taken place in the domain of supervised feature selection.
This paper develops a method for supervised feature selection using Sparse Evolutionary Training applied to Multi-Layer Perceptrons (SET-MLP).
The focus is on investigating the robustness of this feature selection mechanism to changes in the topology of SET-MLPs.
We develop and perform an experimentally driven analysis on
some prominent datasets to evaluate the generalizability, initialization-dependence and similarity of the underlying networks of the feature selection process. We find for the selected datasets that SET-MLP produces similar feature selections for different underlying network topologies and can recover from bad initialization.
This work provides a basis for understanding whether supervised feature selection using sparse training methods are robust to topological changes.
The problem addressed can have further implications in understanding sparse training, given that it visualizes some aspects of the random exploratory nature of these methods.
Furthermore, it discusses the potential viability of sparse-to-sparse training methods for supervised feature selection.


###### References

For an easy understanding of SET, please read the following article:

1. @article{Mocanu2018SET,
  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
  journal =       {Nature Communications},
  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
  year =          {2018},
  doi =           {10.1038/s41467-018-04316-3},
  url =           {https://www.nature.com/articles/s41467-018-04316-3 }}
