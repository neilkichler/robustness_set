# Robustness of Sparse Neural Networks
Author: Neil Kichler

The code for SET was modified from an alternate Numba based version of the official [SET implementation](https://github.com/SelimaC/Tutorial-SCADS-Summer-School-2020-Scalable-Deep-Learning)

##### Dependencies
To run the program, please make sure that the following requirements are met
(if not - install with ```pip install <package>``` or anaconda etc.):

python 3.7.6, numpy 1.17.2, scipy 1.4.1, numba 0.48.0, scikit-learn 0.24.2,
matplotlib 3.4.1, pandas 1.2.4, psutil 5.8.0

######  Sparse Evolutionary Artificial Neural Networks
* Proof of concept implementations of various sparse artificial neural network models with adaptive sparse connectivity trained with the Sparse Evolutionary Training (SET) procedure.  
* The [SET implementations](https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks)
 are distributed in the hope that they may be useful, but without any warranties; Their use is entirely at the user's own risk.


###### References

For an easy understanding of these implementations please read the following articles. Also, if you use parts of this code in your work, please cite the corresponding ones:

1. @article{Mocanu2018SET,
  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
  journal =       {Nature Communications},
  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
  year =          {2018},
  doi =           {10.1038/s41467-018-04316-3},
  url =           {https://www.nature.com/articles/s41467-018-04316-3 }}

2. @article{Mocanu2016XBM,
author={Mocanu, Decebal Constantin and Mocanu, Elena and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
title={A topological insight into restricted Boltzmann machines},
journal={Machine Learning},
year={2016},
volume={104},
number={2},
pages={243--270},
doi={10.1007/s10994-016-5570-z},
url={https://doi.org/10.1007/s10994-016-5570-z }}

3. @phdthesis{Mocanu2017PhDthesis,
title = {Network computations in artificial intelligence},
author = {Mocanu, Decebal Constantin},
year = {2017},
isbn = {978-90-386-4305-2},
publisher = {Eindhoven University of Technology},
url={https://pure.tue.nl/ws/files/69949254/20170629_CO_Mocanu.pdf }
}

4. @article{Liu2019onemillion,
  author =        {Liu, Shiwei and Mocanu, Decebal Constantin and Mocanu and Ramapuram Matavalam, Amarsagar Reddy and Pei, Yulong Pei and Pechenizkiy, Mykola},
  journal =       {arXiv:1901.09181},
  title =         {Sparse evolutionary Deep Learning with over one million artificial neurons on commodity hardware},
  year =          {2019},
  url={https://arxiv.org/abs/1901.09181 }
}
