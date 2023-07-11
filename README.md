# Cluster project

[![python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![tests](https://github.com/elejeune11/microbundlecompute-lite/workflows/tests/badge.svg)](https://github.com/quan4444/cluster_project/actions)
[![codecov](https://codecov.io/gh/quan4444/cluster_project/branch/master/graph/badge.svg?token=W3LXJTOCY8)](https://codecov.io/gh/quan4444/cluster_project)

## Table of contents

* [Project Background and Summary](#summary)
* [Project Pipeline](#pipeline)
* [Tutorials](#tutorial)

## Project Background and Summary <a name="summary"></a>

The goal of this project is to cluster a domain into several sub-domains using the mechanical and positional information available. Broadly speaking, a domain can represent many physical world objects, such as a soft robot or a sample of tissue. We developed a clustering pipeline with 2 main applications in mind:
- Clustering a soft robot into different sub-domains for sensors placement.
- Clustering a heterogeneous soft tissue into mechanically homogeneous sub-domains.
Before explaining the full pipeline, we will introduce some concepts as background. Readers familiar with continuum mechanics and unsupervised machine learning can skip the rest of this section.

### Markers on Domain

In this project, we have a 3D domain with size 1 by 1 by 0.05. As we perform mechanical extensions (e.g., equibiaxial extension, uniaxial extension) on the domain, we can track its movements with randomly sampled markers at the surface of the domain. In many applications, we can use digital markers (e.g., Digital Image Correlation) or physical markers (e.g., hair follicles, skin pores) to track the deformations of the objects.

**add image for randomly sampled markers and domain here**

### Kinematics

Briefly, [Kinematics](https://en.wikipedia.org/wiki/Kinematics) is a field of study describing the motion of objects and markers. In this project, we will use the following kinematics (from [continuum mechanics](http://www.continuummechanics.org/)): displacement $u$, displacement gradient $\nabla u$, strain $E$, deformation gradient $F$, right Cauchy-Green $C$, left Cauchy-Green $b$, and invariants $I$.

### Unsupervised Learning/Clustering

In machine learning, [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) methods try to identify patterns within the data given no labels. Here, we use 2 common methods: [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering), and [Spectral Clustering](https://en.wikipedia.org/wiki/Spectral_clustering). Both methods split a set of data points into clusters with similar behaviors. Due to the lack of labels, unsupervised learning methods cannot utilize more accepted error metrics (e.g., mean squared error), which would require labels or a ground truth. This has led to the development of [clustering performance evaluation metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) to try and assess the performance of the methods.

### Ensemble Clustering

In many applications, we cannot determine the correctness of our unsupervised learning methods. However, we might be able to obtain different clustering results via different methods or via different scenarios, then we aggregate the different clustering results to retrieve the final consensus clustering result using [Ensemble Clustering](https://en.wikipedia.org/wiki/Consensus_clustering).

## Project Pipeline <a name="pipeline"></a>

The general pipeline is as follow:

`Select a domain with a corresponding heterogeneous pattern` $\mapsto$ `Select constitutive model(s)` $\mapsto$ `Select boundary condition(s)` $\mapsto\$ `Generate 1 set of displacement markers per boundary condition` $\mapsto$ `Calculate kinematics features for each set of markers` $\mapsto$ `Clustering pipeline`

The following section will describe the details for our clustering pipeline

### Clustering pipeline

From `Calculate kinematics features for each set of markers`, we will obtain the sets of kinematics for different boundary conditions. Then, we can follow our clustering pipeline to identify the sub-domains:

**image for clustering pipeline**

## Tutorials <a name="tutorial"></a>

This GitHub repository contains a folder called ``tutorials`` that contains two examples, one for running the clustering pipeline on the homogeneous sample for sensors placement, and one for running the clustering pipeline on the heterogeneous samples to identify the different material domains. To run the tutorials, change your current working directory to the ``tutorials`` folder.

### Preparing data for analysis

The data will be contained in the ``files/example_data/`` folder. Critically:
1. The files must have a ``'.npy'`` extension.
2. The files with name starting with ``'pt_'`` must contain the 2D or 3D locations of the markers.
3. The files with the name starting with ``'disp_'`` must contain the 2D or 3D displacements of the markers, corresponding to the ``'pt_'`` files.

Here is how the folders will be structured:
```bash
|___ files
|	|___ example_data
|		|___ 'pt_example1.npy'
|		|___ 'disp_example1.npy'
```

Here, we will import the necessary packages. We will select the files for random markers locations as ``pt_loc_files``, and the files for the corresponding displacements as ``u_mat_files``. Each pair of ``pt_loc_files`` and ``u_mat_files`` contains the information for a set of boundary condition constraints. After selecting the files, we will use the function ``load_multiple`` to load all random markers locations into ``pt_loc_all`` and all displacements into ``u_mat_all``. ``pt_loc_all`` and ``u_mat_all`` are m by n by dim arrays, where m is the number of boundary conditions, n the number of random markers, and dim the dimension of the data.

```python3
import numpy as np
from cluster_project import kinematics as kn
from cluster_project import cluster, plotting

# user inputs for size of sample
length_samp = 1

# load markers positions and displacements
disp_path = 'files/example_data/'
pt_loc_files = np.array(['pt_homog_equi_disp0.4.npy','pt_homog_uni_y_disp0.4.npy',\
                        'pt_homog_uni_x_disp0.4.npy','pt_homog_shear_yf0.1.npy'])
u_mat_files = np.array(['disp_homog_equi_disp0.4.npy','disp_homog_uni_y_disp0.4.npy',\
                       'disp_homog_uni_x_disp0.4.npy','disp_homog_shear_yf0.1.npy'])
pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
disp_type = np.array(['equibiaxial','uniaxial y','uniaxial x','shear'])
```

Finally, we can determine the number of markers in ``pt_len`` and sample grid markers with the function ``sample_points``. The function will automatically round ``pt_len`` down to the nearest perfect square, providing us a squared grid markers (e.g., 8000 markers will be rounded down to 7921 markers).

```python3
# generate grid markers
pt_len = 8000
points_sel = kn.sample_points(pt_len,L=length_samp)
```

### Current core functionality

In this tutorial, there are 2 core functionalities available.

#### Kinematics calculations

The function ``get_kinematics_with_nn`` will take in the random markers ``pt_loc_all``, the displacements ``u_mat_all``, the grid markers ``points_sel``, and the number of neighbors ``num_neigh``, and *generate multiple array of kinematics* (e.g., strain ``strain_list``, deformation gradient ``F_list``) for the corresponding grid markers ``points_sel``. Here, ``num_neigh`` is the number of nearest neighbor used to interpolate the displacement gradients for the grid markers. The output of the code contains multiple arrays of kinematics, each with shape of m by n by dim, where m is the number of boundary conditions, n the number of markers, and dim the dimensions of the kinematics. The detail of the kinematics is as follow:
- ``u_mat_list``: the displacements of the grid markers with shape of m by n by dim, where ``dim=2`` and ``u_mat_list[:,:,0]`` is the displacements in x, and ``u_mat_list[:,:,1]`` the displacements in y.
- ``grad_u_list``: the gradient of the displacements of the grid markers with shape of m by n by dim, where ``dim=4``. grad_u11, grad_u22, grad_u12, and grad_u21 correspond to ``grad_u_list[:,:,0]``,``grad_u_list[:,:,1]``, ``grad_u_list[:,:,2]``, and ``grad_u_list[:,:,3]``, respectively.
- ``strain_list``: the strain of the grid markers with shape of m by n by dim, where ``dim=4``. strain11, strain22, strain12, and strain21 correspond to ``strain_list[:,:,0]``,``strain_list[:,:,1]``, ``strain_list[:,:,2]``, and ``strain_list[:,:,3]``, respectively.
- ``I_strain_list``: the invariants of strain of the grid markers with shape of m by n by dim, where ``dim=2``. The first invariant and second invariant of strain correspond to ``I_strain_list[:,:,0]``, and ``I_strain_list[:,:,2]``, respectively.
- ``F_list``: the deformation gradient of the grid markers with shape of m by n by dim, where ``dim=4``. F11, F22, F12, and F21 correspond to ``F_list[:,:,0]``,``F_list[:,:,1]``, ``F_list[:,:,2]``, and ``F_list[:,:,3]``, respectively.
- ``I_F_list``: the invariants of the deformation gradient of the grid markers with shape of m by n by dim, where ``dim=2``. The first invariant and second invariant of the deformation gradient correspond to ``I_F_list[:,:,0]``, and ``I_F_list[:,:,2]``, respectively.
- ``C_list``: the right Cauchy-Green of the grid markers with shape of m by n by dim, where ``dim=4``. C11, C22, C12, and C21 correspond to ``C_list[:,:,0]``,``C_list[:,:,1]``, ``C_list[:,:,2]``, and ``C_list[:,:,3]``, respectively.
- ``I_C_list``: the invariants of the right Cauchy-Green of the grid markers with shape of m by n by dim, where ``dim=2``. The first invariant and second invariant of the right Cauchy-Green correspond to ``I_C_list[:,:,0]``, and ``I_C_list[:,:,2]``, respectively.
- ``b_list``: the left Cauchy-Green of the grid markers with shape of m by n by dim, where ``dim=4``. b11, b22, b12, and b21 correspond to ``b_list[:,:,0]``,``b_list[:,:,1]``, ``b_list[:,:,2]``, and ``b_list[:,:,3]``, respectively.
- ``I_b_list``: the invariants of the left Cauchy-Green of the grid markers with shape of m by n by dim, where ``dim=2``. The first invariant and second invariant of the left Cauchy-Green correspond to ``I_b_list[:,:,0]``, and ``I_b_list[:,:,2]``, respectively.

```python3
# obtain kinematics at grid markers for each file
num_neigh=40

u_mat_list,grad_u_list,strain_list,I_strain_list,F_list,I_F_list,C_list,I_C_list,b_list,I_b_list = kn.get_kinematics_multiple(pt_loc_all,u_mat_all,points_sel,num_neigh)
```

#### Clustering the domain

First, we select the feature we want to use for clustering (e.g., ``features_all = strain_list``). The function ``cluster_full_pipelines`` will take in ``features_all``, the number of clusters ``k``, and the grid markers ``points_sel``, and will output the clustering results and other variables useful for analysis, which will be discussed below.
- ``cluster_results``: the clustering results for the individual boundary conditions, with shape of m by n, where m is the number of boundary conditions, and n the number of grid markers. The values of the array correspond to the label of the markers.
- ``naive_ensemble_label``: the clustering result for the ensemble, obtained AFTER clustering the affinity matrix with Spectral clustering, but BEFORE the segmentation by position step. The array has the shape n by 1.
- ``ensemble_label``: the clustering result for the ensemble, obtain after the segmentation by position step. The array has the shape n by 1.
- ``medoids_ind``: the indices (``points_sel``) of the medoids for the clusters in ``ensemble_label``. Each cluster has 1 medoid.
- ``features_compressed_all`: The compressed features for all the boundary conditions. For each cluster, the compressed features are obtained by replacing the features of the medoid with all the markers in the cluster. The array has a shape m by n by dim, where m is the number of boundary conditions, n the number of grid markers, and dim the dimensions of the features.
- ``MSE_all``: the mean squared value between the ``features_compressed_all`` and the ``features_all``. The array has a shape m by 1, which each value corresponds to the MSE between the compressed features and the actual features for a set of boundary conditions constraints.

In the example below, we run multiple loops of our clustering pipeline, increasing the number of clusters ``k`` between loop. The for loop will run from ``k=2`` to ``k=highest_k``, where ``highest_k`` is the highest value of k predetermined by the user. The goal is to find the *convergence* numbers of clusters ``k``. We store the data in an array for plotting later.

```python3
# cluster sets
features_all = strain_list
highest_k = 30
thresh = 5
filter_size = (5,5)
segment = True
positional_medoid = False

k_list = np.linspace(2,highest_k,highest_k-1,dtype=int)
medoids_ind_list = []
feature_compressed_list=()
MSE_vs_k_features=[]
for i in range(len(k_list)):

	k_ = k_list[i]
	print(f'i={i} k={k_}')

	cluster_results,naive_ensemble_label,ensemble_label,medoids_ind,features_compressed_all,MSE_all = \
		cluster.cluster_full_pipeline(features_all,k_,points_sel,thresh=thresh,filter_size=filter_size,\
				segment=segment,positional_medoid=positional_medoid)

	medoids_ind_list.append(medoids_ind)
	feature_compressed_list = feature_compressed_list + (features_compressed_all,)
	MSE_vs_k_features.append(MSE_all)
medoids_ind_list = np.array(medoids_ind_list,dtype=object)
MSE_vs_k_features = np.array(MSE_vs_k_features)
```
