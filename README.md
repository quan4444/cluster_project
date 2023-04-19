## Cluster project

This GitHub repository contains a folder called ``tutorials`` that contains two examples, one for running the clustering pipeline on the homogeneous sample for sensors placement, and one for running the clustering pipeline on the heterogeneous samples to identify the different domains. To run the tutorials, change your current working directory to the ``tutorials`` folder.

### Preparing data for analysis
The data will be contained in the ``files/example_data/output_disp`` folder. Critically:
1. The files must have a ``'.npy'`` extension.
2. The files with name starting with ``'pt_'`` must contain the 2D or 3D locations of the markers.
3. The files with the name starting with ``'disp_'`` must contain the 2D or 3D displacements of the markers, corresponding to the files in 2.

Here is how the folders will be structured:
```bash
|___ example_folder
|	|___ output_disp
|		|___ 'pt_example1.npy'
|		|___ 'disp_example1.npy'
```

Here, we will import the necessary packages. We will also load an array ``pt_loc`` storing random markers location, and the corresponding displacements array ``u_mat``. We will use the function ``sample_points`` to sample an array of grid markers ``points_sel``.

```bash
import numpy as np
from cluster_project import kinematics as kn
from cluster_project import cluster, plotting

# user inputs for size of sample
length_samp = 1

# load markers positions and displacements
disp_path = '/home/quan/phd/soft_square_stiff_circle/output_disp'
pt_loc_files = np.array(['pt_homog_equi_disp0.4.npy','pt_homog_uni_y_disp0.4.npy',\
                        'pt_homog_uni_x_disp0.4.npy','pt_homog_shear_yf0.1.npy'])
u_mat_files = np.array(['disp_homog_equi_disp0.4.npy','disp_homog_uni_y_disp0.4.npy',\
                       'disp_homog_uni_x_disp0.4.npy','disp_homog_shear_yf0.1.npy'])
pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
disp_type = np.array(['equibiaxial','uniaxial y','uniaxial x','shear'])

# generate grid markers
pt_len = 8000
points_sel = kn.sample_points(pt_len,L=length_samp)
```

### Current core functionality

In this tutorial, there are _ core functionalities available.

#### Kinematics calculations

The function ``get_kinematics_with_nn`` will take in ``pt_loc``, ``u_mat``, and ``pt_sel``, and generate multiple array of kinematics (e.g., ``F``, ``invariants``) for the corresponding grid markers ``pt_sel``.

```bash
# obtain kinematics at grid markers for each file
num_neigh=40

u_mat_list,grad_u_list,strain_list,I_strain_list,F_list,I_F_list,C_list,I_C_list,b_list,I_b_list = kn.get_kinematics_multiple(pt_loc_all,u_mat_all,points_sel,num_neigh)
```

#### Clustering the domain

First, we select the feature we want to use for clustering (e.g., ``features_all = strain_list``). The function ``cluster_full_pipelines`` will take in the features ``features_all``, the number of clusters ``k``, and the grid markers ``points_sel``, and will output the ``cluster_results``. 
In the example below, we run multiple loops of ...

```bash
# cluster sets
features_all = strain_list
highest_k = 3
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
