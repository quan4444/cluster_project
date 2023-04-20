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

# generate grid markers
pt_len = 8000
points_sel = kn.sample_points(pt_len,L=length_samp)

# obtain kinematics at grid markers for each file
num_neigh=40

_,_,strain_list,I_strain_list,_,_,_,I_C_list,_,_ = kn.get_kinematics_multiple(pt_loc_all,u_mat_all,points_sel,num_neigh)

# cluster sets
features_all = I_C_list
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

	if k_list[i] == 2 or k_list[i] % 10 == 0:
		plotting.plot_cluster_by_bcs(disp_type,cluster_results,points_sel,big_title='boundary conditions')
		plotting.plot_cluster(naive_ensemble_label,points_sel,title_extra=' (naive)')
		plotting.plot_cluster(ensemble_label,points_sel,title_extra=' (segmented)')
		plotting.plot_centroids_on_clusters(medoids_ind,points_sel,ensemble_label)
medoids_ind_list = np.array(medoids_ind_list,dtype=object)
MSE_vs_k_features = np.array(MSE_vs_k_features)

plotting.plot_MSE_multiple(k_list,MSE_vs_k_features,disp_type,big_title='MSE vs. k',x_axis_label='k')
num_sensors = [len(array) for array in medoids_ind_list]
plotting.plot_MSE_multiple(num_sensors,MSE_vs_k_features,disp_type,big_title='MSE vs. num sensors',x_axis_label='num sensors',scatter_plot=True)