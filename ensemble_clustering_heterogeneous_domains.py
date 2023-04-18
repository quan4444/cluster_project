import numpy as np
import sys
sys.path.append('/home/quan/phd/cluster_project/src')

import kinematics as kn
import cluster
import plotting

# user inputs for size of sample
length_samp = 1

# load markers positions and displacements
disp_path = '/home/quan/phd/soft_square_stiff_circle/output_disp'

# # sssc / neo-hookean
# pt_loc_files = np.array(['pt_sssc_biax_x00.4_xf0.4_y00.4_yf0.4.npy','pt_sssc_shear_yf0.4.npy',\
#                         'pt_sssc_confinedcomp_yf0.25.npy','pt_sssc_uni_x_disp0.4.npy',\
#                         'pt_sssc_uni_y_disp0.4.npy'])
# u_mat_files = np.array(['disp_sssc_biax_x00.4_xf0.4_y00.4_yf0.4.npy','disp_sssc_shear_yf0.4.npy',\
# 						'disp_sssc_confinedcomp_yf0.25.npy','disp_sssc_uni_x_disp0.4.npy',\
# 						'disp_sssc_uni_y_disp0.4.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['equibiaxial','shear','confined compression','uni x','uni y'])

# # sssc / ho / background fibers point up / circular inclusion 45deg from x-axis
# pt_loc_files = np.array(['pt_sssc_HO_bgup_45_unid_y_x00.2_xf0.2_y00.0_yf0.0_rank0.npy',\
#                         'pt_sssc_HO_bgup_45_unid_x00.0_xf0.0_y00.2_yf0.2_rank0.npy'])
# u_mat_files = np.array(['disp_sssc_HO_bgup_45_unid_y_x00.2_xf0.2_y00.0_yf0.0_rank0.npy',\
#                             'disp_sssc_HO_bgup_45_unid_x00.0_xf0.0_y00.2_yf0.2_rank0.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['confined uni x','confined uni y'])

# sssc / ho / background fibers along x-axis / circular inclusion 45deg from x-axis
pt_loc_files = np.array(['pt_sssc_HO_45_equi_x00.2_xf0.2_y00.2_yf0.2_rank0.npy',\
                        'pt_sssc_HO_45_unid_x00.0_xf0.0_y00.2_yf0.2_rank0.npy',\
						'pt_sssc_HO_45_unid_y_x00.2_xf0.2_y00.0_yf0.0_rank0.npy'])
u_mat_files = np.array(['disp_sssc_HO_45_equi_x00.2_xf0.2_y00.2_yf0.2_rank0.npy',\
                        'disp_sssc_HO_45_unid_x00.0_xf0.0_y00.2_yf0.2_rank0.npy',\
						'disp_sssc_HO_45_unid_y_x00.2_xf0.2_y00.0_yf0.0_rank0.npy'])
pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
disp_type = np.array(['equibiaxial','confined uni x','confined uni y'])

# generate grid markers
pt_len = 8000
points_sel = kn.sample_points(pt_len,L=length_samp)

# obtain kinematics at grid markers for each file
num_neigh=40

_,_,_,_,F_list,_,_,I_C_list,_,_ = kn.get_kinematics_multiple(pt_loc_all,u_mat_all,points_sel,num_neigh)

# cluster sets
features_all = F_list
highest_k = 2
thresh = 5
filter_size = (5,5)
segment = True
positional_medoid = False
domain_type = 'circle'

# obtain ground truth
truth = cluster.get_ground_truth(points_sel,length=length_samp,\
				 width=length_samp,het_domain=domain_type)

k_list = np.linspace(2,highest_k,highest_k-1,dtype=int)
for i in range(len(k_list)):

	k_ = k_list[i]
	print(f'i={i} k={k_}')

	cluster_results,naive_ensemble_label,ensemble_label = \
		cluster.cluster_full_pipeline(features_all,k_,points_sel,thresh=thresh,filter_size=filter_size,\
				segment=segment,positional_medoid=positional_medoid,only_label=True)

	cluster_results_ARI = cluster.get_ARI_multiple(truth,cluster_results)
	ensemble_ARI = cluster.get_ARI_multiple(truth,ensemble_label)

	if k_list[i] == 2 or k_list[i] % 10 == 0:
		plotting.plot_cluster_and_ARI_by_bcs(disp_type,cluster_results,points_sel,cluster_results_ARI,big_title='boundary conditions')
		plotting.plot_cluster(naive_ensemble_label,points_sel,title_extra=' (naive)')
		plotting.plot_cluster_ARI(ensemble_label,points_sel,ensemble_ARI,title_extra='segmented ensemble')

# plotting.plot_MSE_multiple(k_list,MSE_vs_k_features,disp_type,big_title='MSE vs. k',x_axis_label='k')
# num_sensors = [len(array) for array in medoids_ind_list]
# plotting.plot_MSE_multiple(num_sensors,MSE_vs_k_features,disp_type,big_title='MSE vs. num sensors',x_axis_label='num sensors',scatter_plot=True)