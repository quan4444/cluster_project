import numpy as np
from cluster_project import kinematics as kn
from cluster_project import cluster, plotting

# user inputs for size of sample
length_samp = 1

# load markers positions and displacements

# # circle inclusion / neo-hookean
# disp_path = 'files/example_data/circle_inclusion_NH'
# pt_loc_files = np.array(['pt_sssc_equi_disp0.4.npy','pt_sssc_uni_y_disp0.4.npy',\
#                         'pt_sssc_uni_x_disp0.4.npy','pt_sssc_shear_0.4.npy',\
#                         'pt_sssc_comp_0.2.npy'])
# u_mat_files = np.array(['disp_sssc_equi_disp0.4.npy','disp_sssc_uni_y_disp0.4.npy',\
# 						'disp_sssc_uni_x_disp0.4.npy','disp_sssc_shear_0.4.npy',\
# 						'disp_sssc_comp_0.2.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['equibiaxial','uniaxial y','uniaxial x','shear','confined compression'])
# domain_type = 'circle_inclusion' # necessary for obtaining ground truth / ARI

# # 4 circle inclusions / neo-hookean
# disp_path = 'files/example_data/four_circle_inclusions_NH'
# pt_loc_files = np.array(['pt_4cirs_equi_disp0.4.npy','pt_4cirs_uni_y_disp0.4.npy',\
#                         'pt_4cirs_uni_x_disp0.4.npy','pt_4cirs_shear_yf0.4.npy',\
#                         'pt_4cirs_comp_yf0.2.npy'])
# u_mat_files = np.array(['disp_4cirs_equi_disp0.4.npy','disp_4cirs_uni_y_disp0.4.npy',\
# 						'disp_4cirs_uni_x_disp0.4.npy','disp_4cirs_shear_yf0.4.npy',\
# 						'disp_4cirs_comp_yf0.2.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['equibiaxial','uniaxial y','uniaxial x','shear','confined compression'])
# domain_type = '4_circle_inclusions' # necessary for obtaining ground truth / ARI

# # cross inclusions / neo-hookean
# disp_path = 'files/example_data/cross_inclusion_NH'
# pt_loc_files = np.array(['pt_cross_equi_disp0.4.npy','pt_cross_uni_y_disp0.4.npy',\
#                         'pt_cross_uni_x_disp0.4.npy','pt_cross_shear_yf0.4.npy',\
# 						'pt_cross_comp_yf0.2.npy'])
# u_mat_files = np.array(['disp_cross_equi_disp0.4.npy','disp_cross_uni_y_disp0.4.npy',\
# 						'disp_cross_uni_x_disp0.4.npy','disp_cross_shear_yf0.4.npy',\
# 						'disp_cross_comp_yf0.2.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['equibiaxial','uniaxial y','uniaxial x','shear','confined compression'])
# domain_type = 'cross' # necessary for obtaining ground truth / ARI

# # ring inclusions / neo-hookean
# disp_path = 'files/example_data/ring_inclusion_NH'
# pt_loc_files = np.array(['pt_ring_equi_disp0.4.npy','pt_ring_uni_y_disp0.4.npy',\
#                         'pt_ring_uni_x_disp0.4.npy','pt_ring_shear_yf0.075.npy',\
# 						'pt_ring_comp_yf0.2.npy'])
# u_mat_files = np.array(['disp_ring_equi_disp0.4.npy','disp_ring_uni_y_disp0.4.npy',\
# 						'disp_ring_uni_x_disp0.4.npy','disp_ring_shear_yf0.075.npy',\
# 						'disp_ring_comp_yf0.2.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['equibiaxial','uniaxial y','uniaxial x','shear','confined compression'])
# domain_type = 'ring' # necessary for obtaining ground truth / ARI

# # Cahn-Hilliard Image12 / neo-hookean
# disp_path = 'files/example_data/Cahn_Hilliard_Image12_NH'
# pt_loc_files = np.array(['pt_CH_Image12_equi_disp0.4.npy','pt_CH_Image12_uni_y_disp0.4.npy',\
#                         'pt_CH_Image12_uni_x_disp0.4.npy','pt_CH_Image12_shear_yf0.2.npy',\
# 						'pt_CH_Image12_comp_yf0.1.npy'])
# u_mat_files = np.array(['disp_CH_Image12_equi_disp0.4.npy','disp_CH_Image12_uni_y_disp0.4.npy',\
# 						'disp_CH_Image12_uni_x_disp0.4.npy','disp_CH_Image12_shear_yf0.2.npy',\
# 						'disp_CH_Image12_comp_yf0.1.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['equibiaxial','uniaxial y','uniaxial x','shear','confined compression'])
# domain_type = 'cahn_hilliard_image12' # necessary for obtaining ground truth / ARI

# # half half / holzapfel-ogden - left half 45deg from y-axis, right half parallel to y-axis
# disp_path = 'files/example_data/halfhalf_HO'
# pt_loc_files = np.array(['pt_halfhalf_equi_disp0.25.npy','pt_halfhalf_uni_y_disp0.2.npy',\
#                         'pt_halfhalf_uni_x_disp0.2.npy','pt_halfhalf_comp_x_xf0.2.npy',\
# 						'pt_halfhalf_comp_y_yf0.2.npy']) #,'pt_halfhalf_shear_yf0.2.npy',\
# u_mat_files = np.array(['disp_halfhalf_equi_disp0.25.npy','disp_halfhalf_uni_y_disp0.2.npy',\
# 						'disp_halfhalf_uni_x_disp0.2.npy','disp_halfhalf_comp_x_xf0.2.npy',\
# 						'disp_halfhalf_comp_y_yf0.2.npy']) # ,'disp_halfhalf_shear_yf0.2.npy',\
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['equibiaxial','uniaxial y','uniaxial x','confined compression x','confined compression y'])
# domain_type = 'halfhalf' # necessary for obtaining ground truth / ARI

# # circle inclusion / neo-hookean / random boundary conditions
# disp_path = 'files/example_data/circle_inclusion_NH_random'
# pt_loc_files = np.array(['pt_cir_random_seed_1.npy','pt_cir_random_seed_2.npy',\
#                         'pt_cir_random_seed_3.npy','pt_cir_random_seed_4.npy',\
# 						'pt_cir_random_seed_5.npy','pt_cir_random_seed_6.npy'])
# u_mat_files = np.array(['disp_cir_random_seed_1.npy','disp_cir_random_seed_2.npy',\
# 						'disp_cir_random_seed_3.npy','disp_cir_random_seed_4.npy',\
# 						'disp_cir_random_seed_5.npy','disp_cir_random_seed_6.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['seed 1','seed 2','seed 3','seed 4','seed 5','seed 6']) # 
# domain_type = 'circle_inclusion' # necessary for obtaining ground truth / ARI

# # ring inclusion / neo-hookean / random boundary conditions
# disp_path = 'files/example_data/ring_inclusion_NH_random'
# pt_loc_files = np.array(['pt_ring_random_seed_100.npy','pt_ring_random_seed_101.npy',\
#                         'pt_ring_random_seed_102.npy','pt_ring_random_seed_103.npy',\
# 						'pt_ring_random_seed_105.npy','pt_ring_random_seed_107.npy'])
# u_mat_files = np.array(['disp_ring_random_seed_100.npy','disp_ring_random_seed_101.npy',\
# 						'disp_ring_random_seed_102.npy','disp_ring_random_seed_103.npy',\
# 						'disp_ring_random_seed_105.npy','disp_ring_random_seed_107.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['seed 100','seed 101','seed 102','seed 103','seed 105','seed 107'])
# domain_type = 'ring' # necessary for obtaining ground truth / ARI

# # 4 circle inclusions / neo-hookean / random boundary conditions
# disp_path = 'files/example_data/four_circle_inclusions_NH_random'
# # pt_loc_files = np.array(['pt_4cirs_random_seed_7.npy','pt_4cirs_random_seed_8.npy',\
# #                         'pt_4cirs_random_seed_9.npy','pt_4cirs_random_seed_10.npy',\
# # 						'pt_4cirs_random_seed_11.npy','pt_4cirs_random_seed_12.npy'])
# # u_mat_files = np.array(['disp_4cirs_random_seed_7.npy','disp_4cirs_random_seed_8.npy',\
# # 						'disp_4cirs_random_seed_9.npy','disp_4cirs_random_seed_10.npy',\
# # 						'disp_4cirs_random_seed_11.npy','disp_4cirs_random_seed_12.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['seed 7','seed 8','seed 9','seed 10','seed 11','seed 12']) # 
# domain_type = '4_circle_inclusions' # necessary for obtaining ground truth / ARI

# # cross inclusion / neo-hookean / random boundary conditions
# disp_path = 'files/example_data/cross_inclusion_NH_random'
# pt_loc_files = np.array(['pt_cross_random_seed_13.npy','pt_cross_random_seed_14.npy',\
#                         'pt_cross_random_seed_15.npy','pt_cross_random_seed_16.npy',\
# 						'pt_cross_random_seed_17.npy','pt_cross_random_seed_18.npy'])
# u_mat_files = np.array(['disp_cross_random_seed_13.npy','disp_cross_random_seed_14.npy',\
# 						'disp_cross_random_seed_15.npy','disp_cross_random_seed_16.npy',\
# 						'disp_cross_random_seed_17.npy','disp_cross_random_seed_18.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['seed 13','seed 14','seed 15','seed 16','seed 17','seed 18']) # 
# domain_type = 'cross' # necessary for obtaining ground truth / ARI

# Cahn-Hilliard Image 12 / neo-hookean / random boundary conditions
disp_path = 'files/example_data/Cahn_Hilliard_Image12_NH_random'
# pt_loc_files = np.array(['pt_CH_Image12_random_seed_111.npy','pt_CH_Image12_random_seed_112.npy',\
#                         'pt_CH_Image12_random_seed_113.npy','pt_CH_Image12_random_seed_115.npy',\
# 						'pt_CH_Image12_random_seed_119.npy','pt_CH_Image12_random_seed_120.npy'])
# u_mat_files = np.array(['disp_CH_Image12_random_seed_111.npy','disp_CH_Image12_random_seed_112.npy',\
# 						'disp_CH_Image12_random_seed_113.npy','disp_CH_Image12_random_seed_115.npy',\
# 						'disp_CH_Image12_random_seed_119.npy','disp_CH_Image12_random_seed_120.npy'])
pt_loc_files = np.array(['pt_CH_Image12_random_seed_111.npy','pt_CH_Image12_random_seed_112.npy'])
u_mat_files = np.array(['disp_CH_Image12_random_seed_111.npy','disp_CH_Image12_random_seed_112.npy'])
pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
disp_type = np.array(['seed 111','seed 112']) # ,'seed 113','seed 115','seed 119','seed 120'
domain_type = 'cahn_hilliard_image12' # necessary for obtaining ground truth / ARI

# # split / holzapfel-ogden / random boundary conditions
# disp_path = 'files/example_data/halfhalf_HO_random'
# pt_loc_files = np.array(['pt_halfhalf_HO_random_seed_21.npy','pt_halfhalf_HO_random_seed_22.npy',\
#                         'pt_halfhalf_HO_random_seed_23.npy','pt_halfhalf_HO_random_seed_24.npy',\
# 						'pt_halfhalf_HO_random_seed_25.npy','pt_halfhalf_HO_random_seed_26.npy'])
# u_mat_files = np.array(['disp_halfhalf_HO_random_seed_21.npy','disp_halfhalf_HO_random_seed_22.npy',\
# 						'disp_halfhalf_HO_random_seed_23.npy','disp_halfhalf_HO_random_seed_24.npy',\
# 						'disp_halfhalf_HO_random_seed_25.npy','disp_halfhalf_HO_random_seed_26.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['seed 21','seed 22','seed 23','seed 24','seed 25','seed 26']) # 
# domain_type = 'halfhalf' # necessary for obtaining ground truth / ARI

# # circle inclusion / holzapfel-ogden / random boundary conditions
# disp_path = 'files/example_data/circle_inclusion_HO_random'
# pt_loc_files = np.array(['pt_cir_HO_random_seed_27.npy','pt_cir_HO_random_seed_28.npy',\
#                         'pt_cir_HO_random_seed_29.npy','pt_cir_HO_random_seed_30.npy',\
# 						'pt_cir_HO_random_seed_31.npy','pt_cir_HO_random_seed_32.npy'])
# u_mat_files = np.array(['disp_cir_HO_random_seed_27.npy','disp_cir_HO_random_seed_28.npy',\
# 						'disp_cir_HO_random_seed_29.npy','disp_cir_HO_random_seed_30.npy',\
# 						'disp_cir_HO_random_seed_31.npy','disp_cir_HO_random_seed_32.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['seed 27','seed 28','seed 29','seed 30','seed 31','seed 32']) # 
# domain_type = 'circle_inclusion' # necessary for obtaining ground truth / ARI

# # circle inclusion / holzapfel-ogden / different background materials / random boundary conditions
# disp_path = 'files/example_data/circle_inclusion_HO_diff_mat_random'
# pt_loc_files = np.array(['pt_cir_HO_diff_mat_random_seed_33.npy','pt_cir_HO_diff_mat_random_seed_34.npy',\
#                         'pt_cir_HO_diff_mat_random_seed_35.npy','pt_cir_HO_diff_mat_random_seed_36.npy',\
# 						'pt_cir_HO_diff_mat_random_seed_37.npy','pt_cir_HO_diff_mat_random_seed_38.npy'])
# u_mat_files = np.array(['disp_cir_HO_diff_mat_random_seed_33.npy','disp_cir_HO_diff_mat_random_seed_34.npy',\
# 						'disp_cir_HO_diff_mat_random_seed_35.npy','disp_cir_HO_diff_mat_random_seed_36.npy',\
# 						'disp_cir_HO_diff_mat_random_seed_37.npy','disp_cir_HO_diff_mat_random_seed_38.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['seed 33','seed 34','seed 35','seed 36','seed 37','seed 38']) # 
# domain_type = 'circle_inclusion' # necessary for obtaining ground truth / ARI

# # split inclusion / holzapfel-ogden / different background materials / random boundary conditions
# disp_path = 'files/example_data/halfhalf_HO_diff_mat_random'
# pt_loc_files = np.array(['pt_halfhalf_HO_diff_mat_random_seed_39.npy','pt_halfhalf_HO_diff_mat_random_seed_40.npy',\
#                         'pt_halfhalf_HO_diff_mat_random_seed_41.npy','pt_halfhalf_HO_diff_mat_random_seed_42.npy',\
# 						'pt_halfhalf_HO_diff_mat_random_seed_43.npy','pt_halfhalf_HO_diff_mat_random_seed_44.npy'])
# u_mat_files = np.array(['disp_halfhalf_HO_diff_mat_random_seed_39.npy','disp_halfhalf_HO_diff_mat_random_seed_40.npy',\
# 						'disp_halfhalf_HO_diff_mat_random_seed_41.npy','disp_halfhalf_HO_diff_mat_random_seed_42.npy',\
# 						'disp_halfhalf_HO_diff_mat_random_seed_43.npy','disp_halfhalf_HO_diff_mat_random_seed_44.npy'])
# pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_files,u_mat_files)
# disp_type = np.array(['seed 39','seed 40','seed 41','seed 42','seed 43','seed 44']) # 
# domain_type = 'halfhalf' # necessary for obtaining ground truth / ARI

# generate grid markers
pt_len = 8000
points_sel = kn.sample_points(pt_len,L=length_samp)

# obtain kinematics at grid markers for each file
num_neigh=40

_,_,_,_,F_list,_,_,I_C_list,_,_ = kn.get_kinematics_multiple(pt_loc_all,u_mat_all,points_sel,num_neigh)

# cluster sets
features_all = I_C_list
highest_k = 2
min_clus_size = 5
max_clus_size = pt_len
filter_size = (5,5)
segment = True
positional_medoid = False

# obtain ground truth
if domain_type == 'cahn_hilliard_image12':
	domain_path = 'files/example_data/Cahn_Hilliard_Image12_NH/'
else:
	domain_path = None
truth = cluster.get_ground_truth(points_sel,length=length_samp,\
				 width=length_samp,het_domain=domain_type,path=domain_path)

k_list = np.linspace(2,highest_k,highest_k-1,dtype=int)
for i in range(len(k_list)):

	k_ = k_list[i]
	print(f'i={i} k={k_}')

	cluster_results,naive_ensemble_label,ensemble_label = \
		cluster.cluster_full_pipeline(features_all,k_,points_sel,min_clus_size=min_clus_size,\
				max_clus_size=max_clus_size,filter_size=filter_size,segment=segment,\
				positional_medoid=positional_medoid,only_label=True)

	cluster_results_ARI = cluster.get_ARI_multiple(truth,cluster_results)
	ensemble_ARI = cluster.get_ARI_multiple(truth,ensemble_label)

	# if k_list[i] == 2 or k_list[i] % 10 == 0:
	plotting.plot_cluster_and_ARI_by_bcs(disp_type,cluster_results,points_sel,cluster_results_ARI,big_title='boundary conditions')
	plotting.plot_cluster(naive_ensemble_label,points_sel,title_extra=' (naive)')
	plotting.plot_cluster_ARI(ensemble_label,points_sel,ensemble_ARI,title_extra='segmented ensemble')

# plotting.plot_MSE_multiple(k_list,MSE_vs_k_features,disp_type,big_title='MSE vs. k',x_axis_label='k')
# num_sensors = [len(array) for array in medoids_ind_list]
# plotting.plot_MSE_multiple(num_sensors,MSE_vs_k_features,disp_type,big_title='MSE vs. num sensors',x_axis_label='num sensors',scatter_plot=True)