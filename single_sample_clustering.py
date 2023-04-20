import numpy as np
from cluster_project import cluster, plotting
from cluster_project import kinematics as kn

# user inputs for size of sample
length_samp = 1

# load markers positions and displacements
disp_path = '/home/quan/phd/soft_square_stiff_circle/output_disp'
pt_loc_file = 'pt_homog_uni_y_disp0.4.npy'
u_mat_file = 'disp_homog_uni_y_disp0.4.npy'
pt_loc_og,u_mat_og = kn.load_pt_disp(disp_path,pt_loc_file,u_mat_file)

# generate grid markers
pt_len = 8000
points_sel = kn.sample_points(pt_len,L=length_samp)

# obtain kinematics for grid markers
num_neigh=40
u_mat_int,grad_u,strain,I_strain,F,I_F,C,I_C,b,I_b = kn.get_kinematics_with_nn(pt_loc_og,points_sel,u_mat_og,num_neigh)

# user input for number of sensors tuning
highest_k = 3
feature = I_strain
thresh = 5
filter_size = (5,5)
segment = True
positional_medoid = False

# loop
k_list = np.linspace(2,highest_k,highest_k-1,dtype=int)
medoids_ind_list=[]
feature_compressed_list=()
MSE_vs_k_features=[]

for i in range(len(k_list)):

    # pick k
    k_ = k_list[i]
    print(f'i={i} k={k_}')

    # perform clustering
    cluster_label,label_active,medoids_ind,feature_compressed,MSE_ = \
        cluster.cluster_single_set(feature,k_,thresh=thresh,filter_size=filter_size,segment=segment,\
                                   positional_medoid=positional_medoid)

    # append outputs
    medoids_ind_list.append(medoids_ind)
    feature_compressed_list = feature_compressed_list + (feature_compressed,)
    MSE_vs_k_features.append(MSE_)

    # plot the outputs
    if k_list[i] == 2 or k_list[i] % 10 == 0:
        plotting.plot_og_compressed_img(feature,feature_compressed,k_,MSE_)
        plotting.plot_cluster(cluster_label,points_sel,title_extra=' (naive)')
        plotting.plot_centroids_on_clusters(medoids_ind,points_sel,label_active)

medoids_ind_list = np.array(medoids_ind_list,dtype=object)
feature_compressed_list = np.array(feature_compressed_list,dtype=object)
MSE_vs_k_features = np.array(MSE_vs_k_features)

# plot MSE comparing compressed features vs original features
plotting.plot_MSE(k_list,MSE_vs_k_features,plot_type='k')
num_sensors = [len(array) for array in medoids_ind_list]
plotting.plot_MSE(num_sensors,MSE_vs_k_features,plot_type='num_sensors')