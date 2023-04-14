import numpy as np
import sklearn
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances_argmin_min, adjusted_rand_score
from skimage import measure, filters
import copy

def get_kmeans_label(x,n=2):
    '''Normalize and perform k-mean clustering on a set of feature x.'''

    x = sklearn.preprocessing.normalize(x,axis=0)
    kmeans = KMeans(n_clusters = n, init = 'k-means++', max_iter = 300, n_init=10)
    kmeans.fit(x)
    return kmeans

def arr_to_img(arr):
    '''Convert array/label to image, ensuring the orientation of the label relative to the corresponding markers positions.'''

    n = len(arr)
    n_sqrt = int(np.sqrt(n))
    # setting order='F' and np.flipud to put the image into the matching orientation
    # technically we don't have to do this step if we're gonna inverse this later, but this exists for clarity
    img_arr = np.reshape(arr, (n_sqrt, n_sqrt), order='F')
    img_arr = np.flipud(img_arr)
    return img_arr

def img_to_arr(img):
    '''Convert image to array/label, ensuring the orientation of the label relative to the corresponding markers positions.'''

    arr = np.rot90(img,k=-1).flatten()
    return arr

def label_by_position(img):
    '''Segment all the pixels in an image using the skimage.measure.label function'''

    img_labeled = measure.label(img+1)
    return img_labeled

def find_small_regions(img_labeled,thresh=5):
    '''Find regions inside the image with the number of pixels lower than the threshold.'''

    # find regions with size below the threshold
    region_props = measure.regionprops(img_labeled)
    below_thresh_label=[]

    for region in region_props:
        if region.area <= thresh:
            below_thresh_label.append(region.label)

    return np.array(below_thresh_label)

def set_small_regions_zero(img_labeled,below_thresh_label):
    '''Set the values of the regions with size lower than the threshold to 0.'''

    has_zero = np.any(img_labeled == 0)
    if has_zero == True:
        img_labeled += 1
        below_thresh_label += 1
    img_zero_regions = copy.deepcopy(img_labeled)
    img_zero_regions = np.array(img_zero_regions,dtype=np.float64)
    zero_idx = np.where(np.isin(img_zero_regions,below_thresh_label))
    img_zero_regions[zero_idx]=0
    return img_zero_regions,zero_idx

def median_filter(img, filter_size=(3,3)):
    '''Perform median filter on an image given image and filter size.'''

    filtered = filters.median(img,footprint=np.ones(filter_size))
    return filtered

def replace_small_regions_vals(img,img_filtered):
    '''Replace the values of the small regions in an image with the median filtered values.'''

    img[img==0] = img_filtered[img==0]
    return img

def clean_tiny_clusters(img_labeled,thresh=5,filter_size=(5,5)):
    '''Remove the small regions in an image with a median filter.'''

    below_thresh_label = find_small_regions(img_labeled,thresh=thresh)
    img_zero_regions,_ = set_small_regions_zero(img_labeled,below_thresh_label)
    img_filtered = median_filter(img_zero_regions,filter_size=filter_size)
    img_replaced_zeros = replace_small_regions_vals(img_zero_regions,img_filtered)
    final_img = label_by_position(img_replaced_zeros)
    return final_img

def segment_by_position(label,thresh=5,filter_size=(5,5)):
    '''Segment the different clusters in an image by their position.'''

    img = arr_to_img(label)
    img_labeled = label_by_position(img)
    final_img = clean_tiny_clusters(img_labeled,thresh=thresh,filter_size=filter_size)
    # final_img = clean_tiny_clusters(img_cleaned,thresh=thresh,filter_size=filter_size)
    final_label = img_to_arr(final_img)
    return final_label

def obtain_ind_in_clusters(labels):
    '''Obtain the indices per cluster for all cluster.'''

    cluster=[]
    for unique_label in np.unique(labels):
        cluster.append(np.where(labels==unique_label)[0])
    cluster = np.asarray(cluster,dtype=list)
    return cluster

def find_medoid_ind(feature,labels,markers=None,positional_medoid=False):
    '''Find the medoid indices for all clusters given the feature and labels.'''

    if positional_medoid == False:
        feature_for_medoid = feature
    elif positional_medoid == True:
        feature_for_medoid = markers

    cluster = obtain_ind_in_clusters(labels)

    medoid_ind = []
    for i in range(len(cluster)):
        feature_within_cluster = feature_for_medoid[cluster[i].tolist(),:]
        medoid_pos = np.mean(feature_within_cluster,axis=0)
        medoid_ind_rel_cluster,_ = pairwise_distances_argmin_min(medoid_pos.reshape(1,-1),feature_within_cluster)
        medoid_ind_global = cluster[i][medoid_ind_rel_cluster][0]
        medoid_ind.append(medoid_ind_global)
    medoid_ind = np.sort(np.array(medoid_ind))

    return medoid_ind

def get_medoid_val_by_sorted_labels(feature,medoids_ind,labels):
    '''Obtain the feature for the medoids, given the medoids indices.'''

    medoids_labels = labels[medoids_ind]
    medoids_labels_and_medoids_ind = np.concatenate((medoids_labels.reshape(-1,1),medoids_ind.reshape(-1,1)),axis=1)
    medoids_ind_sorted = medoids_labels_and_medoids_ind[np.argsort(medoids_labels_and_medoids_ind[:,0]),1]
    medoids_val_sorted = feature[medoids_ind_sorted,:]
    return medoids_val_sorted

def get_compressed_features_single(feature,labels,medoids_ind):
    '''Obtain the compressed features for each cluster.
    For each cluster, replace the feature values for all markers in the cluster with the feature values of the medoid.'''
    
    feature_compressed = copy.deepcopy(feature)
    unique_labels = np.unique(labels)
    medoids_val_sorted = get_medoid_val_by_sorted_labels(feature,medoids_ind,labels)
    for i in range(len(unique_labels)):
        pt_in_cluster_ind = np.where(labels==unique_labels[i])
        feature_center = medoids_val_sorted[i]
        feature_compressed[pt_in_cluster_ind,:]=feature_center
    return feature_compressed

def get_compressed_features_multiple(features_all,labels,medoids_ind):
    '''Obtain the compressed feature for multiple sets of labels and medoid indices.
    For each cluster, replace the feature values for all markers in the cluster with the feature values of the medoid.'''
    
    feature_compressed_all = ()
    for i in range(len(features_all)):
        feature = features_all[i]
        feature_compressed = get_compressed_features_single(feature,labels,medoids_ind)
        feature_compressed_all = feature_compressed_all + (feature_compressed,)
    feature_compressed_all = np.array(feature_compressed_all)
    return feature_compressed_all

def MSE(A,B,ax=0):
    '''Mean squared error between 2 matrices.'''

    mse = (np.square(A - B)).mean(axis=ax)
    return mse

def get_MSE_multiple(features,features_compressed_all):
    '''Mean squared error for multiple sets of matrices'''

    MSE_all = []
    for i in range(len(features)):
        mse = MSE(features[i],features_compressed_all[i])
        MSE_all.append(mse.mean())
    return MSE_all

def cluster_single_set(feature,k,thresh=5,filter_size=(5,5),segment=True,positional_medoid=False,only_cluster=False):
    '''Perform the full clustering pipeline on a single set of features.
    First, cluster the features using kmeans.
    Then, convert the label to an image and perform image segmentation on the labels.'''
    
    # kmeans clustering
    cluster_label = get_kmeans_label(feature,n=k).labels_

    if segment == True:
        # segment by position
        label_seg = segment_by_position(cluster_label,thresh=thresh,filter_size=filter_size)
        label_active = label_seg
    elif segment == False:
        label_active = cluster_label
    
    # if only_cluster == True, return cluster label
    # if only_cluster == False, find medoids of cluster and perform feature compression
    if only_cluster == True:
        return label_active
    elif only_cluster == False:
        # obtain indices of medoids
        medoids_ind = find_medoid_ind(feature,label_active,positional_medoid=positional_medoid)

        # obtain compressed features
        feature_compressed = get_compressed_features_single(feature,label_active,medoids_ind)

        # MSE between feature and feature_compressed
        MSE_ = MSE(feature,feature_compressed).mean()

        return label_active,cluster_label,medoids_ind,feature_compressed,MSE_
    
def cluster_sets(features_all,k=2,thresh=5,filter_size=(5,5),segment=True,positional_medoid=False):
    '''Perform the clustering pipeline on multiple sets of features.'''

    cluster_results = []
    for i in range(len(features_all)):
        feature=features_all[i]
        label = cluster_single_set(feature,k,thresh=thresh,filter_size=filter_size,segment=segment,positional_medoid=positional_medoid,only_cluster=True)
        cluster_results.append(label)
    cluster_results = np.array(cluster_results,dtype=object)
    return cluster_results

def get_indicator_matrix(cluster_results):
    '''Obtain the indicator matrix given multiple sets of labels.'''

    num_cluster_results,num_markers=cluster_results.shape

    shape1 = 0
    for i in range(num_cluster_results):
        num_clusters = len(np.unique(cluster_results[i]))
        shape1 += num_clusters

    indicator_mat = np.zeros((num_markers,shape1),dtype=int)

    col_count = 0
    for i in range(num_cluster_results):
        cur_labels = cluster_results[i]
        unique_labels = np.unique(cur_labels)
        num_clusters = len(unique_labels)
        for label_ in unique_labels:
            ind_label_ = np.where(cur_labels==label_)[0]
#             print(ind_label_)
            indicator_mat[ind_label_,col_count]=1
            col_count+=1
    return indicator_mat,num_cluster_results

def get_similarity_matrix(cluster_results):
    '''Obtain the similarity matrix given multiple sets of labels.'''

    # convert to hypergraph
    indicator_mat,num_cluster_results=get_indicator_matrix(cluster_results)

    # similarity matrix
    S=np.matmul(indicator_mat,indicator_mat.transpose())/num_cluster_results
    
    return S

def cluster_similarity_matrix(S,k,thresh=5,filter_size=(5,5),segment=True):
    '''Cluster the similarity matrix '''

    spectral = SpectralClustering(n_clusters=k,affinity='precomputed')
    spectral.fit(S)
    cluster_label = spectral.labels_

    if segment == True:
        # segment by position
        label_seg = segment_by_position(cluster_label,thresh=thresh,filter_size=filter_size)
        label_active = label_seg
    elif segment == False:
        label_active = cluster_label

    return label_active,cluster_label

def cluster_full_pipeline(features_all,k,points_sel,thresh=5,filter_size=(5,5),segment=True,positional_medoid=False,only_label=False):
    '''Cluster and segment multiple sets of features.'''
    
    cluster_results = cluster_sets(features_all,k=k,\
                    thresh=thresh,filter_size=filter_size,segment=segment,positional_medoid=positional_medoid)

    S = get_similarity_matrix(cluster_results)

    ensemble_label,naive_ensemble_label = cluster_similarity_matrix(S,k=k,thresh=thresh,filter_size=filter_size,segment=segment)

    if only_label==False:
        medoids_ind = find_medoid_ind(points_sel,ensemble_label,markers=points_sel,positional_medoid=True)

        features_compressed_all = get_compressed_features_multiple(features_all,ensemble_label,medoids_ind)

        MSE_all = get_MSE_multiple(features_all,features_compressed_all)

        return cluster_results,naive_ensemble_label,ensemble_label,medoids_ind,features_compressed_all,MSE_all
    elif only_label==True:
        return cluster_results,naive_ensemble_label,ensemble_label
    
def get_ground_truth(points,length=1,width=1,het_domain='circle'):
    '''Obtain ground truth for heterogeneous domains.'''
    
    truth = []
    if het_domain == 'circle':
        radius = 0.2
        cent_cir = [length/2,width/2]
        for i in range(len(points)):
            if np.sqrt((points[i,0]-cent_cir[0])**2 + (points[i,1]-cent_cir[1])**2) <= radius:
                truth.append(1)
            else:
                truth.append(0)

    elif het_domain == '4_circles':
        radius = 0.2
        c1=[length/4,width/4]
        c2=[length/4,width*3/4]
        c3=[length*3/4,width/4]
        c4=[length*3/4,width*3/4]
        for i in range(len(points)):
            if np.sqrt((points[i,0]-c1[0])**2 + (points[i,1]-c1[1])**2) <= radius:
                truth.append(1)
            elif np.sqrt((points[i,0]-c2[0])**2 + (points[i,1]-c2[1])**2) <= radius:
                truth.append(1)
            elif np.sqrt((points[i,0]-c3[0])**2 + (points[i,1]-c3[1])**2) <= radius:
                truth.append(1)
            elif np.sqrt((points[i,0]-c4[0])**2 + (points[i,1]-c4[1])**2) <= radius:
                truth.append(1)
            else:
                truth.append(0)
                
    elif het_domain == 'split':
        for i in range(len(points)):
            pos_x = points[i,0]
            if pos_x <= width/2:
                truth.append(0)
            else:
                truth.append(1)
    truth = np.array(truth)
    return truth

def get_ARI_multiple(truth,labels):
    '''Obtain ARI score for sets of labels given truth.'''
    
    if len(labels.shape) == 1:
        ARI_score = adjusted_rand_score(truth,labels)
        return ARI_score
    else:
        ARI_scores = []
        for i in range(len(labels)):
            label = labels[i]
            ARI_score = adjusted_rand_score(truth,label)
            ARI_scores.append(ARI_score)
        ARI_scores = np.array(ARI_scores)

        return ARI_scores
