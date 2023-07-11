import numpy as np
from cluster_project import cluster
from cluster_project import kinematics as kn

def test_get_kmeans_label():
    feature = [[5,2],[0,1],[0,1],[6,1],[0,1],[5,2]]
    known1 = np.array([0,1,1,0,1,0])
    known2 = np.array([1,0,0,1,0,1])
    found = cluster.get_kmeans_label(feature,n=2).labels_
    assert np.allclose(known1,found) or np.allclose(known2,found)

def test_arr_to_img():
    arr = np.array([0,1,2,3])
    known = np.array([[1,3],[0,2]])
    found = cluster.arr_to_img(arr)
    assert np.allclose(known,found)

def test_img_to_arr():
    img = np.array([[3,6,9],[2,5,8],[1,4,7]])
    found = cluster.img_to_arr(img)
    known = np.array([1,2,3,4,5,6,7,8,9])
    assert np.allclose(known,found)

def test_label_by_position():
    img = np.array([[100,255,0],[0,1,0],[0,100,100]])
    # known = np.array([[4,6,3],[1,2,3],[1,5,5]])
    known = np.array([[1,2,3],[4,5,3],[4,6,6]])
    found = cluster.label_by_position(img)
    assert np.allclose(known,found)

def test_find_small_regions():
    img = np.array([[0,0,0,1],[0,1,1,1],[2,2,3,3],[4,4,3,3]])
    known = np.array([2,4])
    found = cluster.find_small_regions(img,thresh=3)
    assert np.allclose(known,found)

def test_set_small_regions_zero():
    img = np.array([[0,0,0,1],[0,1,1,1],[2,2,3,3],[4,4,3,3]])
    below_thresh_label = np.array([2,4])
    known_img_zero_regions = np.array([[1,1,1,2],[1,2,2,2],[0,0,4,4],[0,0,4,4]])
    known_zero_idx = np.array([(0,0),(0,1),(1,0),(1,1)],dtype=object)
    known_zero_idx = (np.array([2,2,3,3]),np.array([0,1,0,1]))
    img_zero_regions,zero_idx=cluster.set_small_regions_zero(img,below_thresh_label)
    assert np.allclose(known_img_zero_regions,img_zero_regions)
    assert np.allclose(known_zero_idx,zero_idx)

def test_median_filter():
    # fail when any corner has 2 isolated pixels
    arr = np.ones((10,10))
    arr[1,5] = 10
    arr[7,3] = 10
    filter_size = (3,3)
    known = np.ones((10,10))
    found= cluster.median_filter(arr,filter_size)
    assert np.allclose(known,found)

def test_replace_small_regions_vals():
    arr = np.ones((10,10))
    arr[1,5] = 0
    arr[7,3] = 0
    arr_filtered = np.ones((10,10))
    known = np.ones((10,10))
    found = cluster.replace_small_regions_vals(arr,arr_filtered)
    assert np.allclose(known,found)

def test_clean_tiny_clusters():
    arr1 = np.ones((10,5))*2
    arr2 = np.ones((10,5))*10
    arr3 = np.concatenate((arr1,arr2),axis=1)
    # arr3[0,0]=3
    arr3[0,1]=3
    arr3[0,2]=3
    arr3[7,8]=9
    arr3[7,9]=9
    arr3[6,8]=3
    arr3[6,9]=3
    arr3_labeled = cluster.label_by_position(arr3)
    found = cluster.clean_tiny_clusters(arr3_labeled,thresh=5,filter_size=(5,5))
    known1 = np.ones((10,5))
    known2 = np.ones((10,5))*2
    known = np.concatenate((known1,known2),axis=1)
    assert np.allclose(known,found)

def test_segment_by_position():
    # arr = np.array([2,2,2,2,2,0,0,0,0,0,\
    #                  2,2,2,2,2,0,0,0,0,0,\
    #                  1,2,2,3,2,0,1,4,4,0,\
    #                  2,2,2,4,4,0,0,0,4,0,\
    #                  2,2,2,4,4,0,0,5,4,0,\
    #                  3,3,3,3,3,1,1,1,1,1,\
    #                  3,3,3,3,3,1,1,1,1,1,\
    #                  3,3,3,3,3,1,1,1,1,1,\
    #                  3,3,3,3,3,1,1,5,1,1,\
    #                  3,3,3,3,3,1,1,5,1,1])
    # known = np.array([3,3,3,3,3,1,1,1,1,1,\
    #                  3,3,3,3,3,1,1,1,1,1,\
    #                  3,3,3,3,3,1,1,1,1,1,\
    #                  3,3,3,3,3,1,1,1,1,1,\
    #                  3,3,3,3,3,1,1,1,1,1,\
    #                  4,4,4,4,4,2,2,2,2,2,\
    #                  4,4,4,4,4,2,2,2,2,2,\
    #                  4,4,4,4,4,2,2,2,2,2,\
    #                  4,4,4,4,4,2,2,2,2,2,\
    #                  4,4,4,4,4,2,2,2,2,2])
    arr1 = np.ones((10,1))
    arr2 = np.ones((15,1))*4
    arr = np.concatenate((arr1,arr2),axis=0).flatten()
    arr[5] = arr[6] = 100
    arr[15] = arr[16] = arr[17] = 100
    known1 = np.ones((10,1))
    known2 = np.ones((15,1))*2
    known = np.concatenate((known1,known2),axis=0).flatten()
    found = cluster.segment_by_position(arr)
    assert np.allclose(known,found)

def test_obtain_ind_in_clusters():
    labels = np.array([0,0,1,1,0,2,1,1,0,2,1,1,2,2,0])
    known = np.array([np.array([0,1,4,8,14]),np.array([2,3,6,7,10,11]),np.array([5,9,12,13])],dtype=object)
    found = cluster.obtain_ind_in_clusters(labels)
    for i in range(len(known)):
        assert np.allclose(known[i],found[i])

def test_find_medoid_ind():
    # feature = np.array([[3,1],[4,2],[11,9],[5,1],[13,10],[15,11]])
    # label = np.array([100,100,10,100,10,10])
    # known = np.array([1,4])

    # single label and feature case
    feature=np.array([[3,1],[100,100],[99,99],[101,101]])
    label = np.array([1,0,0,0])
    known = np.array([0,1])
    found =  cluster.find_medoid_ind(feature,label)
    assert np.allclose(known,found)

def test_get_medoid_val_by_sorted_labels():
    feature = np.array([[0,1],[5,5],[1,2],[6,6],[9,9],[50,50]])
    labels = np.array([100,50,100,50,50,1])
    medoids_ind = np.array([0,3,5])
    known = np.array([[50,50],[6,6],[0,1]])
    found = cluster.get_medoid_val_by_sorted_labels(feature,medoids_ind,labels)
    assert np.allclose(known,found)

def test_get_compressed_features_single():
    feature = np.array([[0,1],[5,5],[1,2],[6,6],[9,9],[50,50]])
    labels = np.array([100,50,100,50,50,1])
    medoids_ind = np.array([0,3,5])
    known = np.array([[0,1],[6,6],[0,1],[6,6],[6,6],[50,50]])
    found = cluster.get_compressed_features_single(feature,labels,medoids_ind)
    assert np.allclose(known,found)

def test_get_compressed_features_multiple():
    feature1 = np.array([[0,0],[1,1],[2,2],[10,10],[11,11],[12,12]])
    feature2 = np.array([[1,1],[2,2],[3,3],[11,11],[12,12],[13,13]])
    feature = np.array([feature1,feature2])
    labels = np.array([1,1,1,2,2,2])
    medoids_ind = np.array([1,4])
    known = np.array([[[1,1],[1,1],[1,1],[11,11],[11,11],[11,11]],
                      [[2,2],[2,2],[2,2],[12,12],[12,12],[12,12]]])
    found = cluster.get_compressed_features_multiple(feature,labels,medoids_ind)
    assert np.allclose(known,found)

def test_MSE():
    arr1 = np.array([[0,1],[1,2],[3,4]])
    arr2 = np.array([[0,1],[1,1],[2,5]])
    known = np.array([1/3,2/3])
    found = cluster.MSE(arr1,arr2)
    assert np.allclose(known,found)

def test_get_MSE_multiple():
    feature1 = np.array([[0,0],[1,1],[2,2],[10,10],[11,11],[12,12]])
    feature2 = np.array([[1,1],[2,2],[3,3],[11,11],[12,12],[13,14]])
    features = np.array([feature1,feature2])
    features_compressed_all = np.array([[[1,1],[1,1],[1,1],[11,11],[11,11],[11,11]],
                                        [[2,2],[2,2],[2,2],[12,12],[12,12],[12,12]]])
    known = np.array([2/3,11/12])
    found = cluster.get_MSE_multiple(features,features_compressed_all)
    assert np.allclose(known,found)
    

def test_cluster_single_set():
    k=2
    feature = np.array([[0,0],[1,1],[2,2],[3,3],[100,100],[0,0],[1,1],[2,2],[100,100],[0,0],[100,100],[0,0],[1,1],[2,2],[100,100],[0,0]])
    # known = np.array([[1,1,2],[2,2,2],[2,3,3]])
    known = np.array([1,1,1,1,3,1,1,1,3,1,2,1,1,1,2,1])
    found_label,_,_,_,_ = cluster.cluster_single_set(feature,k,thresh=1,filter_size=(3,3),segment=True)
    assert np.allclose(known,found_label)

def test_cluster_sets():
    k=2
    feature1 = np.random.rand(9,4)
    feature2 = np.random.rand(9,4)
    feature3 = np.random.rand(9,4)
    features_all = np.array([feature1,feature2,feature3],dtype=object)
    found = cluster.cluster_sets(features_all,k=k,thresh=5,filter_size=(5,5),segment=True,positional_medoid=False)
    pass

# still hasn't accounted for equivalent indicator matrices
# function still works properly
def test_get_indicator_matrix():
    cluster_results = np.array([[0,0,10,20],[22,0,0,11],[0,1,1,1]])
    # known1 = np.array([[0,0,1,0,1,0,0,1],[0,0,1,0,0,1,1,0],[1,0,0,0,0,1,1,0],[0,1,0,1,0,0,1,0]])
    # known2 = np.array([[1,0,0,1,0,0,1,0],[1,0,0,0,1,0,0,1],[0,1,0,0,1,0,0,1],[0,0,1,0,0,1,0,1]])
    known3 = np.array([[1,0,0,0,0,1,1,0],[1,0,0,1,0,0,0,1],[0,1,0,1,0,0,0,1],[0,0,1,0,1,0,0,1]])
    found,_ = cluster.get_indicator_matrix(cluster_results)
    assert np.allclose(known3,found)


def test_get_similarity_matrix():
    cluster_results = np.array([[0,0,10,20],[22,0,0,11],[0,1,1,1]])
    H = np.array([[1,0,0,0,0,1,1,0],[1,0,0,1,0,0,0,1],[0,1,0,1,0,0,0,1],[0,0,1,0,1,0,0,1]])
    known = (1/3)*np.matmul(H,H.transpose())
    found = cluster.get_similarity_matrix(cluster_results)
    assert np.allclose(known,found)

def test_cluster_similarity_matrix():
    label1 = np.array([1,1,1,1,2,1,1,1,2,1,2,1,1,1,2,1])
    label2 = np.array([1,1,1,1,2,1,1,1,2,1,2,1,1,1,2,1])
    labels = np.array([label1,label2])
    S = cluster.get_similarity_matrix(labels)
    found,_ = cluster.cluster_similarity_matrix(S,2,thresh=1,filter_size=(3,3),segment=True)
    known = np.array([1,1,1,1,3,1,1,1,3,1,2,1,1,1,2,1])
    assert np.allclose(known,found)

def test_cluster_full_pipelines():
    feature1 = np.array([[0,0],[1,1],[2,2],[10,10],[11,11],[12,12],[20,20],[21,21],[21,21]])
    feature2 = np.array([[1,1],[2,2],[3,3],[11,11],[12,12],[13,13],[20,20],[21,21],[22,22]])
    feature_all = np.array([feature1,feature2])
    points_sel = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
    cluster_results,naive_ensemble_label,ensemble_label,medoids_ind,features_compressed_all,MSE_all = \
        cluster.cluster_full_pipeline(feature_all,3,points_sel)
    # pass
    assert len(np.unique(ensemble_label)) == len(medoids_ind)

def test_get_ground_truth():
    points_sel = kn.sample_points(25,1)
    known = np.array([0,0,0,0,0,\
                      0,0,0,0,0,\
                      0,0,1,1,0,\
                      0,0,1,1,0,\
                      0,0,0,0,0])
    length = width = 1
    found = cluster.get_ground_truth(points_sel,length,width,het_domain='circle_inclusion')
    path = 'tutorials/files/example_data/Cahn_Hilliard_Image12_NH/'
    found2 = cluster.get_ground_truth(points_sel,length,width,het_domain='cahn_hilliard_image12',path=path)
    assert np.allclose(known,found)
    assert isinstance(found2, np.ndarray)

def test_get_ARI_multiple():
    truth = np.array([0,0,0,0,1,1])
    labels = np.array([[0,0,0,0,1,1],[1,1,1,1,0,0]])
    known = np.array([1.0,1.0])
    found = cluster.get_ARI_multiple(truth,labels)
    assert np.allclose(known,found)

def test_segment_large_clusters():
    arr1 = np.ones((10,5)) * 10
    arr2 = np.zeros((10,5))
    label_img = np.concatenate((arr1,arr2),axis=1)
    # label_img[0,0]=label_img[0,1]=label_img[0,2]=label_img[1,0]=label_img[2,0]=5
    grid_markers = kn.sample_points(100,1)

    # there are multiple solutions here
    arr3 = np.ones((5,5))*1
    arr4 = np.ones((5,5))*3
    arr5 = np.ones((5,5))*2
    arr6 = np.ones((5,5))*4
    arr34 = np.concatenate((arr3,arr4),axis=1)
    arr56 = np.concatenate((arr5,arr6),axis=1)
    known1 = np.concatenate((arr34,arr56),axis=0).astype(int)
    
    arr3 = np.ones((10,3))*1
    arr4 = np.ones((10,2))*2
    arr5 = np.ones((5,5))*3
    arr6 = np.ones((5,5))*4
    arr34 = np.concatenate((arr3,arr4),axis=1)
    arr56 = np.concatenate((arr5,arr6),axis=0)
    known2 = np.concatenate((arr34,arr56),axis=1).astype(int)

    arr3 = np.ones((10,2))*1
    arr4 = np.ones((10,3))*2
    arr5 = np.ones((5,5))*3
    arr6 = np.ones((5,5))*4
    arr34 = np.concatenate((arr3,arr4),axis=1)
    arr56 = np.concatenate((arr5,arr6),axis=0)
    known3 = np.concatenate((arr34,arr56),axis=1).astype(int)

    arr3 = np.ones((10,2))*1
    arr4 = np.ones((10,3))*2
    arr5 = np.ones((5,5))*3
    arr6 = np.ones((5,5))*4
    arr34 = np.concatenate((arr3,arr4),axis=1)
    arr56 = np.concatenate((arr5,arr6),axis=0)
    known4 = np.concatenate((arr34,arr56),axis=1).astype(int)

    found = cluster.segment_large_clusters(label_img,grid_markers,25)
    assert np.allclose(known1,found) or np.allclose(known2,found) or np.allclose(known3,found) or np.allclose(known4,found)