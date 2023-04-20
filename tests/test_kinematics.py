import numpy as np
import os
from cluster_project import kinematics as kn
from sklearn.model_selection import train_test_split
import pathlib
pathlib.Path().resolve()

def test_load_pt_disp():
    disp_path = 'tests/files/test_data/'
    pt_loc_filename = 'pt_homog_uni_y_disp0.4.npy'
    u_mat_filename = 'disp_homog_uni_y_disp0.4.npy'
    pt_loc,u_mat = kn.load_pt_disp(disp_path,pt_loc_filename,u_mat_filename)
    assert pt_loc.shape == (1000,2)
    assert u_mat.shape == (1000,2)

def test_load_multiple():
    disp_path = 'tests/files/test_data/'
    pt_loc_filenames = np.array(['pt_homog_uni_y_disp0.4.npy','pt_homog_uni_y_disp0.4.npy'])
    u_mat_filenames = np.array(['disp_homog_uni_y_disp0.4.npy','disp_homog_uni_y_disp0.4.npy'])
    pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_filenames,u_mat_filenames)
    assert pt_loc_all.shape == (2,1000,2)
    assert u_mat_all.shape == (2,1000,2)

def test_get_nn_mat():
    arr1 = np.array([[0,1,0],[2,1,1],[3,3,3],[4,4,4]])
    arr2 = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
    known = np.array([[0,1,2,3],[1,0,2,3],[1,2,0,3],[2,3,1,0]])
    found = kn.get_nn_mat(arr1,arr2)
    assert np.allclose(known,found)

def test_get_displacement_grad():

    disp_path = 'tests/files/test_data/'
    # points locations
    pt_loc = np.load(os.path.join(disp_path,'pt_homog_uni_y_disp0.4.npy'))[:,0:2]
    # displacement values
    u_mat = np.load(os.path.join(disp_path,'disp_homog_uni_y_disp0.4.npy'))[:,0:2]

    # split train vali
    # known is the vali set
    pt_loc_train, pt_loc_test, u_train, u_test = train_test_split(pt_loc, u_mat, test_size=100, random_state=0)
    known = u_test

    # interpolate at vali points location
    # this is found
    num_neigh = 16
    u_mat_pred,_,_,_,_,_,_,_,_,_ = kn.get_kinematics_with_nn(pt_loc_train,pt_loc_test,u_train,num_neigh)
    found = u_mat_pred

    assert np.allclose(known,found,atol=1e-3)


def test_get_strain():
    grad_u = np.array([[0.8,0.5,0.1,0.2],[0.6,0.8,0.2,0.4]])
    known = np.array([[0.8,0.5,0.15,0.15],[0.6,0.8,0.3,0.3]])
    found = kn.get_strain(grad_u)
    assert np.allclose(known,found)

def test_get_invariants_of_tensor():
    tensor = np.array([[0.8,0.5,0.2,0.3],[0.4,1,0.1,0.2]])
    known = np.array([[1.3,0.34],[1.4,0.38]])
    found = kn.get_invariants_of_tensor(tensor)
    assert np.allclose(known,found)

def test_get_F():
    grad_u = np.array([[0.8,0.5,0.1,0.2],[0.6,0.8,0.2,0.4]])
    known = np.array([[1.8,1.5,0.1,0.2],[1.6,1.8,0.2,0.4]])
    found = kn.get_F(grad_u)
    assert np.allclose(known,found)

def test_get_C():
    F = np.array([[1.8,1.5,0.1,0.2],[1.6,1.8,0.2,0.4]])
    known = np.array([[3.28,2.26,0.48,0.48],[2.72,3.28,1.04,1.04]])
    found = kn.get_C(F)
    assert np.allclose(known,found)

def test_get_b():
    F = np.array([[1.8,1.5,0.1,0.2],[1.6,1.8,0.2,0.4]])
    known = np.array([[3.25,2.29,0.51,0.51],[2.6,3.4,1,1]])
    found = kn.get_b(F)
    assert np.allclose(known,found)

def test_sample_points():
    pt_len = 4
    L=1
    known = np.array([[0,0],[0,0.5],[0.5,0],[0.5,0.5]])
    found = kn.sample_points(pt_len,L)
    assert np.allclose(known,found)

def test_get_kinematics_with_nn():
    pt_loc = np.random.rand(100,2)
    pt_sel = np.random.rand(100,2)
    u_mat = np.random.rand(100,4)
    num_neigh = 20
    found = kn.get_kinematics_with_nn(pt_loc,pt_sel,u_mat,num_neigh)
    pass

def test_get_kinematics_multiple():
    disp_path = '/home/quan/phd/soft_square_stiff_circle/output_disp'
    pt_loc_filenames = np.array(['pt_homog_uni_y_disp0.4.npy','pt_homog_uni_y_disp0.4.npy'])
    u_mat_filenames = np.array(['disp_homog_uni_y_disp0.4.npy','disp_homog_uni_y_disp0.4.npy'])
    pt_loc_all,u_mat_all = kn.load_multiple(disp_path,pt_loc_filenames,u_mat_filenames)

    pt_len = 8000
    points_sel = kn.sample_points(pt_len,L=1)
    num_neigh = 40
    u_mat_list,grad_u_list,strain_list,I_strain_list,\
        F_list,I_F_list,C_list,I_C_list,b_list,I_b_list = \
            kn.get_kinematics_multiple(pt_loc_all,u_mat_all,points_sel,num_neigh)

    assert u_mat_list.shape == (2,7921,2)
    assert grad_u_list.shape == (2,7921,4)
    assert strain_list.shape == (2,7921,4)
    assert I_strain_list.shape == (2,7921,2)
    assert F_list.shape == (2,7921,4)
    assert I_F_list.shape == (2,7921,2)
    assert C_list.shape == (2,7921,4)
    assert I_C_list.shape == (2,7921,2)
    assert b_list.shape == (2,7921,4)
    assert I_b_list.shape == (2,7921,2)