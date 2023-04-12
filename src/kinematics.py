import numpy as np
from scipy import spatial, interpolate
import copy
import os

def load_pt_disp(path,pt_loc_filename,u_mat_filename,markers_len=1000):
    pt_loc = np.load(os.path.join(path,pt_loc_filename))[:markers_len,0:2]
    u_mat = np.load(os.path.join(path,u_mat_filename))[:markers_len,0:2]
    return pt_loc,u_mat

def load_multiple(path,pt_filenames,u_filenames,markers_len=1000):
    pt_loc_all=[]
    u_mat_all=[]
    for i in range(len(pt_filenames)):
        pt_file = pt_filenames[i]
        u_file = u_filenames[i]
        pt_loc,u_mat = load_pt_disp(path,pt_file,u_file,markers_len=markers_len)
        pt_loc_all.append(pt_loc)
        u_mat_all.append(u_mat)
    pt_loc_all = np.array(pt_loc_all)
    u_mat_all = np.array(u_mat_all)
    return pt_loc_all,u_mat_all

def get_nn_mat(pt_loc,pt_sel):
    # obtain distance matrix for point cloud
    D = spatial.distance_matrix(pt_sel,pt_loc)
    # sort distance matrix and obtain nearest neighbors
    # this array contains the index of the current node follow by its neighbors
    nn = np.argsort(D,axis=1)
    return nn

def get_displacement_grad(pt_loc,pt_sel,u_mat,num_neigh):

    u_mat_int=[]
    grad_u=[]
    nn_mat=get_nn_mat(pt_loc,pt_sel) # obtain nearest neighbors matrix

    for i in range(len(pt_sel)):

        cur_pt = pt_sel[i,:]
        patch_ind = nn_mat[i,0:num_neigh]
        patch = pt_loc[patch_ind,:]
        u_patch_x = u_mat[patch_ind,0]
        u_patch_y = u_mat[patch_ind,1]

        # interpolate
        tck_ux = interpolate.bisplrep(patch[:,0],patch[:,1],u_patch_x)
        tck_uy = interpolate.bisplrep(patch[:,0],patch[:,1],u_patch_y)
        du1_dx1 = interpolate.bisplev(cur_pt[0],cur_pt[1],tck_ux,dx=1,dy=0)
        du2_dx2 = interpolate.bisplev(cur_pt[0],cur_pt[1],tck_uy,dx=0,dy=1)
        du1_dx2 = interpolate.bisplev(cur_pt[0],cur_pt[1],tck_ux,dx=0,dy=1)
        du2_dx1 = interpolate.bisplev(cur_pt[0],cur_pt[1],tck_uy,dx=1,dy=0)
        grad_u.append([du1_dx1,du2_dx2,du1_dx2,du2_dx1])
        
        # interpolate u_mat for error assessment
        ux = interpolate.bisplev(cur_pt[0],cur_pt[1],tck_ux,dx=0,dy=0)
        uy = interpolate.bisplev(cur_pt[0],cur_pt[1],tck_uy,dx=0,dy=0)
        u_mat_int.append([ux,uy])

    grad_u = np.array(grad_u)
    u_mat_int = np.array(u_mat_int)

    return u_mat_int,grad_u

def get_strain(grad_u):
    grad_uT = copy.deepcopy(grad_u)
    grad_uT[:,[-2,-1]] = grad_u[:,[-1,-2]]
    strain = 0.5 * (grad_u + grad_uT)
    return strain

def get_invariants_of_tensor(tensor):
    '''
    Tensor format as an array:
    tensor[i,0]: the 11 position in a tensor for marker i
    tensor[i,1]: the 22 position in a tensor for marker i
    tensor[i,2]: the 12 position in a tensor for marker i
    tensor[i,3]: the 21 position in a tensor for marker i
    '''
    a11 = tensor[:,0]
    a22 = tensor[:,1]
    a12 = tensor[:,2]
    a21 = tensor[:,3]
    I1 = np.array(a11 + a22)
    I2 = np.array(a11*a22 - a12*a21)
    I = np.concatenate((I1.reshape(-1,1),I2.reshape(-1,1)),axis=1)

    return I

def get_F(grad_u):
    return grad_u+[1,1,0,0]

def get_C(F):
    C = [F[:,0]*F[:,0]+F[:,3]*F[:,3],
        F[:,2]*F[:,2]+F[:,1]*F[:,1],
        F[:,0]*F[:,2]+F[:,3]*F[:,1],
        F[:,2]*F[:,0]+F[:,1]*F[:,3]]
    C = np.transpose(np.array(C))
    return C

def get_b(F):
    b = [F[:,0]*F[:,0]+F[:,2]*F[:,2],
          F[:,3]*F[:,3]+F[:,1]*F[:,1],
          F[:,0]*F[:,3]+F[:,2]*F[:,1],
          F[:,3]*F[:,0]+F[:,1]*F[:,2]]
    b = np.transpose(np.array(b))
    return b

def sample_points(pt_len,L):
    points_sel = []

    # pt_img_x = np.zeros((int(np.sqrt(pt_len)), int(np.sqrt(pt_len))))
    # pt_img_y = np.zeros((int(np.sqrt(pt_len)), int(np.sqrt(pt_len))))
    pix_len = L/np.sqrt(pt_len)
    for i in range(int(np.sqrt(pt_len))):
        pos_x = pix_len*i
        for j in range(int(np.sqrt(pt_len))):
            pos_y = pix_len*j
            points_sel.append([pos_x,pos_y])
    points_sel = np.array(points_sel)
    return points_sel

def get_kinematics_with_nn(pt_loc,pt_sel,u_mat,num_neigh):

    # displacement and displacement gradients at each marker via interpolation
    u_mat_int, grad_u = get_displacement_grad(pt_loc,pt_sel,u_mat,num_neigh)

    # strain
    strain = get_strain(grad_u)

    # invariants of strain
    I_strain = get_invariants_of_tensor(strain)

    # deformation gradient
    F = get_F(grad_u)

    # invariants of F
    I_F = get_invariants_of_tensor(F)

    # right Cauchy-Green
    C = get_C(F)

    # invariants of C
    I_C = get_invariants_of_tensor(C)

    # left Cauchy-Green
    b = get_b(F)

    # invariants of b
    I_b = get_invariants_of_tensor(b)

    return u_mat_int,grad_u,strain,I_strain,F,I_F,C,I_C,b,I_b

def get_kinematics_multiple(pt_loc_all,u_mat_all,points_sel,num_neigh):
    u_mat_list = ()
    grad_u_list = ()
    strain_list = ()
    I_strain_list = ()
    F_list = ()
    I_F_list = ()
    C_list = ()
    I_C_list = ()
    b_list = ()
    I_b_list = ()

    for i in range(len(pt_loc_all)):
        pt_loc = pt_loc_all[i]
        u_mat = u_mat_all[i]
        u_mat_int,grad_u,strain,I_strain,F,I_F,C,I_C,b,I_b = get_kinematics_with_nn(pt_loc,points_sel,u_mat,num_neigh)

        # append kinematics for each file
        u_mat_list = u_mat_list + (u_mat_int,)
        grad_u_list = grad_u_list + (grad_u,)
        strain_list = strain_list + (strain,)
        I_strain_list = I_strain_list + (I_strain,)
        F_list = F_list + (F,)
        I_F_list = I_F_list + (I_F,)
        C_list = C_list + (C,)
        I_C_list = I_C_list + (I_C,)
        b_list = b_list + (b,)
        I_b_list = I_b_list + (I_b,)

    u_mat_list = np.array(u_mat_list)
    grad_u_list = np.array(grad_u_list)
    strain_list = np.array(strain_list)
    I_strain_list = np.array(I_strain_list)
    F_list = np.array(F_list)
    I_F_list = np.array(I_F_list)
    C_list = np.array(C_list)
    I_C_list = np.array(I_C_list)
    b_list = np.array(b_list)
    I_b_list = np.array(I_b_list)

    return u_mat_list,grad_u_list,strain_list,\
        I_strain_list,F_list,I_F_list,C_list,I_C_list,b_list,I_b_list