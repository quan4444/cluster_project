import matplotlib.pyplot as plt
import numpy as np
from cluster_project.cluster import arr_to_img

# untested ----------------------------------------
def plot_og_compressed_img(cat_flat,cat_flat_compressed,k,mse):
    row=2
    col=2
    name=['original I1','original I2','compressed I1','compressed I2']
    fig,ax = plt.subplots(row,col,figsize=(15,6))
    fig.suptitle(f'k={k}, mse={mse}',fontsize=20)
    count=0
    for ii in range(row):
        for jj in range(col):
            if ii==0:
                cat_flat_im = arr_to_img(cat_flat[:,jj])
                im = ax[ii,jj].imshow(cat_flat_im,cmap='Greys') # ,vmin=0,vmax=255
            else:
                cat_flat_compressed_im = arr_to_img(cat_flat_compressed[:,jj])
                im = ax[ii,jj].imshow(cat_flat_compressed_im,cmap='Greys')
            cbar = fig.colorbar(im, ax=ax[ii, jj])
            ax[ii,jj].set_title(name[count],fontsize=15)
            ax[ii,jj].get_xaxis().set_visible(False)
            ax[ii,jj].get_yaxis().set_visible(False)
            ax[ii,jj].axis('equal')
            ax[ii,jj].axis('off')
            count+=1

    fig.tight_layout()

def plot_cluster(labels,points_sel,title_extra=''):

    fig,ax = plt.subplots(figsize=(4,4))
    cluster=[]
    for unique_label in np.unique(labels):
        cluster.append(np.where(labels==unique_label))

    for i in range(len(cluster)):
        ax.scatter(points_sel[cluster[i][0],0],points_sel[cluster[i][0],1],s=5)

    ax.set_title(f'cluster result'+title_extra,size=15)
    ax.axis('equal')

def plot_centroids_on_clusters(centroids_ind,points_sel,labels):

    fig,ax = plt.subplots(figsize=(4,4))
    cluster=[]
    for unique_label in np.unique(labels):
        cluster.append(np.where(labels==unique_label))
        
    # colors_dict = ilikecolors.CSS4_COLORS
    # color_names = np.array(list(colors_dict.keys()))
    
    cent_ind = np.array(centroids_ind)
    for i in range(len(cluster)):
#         ax.scatter(points_sel[cluster[i][0],0],points_sel[cluster[i][0],1],s=5,c=color_names[2*i])
#         ax.scatter(points_sel[cent_ind[i],0],points_sel[cent_ind[i],1],s=40,c=color_names[2*i+1])
        ax.scatter(points_sel[cluster[i][0],0],points_sel[cluster[i][0],1],s=5)
    ax.scatter(points_sel[cent_ind,0],points_sel[cent_ind,1],s=40,c='red')
    
    
    ax.set_title(f'Centroids (big dots)',size=15)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_MSE(x_val,MSE,plot_type='k'):
    fig,ax = plt.subplots(figsize=(8,6))

    if plot_type =='k':
        ax.plot(x_val[:len(MSE)],MSE)
        ax.set_title('MSE vs. k',fontsize=20)
        ax.set(xlabel='k', ylabel='MSE')
        ax.tick_params(axis='both', which='major', labelsize=12)
    elif plot_type == 'num_sensors':
        ax.scatter(x_val[:len(MSE)],MSE)
        ax.set_title('MSE vs. num sensors',fontsize=20)
        ax.set(xlabel='num sensors', ylabel='MSE')
        ax.tick_params(axis='both', which='major', labelsize=12)

    fig.tight_layout()

def plot_MSE_multiple(x_ax,MSE_,disp_type,big_title='MSE vs. k',x_axis_label='k',scatter_plot=False):

    row = 1
    col = MSE_.shape[1]

    fig,ax = plt.subplots(row,col,figsize=(25,6),sharey=True)
    fig.suptitle(big_title,fontsize=25)

    count=0
    for i in range(col):
        if scatter_plot==False:
            ax[i].plot(x_ax[:len(MSE_)],MSE_[:,count])
        elif scatter_plot==True:
            ax[i].scatter(x_ax[:len(MSE_)],MSE_[:,count])

        ax[i].set_title(disp_type[count],fontsize=15)
        ax[i].set(xlabel=x_axis_label, ylabel='MSE')
        ax[i].tick_params(axis='both', which='major', labelsize=12)

        count+=1
    fig.tight_layout()

def plot_cluster_by_bcs(disp_type,cluster_results,points_sel,big_title=None):
    row=1
    col=len(disp_type)
    fig, axs = plt.subplots(row, col,figsize=(12,2.5))
    fig.suptitle(big_title,fontsize=18)
    count=0

    for ii in range(row):
        for jj in range(col):

            labels = cluster_results[count]

            cluster=[]
            for unique_label in np.unique(labels):
                cluster.append(np.where(labels==unique_label))

            for i in range(len(cluster)):
                axs[jj].scatter(points_sel[cluster[i][0],0],points_sel[cluster[i][0],1],s=5)

            axs[jj].set_title(disp_type[count],fontsize=15)
            axs[jj].get_xaxis().set_visible(False)
            axs[jj].get_yaxis().set_visible(False)
            axs[jj].axis('equal')
            axs[jj].axis('off')
            count=count+1
    fig.tight_layout()

def plot_cluster_and_ARI_by_bcs(disp_type,cluster_results,points_sel,cluster_ARIs,big_title=None):
    row=1
    col=len(cluster_results)
    fig, axs = plt.subplots(row, col,figsize=(12,2.5))
    fig.suptitle(big_title,fontsize=18)
    count=0

    for ii in range(row):
        for jj in range(col):

            labels = cluster_results[count]

            cluster=[]
            for unique_label in np.unique(labels):
                cluster.append(np.where(labels==unique_label))

            for i in range(len(cluster)):
                axs[jj].scatter(points_sel[cluster[i][0],0],points_sel[cluster[i][0],1],s=5)

            axs[jj].set_title(f'{disp_type[count]}\nARI = {np.around(cluster_ARIs[count],3)}',fontsize=15)
            axs[jj].get_xaxis().set_visible(False)
            axs[jj].get_yaxis().set_visible(False)
            axs[jj].axis('equal')
            axs[jj].axis('off')
            count=count+1
    fig.tight_layout()

def plot_cluster_ARI(labels,points_sel,ARI,title_extra=''):

    fig,ax = plt.subplots(figsize=(4,4))
    cluster=[]
    for unique_label in np.unique(labels):
        cluster.append(np.where(labels==unique_label))

    for i in range(len(cluster)):
        ax.scatter(points_sel[cluster[i][0],0],points_sel[cluster[i][0],1],s=5)

    ax.set_title(title_extra+f'\nARI = {np.around(ARI,3)}',size=15)
    ax.axis('equal')