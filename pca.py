import os
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


# Gets the path of a file stored in a folder called data
# and then returns the filepath
def getPath(myFile):
    current_directory = os.path.dirname(__file__)
    filepath = os.path.join(current_directory)
    filepath += "\\data\\" + myFile
    return filepath


def saveEigs(e_scaled):
    eigs = open(getPath("eigs.txt"), 'w')
    eigs.write(str(e_scaled))
    eigs.close()


def z_norm(X):
    """ Normalize the dataset X by Z-score: subtract the mean and divide by the standard deviation.

	INPUT:
	X -- (n,m) ndarray of raw data, assumed to contain 1 row per sample and 1 column per feature.

	OUTPUT:
	X_norm -- (n,m) ndarray of Z-score normalized data
	X_mean -- (m,) ndarray of the means of the features (columns) of the raw dataset X
	X_std -- (m,) ndarray of the standard deviations of the features (columns) of the raw dataset X
	"""

    X_mean = np.mean(X, axis=0)
    X_std = np.mean(X, axis=0)
    X_norm = (X - X_mean) / X_std

    return X_norm, X_mean, X_std


# Takes in a comma-separated filepath and creates a numpy array
# Assumes:
# 1) the data is separated by commas
# 2) the first line of the header is the column labels
def arr_csv(filename):
    filepath = getPath(filename)
    arr = np.genfromtxt(filepath, skip_header=1, delimiter=',')
    return arr


# Takes in a file stored in a .npy file and loads the file.
def arr_npy(filename):
    filepath = getPath(filename)
    arr = np.load(filepath)
    return arr


def test_npy():
    lfwcrop = arr_npy("lfwcrop.npy")
    first_face = lfwcrop[0, :, :]
    # first_name = lfw_names[0]
    plt.imshow(first_face)
    plt.show()
    mean_face = np.mean(lfwcrop, axis=0)
    plt.imshow(mean_face, cmap='bone')
    plt.show()
    mean_face = mean_face.reshape(13231, 4096)
    X = lfwcrop.reshape(13231, 4096)
    X_diff = X - mean_face
    C = np.cov(X_diff, rowvar=False)
    e, P = np.linalg.eig(C)
    print(P)


def read_lfwcrop():
    ''' Return an ndarray of LFW Crop's image data and a list of the corresponding names. '''

    # Read the images in from the lfwcrop.npy file
    lfw_faces = arr_npy("lfwcrop.npy")
    # Read the name of each image in from the lfwcrop_ids.txt file
    names_filepath = getPath("lfwcrop_ids.txt")
    lfw_names = np.loadtxt(names_filepath, dtype=str, delimiter="\n")

    return lfw_faces, lfw_names


def plot_face(iamge, title, ax=None):
    '''Given an image in a 2D ndarray and the desired figure title, visualizes the image as a heatmap.
	Optional ax parameter: can provide a particular axis object in which to display the image. Returns 
	nothing (None).'''
    if (ax != None):
        fig, ax = plt.subplots()

    ax.imshow(iamge)
    ax.set_title(title)
    plt.show()


def main():

    ''' Draw some faces from the LFW Crop dataset. Draw the mean face. Try out PCA-Cov and PCA-SVD.
	Try out a reconstruction.'''

    # Read in the dataset, including all images and the names of the associated people
    X, lfw_names = read_lfwcrop()
    n = X.shape[0]
    m = X.shape[1] * X.shape[2]
    print("faces:", X.shape)
    print("names:", len(lfw_names))
    print("features:", m)

    # Visualize the first face
    first_face = X[0, :, :]
    first_name = lfw_names[0]
    plt.figure()
    plt.imshow(first_face, cmap="bone")
    plt.title(first_name)

    """
    # Visualize a random face
    rand_idx = np.random.randint(n)
    rand_face = X[rand_idx, :, :]
    rand_name = lfw_names[rand_idx]
    plt.figure()
    plt.imshow(rand_face, cmap="bone")
    plt.title(rand_name)

    # Visualize the mean face
    mean_face = np.mean(X, axis=0)
    mean_name = "Mean Face"
    plt.figure()
    plt.imshow(mean_face, cmap="bone")
    plt.title(mean_name)

    # PCA-Cov?
    n_tiny = 100
    X = X[0:n_tiny, :, :]
    X = X.reshape((n_tiny, m))
    print(X.shape)
    X_diff = X - mean_face.reshape((1, m))
    C = np.cov(X_diff, rowvar=False)
    e, P = np.linalg.eig(C)
    print(P)
    """
    n_tiny = 100
    X = X[0:n_tiny, :, :]
    # X = X.reshape((n_tiny, m))
    print(X.shape)

    mean_face = np.mean(X, axis=0)
    n = 0
    # PCA-SVD?
    random_face = X[0, :, :]

    plt.figure()
    plt.imshow(random_face, cmap="bone")
    plt.title(first_name)
    print(random_face.shape)
    first_name = lfw_names[n]
    X = X.reshape(100, 4096)
    print(X.shape)
    X_diff = X - mean_face.reshape((1, m))
    X_norm, X_mean, X_std = z_norm(X)
    print(X_mean.shape, X_std.shape, X_norm.shape)
    U, W, Vt = np.linalg.svd(X_norm)
    eigvals = W ** 2
    P = Vt.T
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    e_scaled = eigvals / np.sum(eigvals)
    # saveEigs(e_scaled)
    d = 1
    P = P[:, :d]
    print(P, '\n', eigvals)
    Y = X_norm @ P
    print(Y.shape)
    percent_retained = 0
    start = 9

    percent_retained = e_scaled[0]
    for d in range(1, 20):
        percent_retained += e_scaled[d]
        Y_proj = Y[0:d]
        print(Y_proj.shape, P.T.shape)
        X_rec = (Y_proj @ P.T) * X_std + X_mean
        print(X_rec.shape, X.shape)

        X_rec = X_rec.reshape(X_rec.shape[0], 64, 64)
        print(X_rec.shape)
        rand_id = n
        print(rand_id)
        rand_face = X_rec[rand_id, :]
        rand_name = lfw_names[rand_id]
        plt.figure()
        norm = plt.Normalize(vmin=rand_face.min(), vmax=rand_face.max())
        cmap = plt.cm.bone
        face = cmap(norm(rand_face))
        name = "aaron" + str(np.around(percent_retained * 100, decimals=1))
        name += ".png"
        plt.title(rand_name + " Reconstructed: " + str(percent_retained))
        plt.imsave(getPath(name), face)
        # plt.show()
        print("Percent retained: ", percent_retained)
    Y_proj = Y[:]
    print(Y_proj.shape, P.T.shape)
    X_rec = (Y_proj @ P.T) * X_std + X_mean
    print(X_rec.shape, X.shape)
    np.save(getPath('lfw_compressed'), Y_proj)
    X_rec = X_rec.reshape(X_rec.shape[0], 64, 64)
    rand_id = np.random.randint(X_rec.shape[0])
    print(rand_id)
    rand_face = X_rec[rand_id, :]
    rand_name = lfw_names[1]
    norm = plt.Normalize(vmin=rand_face.min(), vmax=rand_face.max())
    cmap = plt.cm.bone
    face = cmap(norm(rand_face))
    plt.imsave
    plt.figure()
    plt.imshow(rand_face, cmap="bone")
    plt.title(rand_name + " Reconstructed: " + str(100.0))
    plt.show()
    name = "aaron" + str(100.0)
    name += ".png"
    plt.imsave(getPath(name), face)
    print("Percent retained: ", 100)


def scree_plot(eigenvals):
    """Visualize information retention per eigenvector.
	
	INPUT:	
	eigenvals -- (d,) ndarray of scaled eigenvalues.
	
	OUTPUT:
	info_retention -- (d,) ndarray of accumulated information retained by multiple eigenvectors.  """

    # Visaulize individual information retention per eigenvector (eigenvalues)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(eigenvals, '-o', linewidth=2, markersize=5, markerfacecolor="w")
    ax[0].set_ylim([-0.1, 1.1])
    ax[0].set_title("Information retained by individual PCs")
    ax[0].grid(True)

    # Visualize accumulated information retained by multiple eigenvectors
    info_retention = np.cumsum(eigenvals)
    ax[1].plot(info_retention, '-o', linewidth=2, markersize=5, markerfacecolor="w")
    ax[1].set_ylim([-0.1, 1.1])
    ax[1].set_title("Cumulative information retained by all PCs")
    ax[1].grid(True)

    plt.pause(0.001)

    return info_retention


def pc_heatmap(P, info_retention):
    ''' Visualize principal components (eigenvectors) as a heatmap.
	
	INPUT:
	P -- (m,m) ndarray of principal components (eigenvectors)
	info_retention -- (m,) ndarray of accumulated scaled eigenvectors: the % info retained by all PCs
	
	OUTPUT: 
	None
	'''
    plt.figure()
    plt.imshow(P)
    plt.colorbar()
    plt.pause(0.001)
    pass


def read_file(filename):
    ''' Read in the data from a file.

	INPUT:
	filename -- string representing the name of a file in the "../data/" directory

	OUTPUT:
	data -- (n,m) ndarray of data from the specified file, assuming 1 row per sample and 1 column per feature
	headers -- list of length m representing the name of each feature (column)
	'''

    # Windows is kind of a jerk about filepaths. My relative filepath didn't
    # work until I joined it with the current directory's absolute path.
    filepath = getPath(filename)

    # Read headers from the 1st row with plain vanilla Python file handling (without Numpy)
    in_file = open(filepath)
    headers = in_file.readline().split(",")
    in_file.close()

    # Read iris's data in, skipping the metadata in 1st row
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)

    return data, headers


def pca_cov(X):
    """Perform Principal Components Analysis (PCA) using the covariance matrix to identify principal components
	(eigenvectors) and their scaled eigenvalues (which measure how much information each PC represents).

	INPUT:
	X -- (n,m) ndarray representing the dataset (observations), assuming one datum per row and one column per feature. 
	Must already be centered, so that the mean is at zero. Usually Z-score normalized. 

	OUTPUT:
	Y -- (n,m) ndarray representing rotated dataset (Y), 
	P -- (m,m) ndarray representing principal components (columns of P), a.k.a. eigenvectors
	e_scaled -- (m,) ndarray of scaled eigenvalues, which measure info retained along each corresponding PC """

    # Pull principal components and eigenvalues from covariance matrix
    C = np.cov(X, rowvar=False)
    eigvals, P = np.linalg.eig(C)
    print("Covariance")
    print(C)
    print("Eigenvalues: ")
    print(eigvals)
    print("PC")
    print(P)
    # Sort principal components in order of descending eigenvalues
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    P = P[:, order]
    # print("Order: " + order)
    # Scale eigenvalues to calculate the percent info retained along each PC
    e_scaled = eigvals / np.sum(eigvals)
    print("scaled eigenvalues")
    print(e_scaled)

    # Rotate data onto the principal components
    Y = X @ P
    print("Original")
    print(X[:5, :])
    print("Rotated")
    print(Y[:5, :])
    return (Y, P, e_scaled)


def pca_analysis_optdigits(filename="optdigits.tra", class_col=63):
    filepath = getPath(filename)
    X = np.genfromtxt(filepath, delimiter=',')
    print(X.shape)
    # Remove the class label from the dataset so that it doesn't prevent us from training a classifier in the future
    species = X[:, class_col]
    k = X.shape[0]
    m = X.shape[1]
    keepers = list(range(m))
    keepers.pop(class_col)
    X_input = X[:, keepers]

    # Visualize raw data
    plt.figure()
    plt.plot(X_input[:, 0], X_input[:, 1], 'ob', alpha=0.5, markeredgecolor='w')
    plt.title("Raw data")
    plt.tight_layout()
    plt.pause(0.1)

    # PCA Cov
    X, headers = read_file(filename)

    # Remove the class label from the dataset so that it doesn't prevent us from training a classifier in the future
    species = X[:, class_col]
    m = X.shape[1]
    keepers = list(range(m))
    keepers.pop(class_col)
    X_input = X[:, keepers]

    # Sanity check
    print("\nOriginal headers:\n\n", headers, "\n")
    print("\nOriginal dataset:\n\n", X[:5, :], "\n")
    print("\nWithout class col:\n\n", X_input[:5, :], "\n")

    # Visualize raw data
    plt.figure()
    plt.plot(X_input[:, 0], X_input[:, 1], 'ob', alpha=0.5, markeredgecolor='w')
    plt.title("Raw data")
    plt.tight_layout()
    plt.pause(0.1)

    # Normalize features by Z-score (so that features' units don't dominate PCs), and apply PCA
    X_norm, X_mean, X_std = z_norm(X_input)
    Y, P, e_scaled = pca_cov(X_norm)

    # Sanity check: Print PCs and eigenvalues in the terminal
    print("Eigenvectors (each column is a PC): \n\n", P, "\n")
    print("\nScaled eigenvalues: \t", e_scaled, "\n")
    saveEigs(e_scaled)
    # Visualize PCs with heatmap and cree plot
    info_retention = scree_plot(e_scaled)
    pc_heatmap(P, info_retention)
    # Visualize a 1D PCA Projection
    plt.figure()
    plt.plot(Y[:, 0], "om", markeredgecolor='w')
    percent_retained = e_scaled[0]
    plt.xlabel("PC0")
    plt.title("1D PCA Projection: " + str(np.around(percent_retained, decimals=2)))
    plt.tight_layout()
    plt.show()

    # Visualize a 2D PCA Projection
    percent_retained += e_scaled[1]
    plt.figure()
    plt.plot(Y[:, 0], Y[:, 1], "om", markeredgecolor='w')
    plt.xlabel("PC0")
    plt.ylabel("PC1")
    plt.title("PCA projection 2D" + str(np.around(percent_retained, decimals=2)))
    plt.tight_layout()
    plt.show()



def pca_analysis(filename="iris.data", class_col=-1):
    """ Apply PCA to the specified dataset."""

    X, headers = read_file(filename)

    # Remove the class label from the dataset so that it doesn't prevent us from training a classifier in the future
    species = X[:, class_col]
    m = X.shape[1]
    keepers = list(range(m))
    keepers.pop(class_col)
    X_input = X[:, keepers]

    # Sanity check
    print("\nOriginal headers:\n\n", headers, "\n")
    print("\nOriginal dataset:\n\n", X[:5, :], "\n")
    print("\nWithout class col:\n\n", X_input[:5, :], "\n")

    # Visualize raw data
    plt.figure()
    plt.plot(X_input[:, 0], X_input[:, 1], 'ob', alpha=0.5, markeredgecolor='w')
    plt.title("Raw data")
    plt.tight_layout()
    plt.pause(0.1)

    # Normalize features by Z-score (so that features' units don't dominate PCs), and apply PCA
    X_norm, X_mean, X_std = z_norm(X_input)
    Y, P, e_scaled = pca_cov(X_norm)

    # Sanity check: Print PCs and eigenvalues in the terminal
    print("Eigenvectors (each column is a PC): \n\n", P, "\n")
    print("\nScaled eigenvalues: \t", e_scaled, "\n")
    saveEigs(e_scaled)
    # Visualize PCs with heatmap and cree plot
    info_retention = scree_plot(e_scaled)
    pc_heatmap(P, info_retention)
    # Visualize a 1D PCA Projection
    plt.figure()
    plt.plot(Y[:, 0], "om", markeredgecolor='w')
    percent_retained = e_scaled[0]
    plt.xlabel("PC0")
    plt.title("1D PCA Projection: " + str(np.around(percent_retained, decimals=2)))
    plt.tight_layout()
    plt.show()

    # Visualize a 2D PCA Projection
    percent_retained += e_scaled[1]
    plt.figure()
    plt.plot(Y[:, 0], Y[:, 1], "om", markeredgecolor='w')
    plt.xlabel("PC0")
    plt.ylabel("PC1")
    plt.title("PCA projection 2D" + str(np.around(percent_retained, decimals=2)))
    plt.tight_layout()
    plt.show()
    sleep(0.5)


if __name__ == "__main__":
    # pca_analysis_optdigits()
    main()
    # pca_analysis("iris.data", class_col=4)
    # pca_analysis_optdigits()
