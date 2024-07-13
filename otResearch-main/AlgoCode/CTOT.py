import numpy as np
from matplotlib import pyplot as plt
from algos import sinkhorn, greenkhorn, stochSinkhorn, sag, apdagd, aam, apdrcd
from util import dist



def compute(I1, I2, eps, otAlgo="sinkhorn", nb=250):
    # Turn images into matrices with one pixel per row
    X1 = im2mat(I1)
    X2 = im2mat(I2)

    rng = np.random.RandomState(123)
    # Randomly sample nb pixels from each image
    idx1 = rng.randint(X1.shape[0], size=(nb,))
    idx2 = rng.randint(X2.shape[0], size=(nb,))

    # Collect these samples into new matrices
    Xs = X1[idx1, :]
    Xt = X2[idx2, :]
    
    # Set up parameters for the Optimal Transport problem
    r, c = np.ones(nb, ) / nb, np.ones(nb,) / nb
    M = dist(Xs, Xt)
    M /= M.max()
    reg = 1e-1

    # Get coupling matrix for our OT problem
    if otAlgo == 'sinkhorn':
        P, t, o = sinkhorn(r, c, M, reg, eps)
    elif otAlgo == 'greenkhorn':
        P, t, o = greenkhorn(r, c, M, reg, eps)
    elif otAlgo == 'stochSinkhorn':
        P, t, o = stochSinkhorn(r, c, M, reg, eps)
    elif otAlgo == 'sag':
        P, t, o = sag(r, c, M, reg, eps)
    elif otAlgo == 'apdagd':
        P, t, o = apdagd(r, c, M, eps)
    elif otAlgo == 'aam':
        P, t, o = aam(r, c, M, eps)
    elif otAlgo == 'apdrcd':
        P, t, o = apdrcd(r, c, M, eps)
    else:
        print("Invalid Algo!")
        print("Try sinkhorn, greenkhorn, stochSinkhorn, sag, apdagd, aam, or apdrcd (case-sensitive):")
        otAlgo = input()
        return compute(I1, I2, otAlgo)

    # Transform Image 1 with colors of Image 2
    indices = np.arange(X1.shape[0])
    batch_ind = [indices[i:i + 128] for i in range(0, len(indices), 128)]

    transp_X1 = []
    for bi in batch_ind:
        # get the nearest neighbor in the source domain
        D0 = dist(X1[bi], Xs)
        idx = np.argmin(D0, axis=1)

        # transport the source samples
        transp = P / np.sum(P, 1)[:, None]
        transp[~ np.isfinite(transp)] = 0
        transp_Xs_ = np.dot(transp, Xt)

        # define the transported points
        transp_Xs_ = transp_Xs_[idx, :] + X1[bi] - Xs[idx, :]

        transp_X1.append(transp_Xs_)
    transp_X1 = np.concatenate(transp_X1, axis=0)

    I1t = np.clip(mat2im(np.array(transp_X1), I1.shape), 0, 1)
    
    # perform out of sample mapping
    indices = np.arange(X2.shape[0])
    batch_ind = [indices[i:i + 128] for i in range(0, len(indices), 128)]

    transp_X2 = []
    for bi in batch_ind:
        D0 = dist(X2[bi], Xt)
        idx = np.argmin(D0, axis=1)

        # transport the target samples
        transp_ = P.T / np.sum(P, 0)[:, None]
        transp_[~ np.isfinite(transp_)] = 0
        transp_X2_ = np.dot(transp_, Xs)

        # define the transported points
        transp_X2_ = transp_X2_[idx, :] + X2[bi] - Xt[idx, :]

        transp_X2.append(transp_X2_)

    transp_X2 = np.concatenate(transp_X2, axis=0)

    I2t = np.clip(mat2im(np.array(transp_X2), I2.shape), 0, 1)

    return I1t, I2t, t, o

def colTrans(eps, algorithm = 'sinkhorn'):
    """
    Setup plotting of results
    """
    plt.figure(1, figsize=(8, 4))
    plt.title(f"{algorithm} results")

    # Get the images

    I1 = plt.imread('data/ocean_day.jpg').astype(np.float64) / 256
    I2 = plt.imread('data/ocean_sunset.jpg').astype(np.float64) / 256

    # Perform the color transfer computation
    I1t, I2t, t, o = compute(I1, I2, eps, algorithm)

    # Plot resulting transfers in one figure
    plt.subplot(2, 2, 1)
    plt.imshow(I1)
    plt.axis('off')
    plt.title('Image 1')

    plt.subplot(2, 2, 2)
    plt.imshow(I1t)
    plt.axis('off')
    plt.title('Image 1 Transfer')

    plt.subplot(2, 2, 3)
    plt.imshow(I2)
    plt.axis('off')
    plt.title('Image 2')

    plt.subplot(2, 2, 4)
    plt.imshow(I2t)
    plt.axis('off')
    plt.title('Image 2 Transfer')
    plt.tight_layout()

    # Save the results to the results folder
    plt.savefig(f"results/{algorithm}Transfer{str(int(1/eps))}")
    return t, o

def im2mat(img):
    """
    Converts image of size m X n to matrix with one pixel per line (mn X 3)
    """
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    """
    Converts back a matrix to an image
    """
    return X.reshape(shape)
