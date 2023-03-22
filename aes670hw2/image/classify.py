
import numpy as np
from . import enhance

def minimum_distance(X:np.ndarray, categories:dict):
    """
    Given a dictionary mapping category labels to lists of pixel coordinates
    for axes 0 and 1 of a (M,N,C) ndarray (for C bands on the same domain),
    categorizes every pixel, and returns an integer-coded categorization.
    """
    labels, pixel_lists = zip(*categories.items())
    means = [] #
    for i in range(X.shape[2]):
        X[:,:,i] = enhance.linear_gamma_stretch(X[:,:,i])
    for i in range(len(pixel_lists)):
        means.append(np.array([
            sum([ X[y,x,j] for y,x in pixel_lists[i] ])/len(pixel_lists[i])
            for j in range(X.shape[2])
            ]))

    means_sq = [np.dot(means[i], means[i]) for i in range(len(means))]

    classified = np.full_like(X[:,:,0], fill_value=np.nan, dtype=np.uint8)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            px = X[i,j,:]
            disc = [ np.dot(px, px) + means_sq[m] - 2*np.dot(means[m], px)
                    for m in range(len(means)) ]
            classified[i,j] = disc.index(min(disc))
    return classified, labels

def mlc(X:np.ndarray, categories:dict):
    """
    Do maximum likelihood classification using the discriminant function

    :@param X: (M,N,b) ndarray with b independent variables
    :@param X: Dictionary mapping category labels to a set of 2-tuple pixel
            indeces of pixels in X belonging to that class.
    """
    cat_keys = list(categories.keys())
    cats = [X[tuple(map(np.asarray, zip(*categories[cat])))]
            for cat in cat_keys]
    means = [ np.mean(c, axis=0) for c in cats ]
    covs = [ np.cov(c.transpose()) for c in cats ]
    ln_covs = [ -1*np.log(np.linalg.det(C)) for C in covs ]
    inv_covs = [ np.linalg.inv(C) for C in covs ]
    def mlc_disc(px):
        G = np.zeros_like(np.arange(len(means)))
        for i in range(len(means)):
            obs_cov = np.dot(inv_covs[i], px-means[i])
            obs_cov = np.dot((px-means[i]).transpose(), obs_cov)
            #obs_cov = np.dot((px-means[i]).transpose(), inv_covs[i])
            G[i] = ln_covs[i]-obs_cov
        return np.argmax(G)
    classified = np.zeros_like(X[:,:,0])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            classified[i,j] = mlc_disc(X[i,j])
    return classified, cat_keys

def k_means(X:np.ndarray, clusters:int):
    pass

def pca(X:np.ndarray):
    pass

