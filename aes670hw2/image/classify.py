
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
