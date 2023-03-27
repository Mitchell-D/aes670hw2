import numpy as np
import math as m
from pprint import pprint as ppt
import matplotlib.pyplot as plt

from aes670hw2 import enhance

def clustering():
    points = [(1,2), (2,3), (2,4), (3,4), (4,3), (4,4), (4,6), (5,7), (6,5)]
    points = list(map(np.asarray, points))
    #means = [ np.array([2,3]), np.array([5,6]) ]
    means = [ np.array([2.67,3.33]), np.array([4.75,5.5]) ]
    euclid = lambda x1,y1,x2,y2: m.sqrt((x2-x1)**2 + (y2-y1)**2)
    means = [ [euclid(*m,*p) for p in points] for m in means ]
    print(points)
    print( "C1: " + " ".join([f"{m:.2f}" for m in means[0]]))
    print( "C2: " + " ".join([f"{m:.2f}" for m in means[1]]))

def gauss(x, mean, stdev):
    t1 = 1/(stdev*m.sqrt(2*m.pi))
    t2 = -.5*( (x-mean)/stdev )**2
    return t1 * m.e**(t2)

def gauss_norm_lookup():
    mean = 8
    stdev = 2
    nbins = 16
    pixels = 36
    gauss_radius = 4.5

    X = np.linspace(mean-gauss_radius, mean+gauss_radius, nbins)
    print(X)
    C = np.cumsum(gauss(X, mean, stdev)) # ew
    C = np.round((C/C[-1])*pixels)
    print(C)
    plt.plot(range(nbins),C)
    #plt.plot(range(nbins),newC)
    plt.show()
    return C

def ch8():
    c1 = [[16,18,20,11,17,8,14,10,4,7],
          [13,13,13,12,12,11,11,10,9,9]]
    c2 = [[8,9,6,8,5,7,4,6,4,3],
          [8,7,7,6,5,5,4,3,2,2]]
    c3 = [[19,19,17,17,16,14,13,13,11,11],
          [6,3,8,1,4,5,8,1,6,3]]
    #fig, ax = plt.subplots()
    plt.scatter(*c1, label="Class 1")
    plt.scatter(*c2, label="Class 2")
    plt.scatter(*c3, label="Class 3")
    plt.scatter((5,), (9,), label="Undetermined pixel")
    plt.title("Pixel class samples")
    plt.xlabel("Band 1")
    plt.ylabel("Band 2")
    plt.legend()
    plt.show()

def q1():
    counts_1 = {
            "water":[np.asarray([20,23,21,21,22,19,17,20,24,19]),
                     np.asarray([11,7,8,7,7,3,1,4,8,4]) ],

            "veg":[np.asarray([60,53,63,52,34,38,38,38,31,50]),
                   np.asarray([142,130,140,126,92,120,151,111,81,158])],

            "bare":[np.asarray([74,103,98,111,84,76,72,98,99,108]),
                    np.asarray([66,82,78,86,67,67,67,71,80,71])],
            }
    counts_2 = {
            "water":[np.asarray([11,13,13,11,9,14,13,15,12,15]),
                     np.asarray([2,5,2,1,1,4,4,5,4,4])],

            "veg":[np.asarray([19,24,20,22,15,14,21,17,25,20]),
                   np.asarray([41,45,44,30,22,26,27,38,37,27])],

            "bare":[np.asarray([43,43,40,27,34,36,34,70,37,44]),
                    np.asarray([27,34,30,19,24,26,27,50,30,30])],
            }
    f1 = [lambda count: 0.44*count+0.5,
          lambda count: 1.18*count+0.9]
    f2 = [lambda count: 3.64*count-1.6,
          lambda count: 1.52*count-2.6]
    for cat in counts_1.keys():
        count_means_1 = { cat:[np.average(counts_1[cat][i])
                               for i in range(2) ]}
        count_means_2 = { cat:[np.average(counts_2[cat][i])
                               for i in range(2) ]}
        count_stdev_1 = { cat:[np.std(counts_1[cat][i]) for i in range(2) ]}
        count_stdev_2 = { cat:[np.std(counts_2[cat][i]) for i in range(2) ]}

        ref_1 = { cat:[f1[i](counts_1[cat][i]) for i in range(2) ]}
        ref_2 = { cat:[f2[i](counts_2[cat][i]) for i in range(2) ]}
        ref_means_1 = { cat:[np.average(f1[i](counts_1[cat][i]))
                             for i in range(2) ]}
        ref_means_2 = { cat:[np.average(f2[i](counts_2[cat][i]))
                             for i in range(2) ]}
        ref_stdev_1 = { cat:[np.std(f1[i](counts_1[cat][i]))
                             for i in range(2) ]}
        ref_stdev_2 = { cat:[np.std(f2[i](counts_2[cat][i]))
                             for i in range(2) ]}

        print()
        print(f"{cat} band 5 obs 1: " + \
                f"{ref_means_1[cat][0]:.3f} {ref_stdev_1[cat][0]:.3f}")
        print(f"{cat} band 7 obs 1: " + \
                f"{ref_means_1[cat][1]:.3f} {ref_stdev_1[cat][1]:.3f}")
        print(f"{cat} band 5 obs 2: " + \
                f"{ref_means_2[cat][0]:.3f} {ref_stdev_2[cat][0]:.3f}")
        print(f"{cat} band 7 obs 2: " + \
                f"{ref_means_2[cat][1]:.3f} {ref_stdev_2[cat][1]:.3f}")

        print("Count means")
        print(f"Mean obs 1: {count_means_1[cat]}")
        print(f"Mean obs 2: {count_means_2[cat]}")
        print("Count stdev")
        print(f"Stdev obs 1: {count_stdev_1[cat]}")
        print(f"Stdev obs 2: {count_stdev_2[cat]}")

if __name__=="__main__":
    #q1()
    #ch8()
    #print(gauss_norm_lookup())
    clustering()
