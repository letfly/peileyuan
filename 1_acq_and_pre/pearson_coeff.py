import numpy as np
import math


def compute_correlation(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    SSR = 0
    varx = 0
    vary = 0
    for i in xrange(0, len(x)):
        diff_xbar = x[i] - xbar
        diff_ybar = y[i] - ybar
        SSR += (diff_xbar * diff_ybar)
        varx += diff_xbar**2
        vary += diff_ybar**2

    SST = math.sqrt(varx*vary)
    return SSR/SST


# Polynomial Regression
def pearson_polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # fit values, and mean
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x) # or [p(z) for z in x]
    ybar = np.sum(y)/len(y) # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar)**2) # or sum([(yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2) # or sum([(yi-ybat)**2 for yi in y])
    results['determination'] = ssreg/sstot

    return results

def pearson_corr(x, y):
    return np.corrcoef(x, y, rowvar=0)[0][1]
test_x = [1, 3, 8, 7, 9]
test_y = [10, 12, 24, 21, 34]
print "r:", compute_correlation(test_x, test_y)
print "r^2", compute_correlation(test_x, test_y)**2

print "pearson_polyfit", pearson_polyfit(test_x, test_y, 1)["determination"]
print "pearson_corr", pearson_corr(test_x, test_y)
