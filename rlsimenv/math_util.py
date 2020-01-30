import numpy as np

def genlogistic_function(t, b=1, a=0, k=1, nu=1, q=1, c=1):
    """Naive implementation of https://en.wikipedia.org/wiki/Generalised_logistic_function

    :param t: input in (-inf, inf)
    :param b: growth rate. As b->inf, the curve approaches a step function, roughly.
    :param a: lower asymptote
    :param k: upper asymptote when c=1
    :param nu: 
    :param q: 
    :param c: 
    :returns: 
    :rtype: 

    """
    num = k - a
    exp = np.exp(-b*t)
    denom = np.power(c+q*exp, 1./nu)
    y = a + num/denom
    return y
