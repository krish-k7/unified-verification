import numpy as np
from scipy.stats import dirichlet

def integrate_dirichlet(alpha_params, intervals, num_samples=100000):
    samples = dirichlet.rvs(alpha_params, size=num_samples)
    is_inside = np.ones(num_samples, dtype=bool)
    for i, (lower, upper) in enumerate(intervals):
        is_inside &= (samples[:, i] >= lower) & (samples[:, i] <= upper)
    integral_estimate = np.mean(is_inside)
    stderr = np.sqrt(integral_estimate*(1-integral_estimate)/num_samples)
    return integral_estimate, stderr