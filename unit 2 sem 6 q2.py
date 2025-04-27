import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, binom
N = 100 # Total number of coin flips
M = 60 # Number of heads
a = 2
b = 2
mu = 0.5 # Mean for Gaussian prior
sigma = 0.1 # Standard deviation for Gaussian prior def likelihood (f, N, M):
def likelihood(f,N,M):
    return binom. pmf (M, N, f)
def beta_prior(f, a, b):
    return beta.pdf(f, a, b)
def gaussian_prior(f, mu, sigma):
    return norm.pdf(f, mu, sigma)
def posterior_beta(f, N, M, a,b):
    return likelihood (f, N, M) * beta_prior(f, a, b)
def posterior_gaussian(f, N, M, mu, sigma):
    return likelihood (f, N, M) * gaussian_prior(f, mu, sigma)
f_values = np. linspace(0, 1, 1000)
likelihood_values = [likelihood (f_val, N, M) for f_val in f_values]
beta_prior_values = [beta_prior (f_val, a, b) for f_val in f_values]
gaussian_prior_values = [gaussian_prior(f_val, mu, sigma) for f_val in f_values]
posterior_beta_values = [posterior_beta(f_val, N, M, a, b) for f_val in f_values]
posterior_gaussian_values = [posterior_gaussian(f_val, N, M, mu, sigma) for f_val in f_values]
likelihood_values /= np.sum (likelihood_values) 
beta_prior_values /= np. sum (beta_prior_values)
gaussian_prior_values /= np.sum(gaussian_prior_values)
posterior_beta_values /= np.sum(posterior_beta_values)
posterior_gaussian_values /= np.sum(posterior_gaussian_values)
print("Maximum Likelihood at ", f_values[np.argmax(likelihood_values)])
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(f_values, beta_prior_values, label="Beta Prior", color="blue", linestyle='--')
plt.plot(f_values, likelihood_values, label="Likelihood", color="gray")
plt.plot(f_values, posterior_beta_values, label="Posterior Beta", color="red", linestyle='-.')
plt.title("Beta Prior")
plt.xlabel("Fraction of Heads (f)")
plt.ylabel("Normalized Probability Density")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(f_values, likelihood_values, label="Likelihood", color="purple")
plt.plot(f_values, gaussian_prior_values, label="Gaussian Prior", color="green", linestyle='--')
plt.plot(f_values, posterior_gaussian_values, label="Posterior Gaussian",color="violet",linestyle='-.')
plt.title("Gaussian Prior")
plt.xlabel("Fraction of Heads (f)")
plt.ylabel("Normalized Probability Density")
plt.legend()
plt.tight_layout()
plt.show()
