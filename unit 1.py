###part1
##import numpy as np
##import matplotlib.pyplot as plt
##from scipy.stats import norm
### Parameters
##N = 50 # Size of each random vector
##M = 10000 # Number of random vectors
### Function to generate means and plot distribution
##def plot_means_and_verify_clt(distribution_name, samples, ax):
##    means = np.mean(samples, axis=1)
##    ax.hist(means, bins=50, density=True, alpha=0.7, label=f"Sampled {distribution_name}")
##    # Fit a normal distribution to the means
##    mu, std = norm.fit(means)
##    x = np.linspace(min(means), max(means), 1000)
##    ax.plot(x, norm.pdf(x, mu, std), label=f"Fitted Normal (\u03bc={mu:.2f}, \u03c3={std:.2f})")
##    ax.set_title(f"{distribution_name} (N={N}, M={M})")
##    ax.set_xlabel("Arithmetic Mean")
##    ax.set_ylabel("Density")
##    ax.legend()
### Subplot setup
##fig, axs = plt.subplots(2, 2, figsize=(14, 10))
##axs = axs.ravel()
### (a) Binomial distribution
##p = 0.5
##binomial_samples = np.random.binomial(n=10, p=p, size=(M, N))
##plot_means_and_verify_clt("Binomial", binomial_samples, axs[0])
### (b) Poisson distribution
##lambda_param = 5
##poisson_samples = np.random.poisson(lam=lambda_param, size=(M, N))
##plot_means_and_verify_clt("Poisson", poisson_samples, axs[1])
### (c) Normal distribution
##normal_samples = np.random.normal(loc=0, scale=1, size=(M, N))
##plot_means_and_verify_clt("Normal", normal_samples, axs[2])
##plt.tight_layout()
##plt.show()

#part2
#Demonstrating Centarl Limit Theorem Violation with the Cauchy-Lorentz Distribution
import numpy as np
import matplotlib.pyplot as plt
# Parameters
Rand_Vec_Size = 50 # Size of each random vector
No_Rand_Vec = 10000 # Number of random vectors
# Generate random samples from the Cauchy distribution
CauchyDist_Samples = np.random.standard_cauchy(size=(No_Rand_Vec, Rand_Vec_Size))
CauchyDist_Means = np.mean(CauchyDist_Samples, axis=1)
# Plot histogram of sample means
plt.figure(figsize=(8, 6))
plt.hist(CauchyDist_Means,bins=100,density=True,alpha=0.7,label="CL Sample Means")
plt.title("Distribution of Samle Means of Cauchy-Lorentz Distribution")
plt.xlabel("Sample Mean")
plt.ylabel("Density")
plt.legend()
# Add details
plt.xlim(-10, 10) # Limit x-axis to visualize better (Cauchy has heavy tails)
plt.grid()
plt.show()

###part3
##import numpy as np
##import matplotlib.pyplot as plt
##from scipy.stats import randint
### Function to compute joint probability for discrete variables
##def joint_probability_discrete(xi, yi):
##    joint_counts = {}
##    # Count occurrences of each pair (xi, yi)
##    for x, y in zip(xi, yi):
##        if (x, y) in joint_counts:
##            joint_counts[(x, y)] += 1
##        else:
##            joint_counts[(x, y)] = 1
##    # Calculate joint probabilities
##    total_samples = len(xi)
##    joint_prob = {key: value / total_samples for key, value in joint_counts.items()}
##    return joint_prob
##num_samples = 10000
##x_discrete = randint.rvs(1, 5, size=num_samples) # Random integers between 1 and 4
##y_discrete = randint.rvs(1, 5, size=num_samples) # Random integers between 1 and 4
### Compute joint probability for discrete variables
##joint_prob_discrete = joint_probability_discrete(x_discrete, y_discrete)
##print("Joint Probability Distribution (Discrete):")
##for (x, y), prob in sorted(joint_prob_discrete.items()):
##    print(f"P(X={x}, Y={y}) = {prob:.4f}")
### Visualize joint probability as a heatmap
##unique_x = np.unique(x_discrete)
##unique_y = np.unique(y_discrete)
##joint_matrix = np.zeros((len(unique_x), len(unique_y)))
##for (x, y), prob in joint_prob_discrete.items():
##    joint_matrix[x- unique_x[0], y- unique_y[0]] = prob
##plt.figure(figsize=(8, 6))
##plt.imshow(joint_matrix, origin="lower", aspect="auto", cmap="viridis",
##extent=[unique_y[0]- 0.5, unique_y[-1] + 0.5, unique_x[0]- 0.5, unique_x[-1] + 0.5])
##plt.colorbar(label="Probability")
##plt.title("Joint Probability Distribution (Discrete)")
##plt.xlabel("Y")
##plt.ylabel("X")
##plt.xticks(unique_y)
##plt.yticks(unique_x)
##plt.grid(False)
##plt.show()

###part4
##import numpy as np
##import matplotlib.pyplot as plt
##from scipy.stats import norm
### Function to compute joint probability for continuous variables
##def joint_probability_continuous(xi, yi, bins=20):
##    histogram, x_edges, y_edges = np.histogram2d(xi, yi, bins=bins, density=True)
##    return x_edges, y_edges, histogram
### Generate continuous data for two independent variables
##num_samples = 10000
##xi_continuous = np.random.normal(0, 1, num_samples) # Standard normal distribution for X
##yi_continuous = np.random.normal(0, 1, num_samples) # Standard normal distribution for Y
### Compute joint probability for continuous variables
##x_edges, y_edges, joint_prob_continuous = joint_probability_continuous(xi_continuous, yi_continuous, bins=20)
### Visualize joint probability as a heatmap
##plt.figure(figsize=(8, 6))
##plt.imshow(joint_prob_continuous.T, origin="lower", aspect="auto", cmap="viridis",
##extent=[y_edges[0], y_edges[-1], x_edges[0], x_edges[-1]])
##plt.colorbar(label="Probability Density")
##plt.title("Joint Probability Distribution (Continuous)")
##plt.xlabel("Y")
##plt.ylabel("X")
##plt.grid(False)
##plt.show()
### Display joint probabilities (sample values for bins)
##print("Sample Joint Probability Density Matrix:")
##print(joint_prob_continuous)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
n=10000
##x=np.random.normal(size=n)
##y=np.random.normal(size=n)
def generate(n):
    a=[]
    while len(a)<n:
        z=np.random.normal(loc=0.5,scale=0.1)
        if z >=0 and z<=1:
            a.append(z)
    return np.array(a)
x=generate(n)
y=generate(n)

hist,x_edg,y_edg =np.histogram2d(x,y,bins=20,density=True)

plt.imshow(hist,origin='lower',cmap='viridis',
extent=[y_edg[0],y_edg[-1],x_edg[0],x_edg[-1]])
##plt.xticks(np.round_(y_edg,2))
##plt.yticks(x_edg)
plt.show()
