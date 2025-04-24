import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

def toss_coin(n, q):
    tosses = np.random.binomial(1, q, n)
    return tosses
def binomial_test(n, q_observed, q_expected=0.5, alpha=0.05):
    # Calculate the number of heads in the observed tosses
    observed_heads = np.sum(q_observed)
    # Calculate the p-value for the test using the binomial distribution
    p_value = 2 * min(binom.cdf(observed_heads, n, q_expected),
                      1- binom.cdf(observed_heads- 1, n, q_expected))
    # Determine whether to reject the null hypothesis (H0) or not
    reject_null = p_value < alpha
    return p_value, reject_null, observed_heads
n = 100
q = 0.5
alpha = 0.1 
observed_tosses = toss_coin(n, q)
p_value, reject_null, observed_heads = binomial_test(n, observed_tosses,
                                                     q_expected=0.5, alpha=alpha)
print(f"Number of tosses: {n}")
print(f"Number of heads observed: {observed_heads}")
print(f"P-value: {p_value}")
if reject_null:
    print("Null hypothesis (q = 0.5) is rejected:\
          The observed result is significantly different from 0.5.")
else:
    print("Fail to reject the null hypothesis (q = 0.5).\
The result is not significantly different from 0.5.")
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, 0.5)
plt.plot(x, pmf, label='Expected Distribution (q = 0.5)', color='blue')
plt.vlines(observed_heads,0,binom.pmf(observed_heads,n,0.5),colors='red',label="Observed Heads")
print(pmf[observed_heads])
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.legend()
plt.title('Binomial Test for Coin Tosses')
plt.show()
