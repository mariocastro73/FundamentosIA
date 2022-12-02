from scipy.stats import beta, binom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Use a vector to represent the 101 different possibilities
n_bowls = 101
hypos = np.arange(n_bowls)


# Define the prior (uniform)
unnorm_prior = np.ones(n_bowls)
prior = unnorm_prior / np.sum(unnorm_prior)


# Define the likelihood
# Here, we implicitly include the type of cookies for every bowl
first_experiment = 'V'
likelihood = {
    'V' : hypos / (n_bowls - 1),
    'C' : 1 - hypos / (n_bowls - 1)
    }

eval_likelihood = likelihood[first_experiment]

# Define the posterior
unnorm_posterior = eval_likelihood * prior
posterior = unnorm_posterior / np.sum(unnorm_posterior)

# 2V posterior
second_experiment = 'V'
sec_unnorm_posterior = posterior * likelihood[second_experiment]
sec_posterior = sec_unnorm_posterior / np.sum(sec_unnorm_posterior)

# 2V, 1C posterior
third_experiment = 'C'
third_unnorm_posterior = sec_posterior * likelihood[third_experiment]
third_posterior = third_unnorm_posterior / np.sum(third_unnorm_posterior)

# MAP of last posterior
map_solution = hypos[np.argmax(third_posterior)]
print(f'MAP solution: {map_solution}')

# Prediction for next cookie
pred = 'V'
integrand_predictive = likelihood[pred] * third_posterior
predictive_proportion = auc(hypos, integrand_predictive)
predictive = predictive_proportion * (n_bowls)              


# ---


# Plot everything
fig, ax = plt.subplots(1, 1)

max_y_val = 1.1 * np.max( np.concatenate((prior, posterior, sec_posterior, third_posterior), axis = None) )

ax.plot(hypos, prior, 'r-', lw = 5, alpha = 0.6, label = 'prior')
ax.plot(hypos, posterior, 'g-', lw = 4, alpha = 0.6, label = "posterior (1V)")
ax.plot(hypos, sec_posterior, 'b-', lw = 4, alpha = 0.6, label = "posterior (2V)")
ax.plot(hypos, third_posterior, 'k-', lw = 4, alpha = 0.6, label = "posterior (2V, 1C)")

# ax.plot(hypos, eval_likelihood, 'b-', lw = 4, alpha = 0.6, label = "likelihood")

plt.vlines(map_solution, 0, max_y_val, color = 'k', linestyles = 'dashed', label = f'MAP: ({map_solution})')
plt.vlines(predictive, 0, max_y_val, color = 'k', linestyles = 'dotted', label = f'Bayesian pred.: ({predictive})')

ax.legend()

plt.show()
