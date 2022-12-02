from scipy.stats import beta, binom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# We will do it all approximately (not using the explicit derivation of the problem)
n_grid_points = 1000
grid_points =   np.linspace(0.0, 1.0, n_grid_points)

# Define the prior
prior_fails =   2.5
prior_pass  =   7.5
prior = beta.pdf(grid_points, prior_fails, prior_pass)

# Mode of the prior
ma_priori_solution = grid_points[np.argmax(prior)]
print(f"A priori solution: {1 - ma_priori_solution}")

# Define the likelihood
n = 3   # Tries
y = 0   # Successes (fails of the exam)

# From a Bernouilli experiment
hypos = np.linspace(0.0, 1.0, n_grid_points)
likelihood = {
    'Y': 1 - hypos,
    'N': hypos}

# Alternative likelihood
alt_lik = binom.pmf(n = n, k = y, p = hypos)

# MLE solution 
mle_solution = grid_points[np.argmax(alt_lik)]

# Obtain the posterior
posterior_unnorm            =   prior * alt_lik
norm_constant_posterior     =   auc(grid_points, posterior_unnorm)
posterior                   =   posterior_unnorm / norm_constant_posterior

# MAP
map_solution = grid_points[np.argmax(posterior)]
print(f"MAP solution: {1 - map_solution}")

# Full Bayesian prediction
# Prediction for the next exam to pass as well
posterior_predictive_integrand  =   posterior * (grid_points)     # p(ŷ | \theta, x, y) * p(\theta | x, y),  with p(ŷ | \theta, x, y) = (1 - \theta) 
full_bayesian_prediction        =   auc(grid_points, posterior_predictive_integrand)

print(f"Full Bayesian solution: {1 - full_bayesian_prediction}")


##################
# Plot the results
##################

max_y_val = 1.1 * np.max( np.concatenate((prior, posterior, alt_lik), axis = None) )

fig, ax = plt.subplots(1, 1)

ax.plot(grid_points, prior, 'r-', lw = 5, alpha = 0.6, label = 'beta prior')
ax.plot(grid_points, posterior, 'g-', lw = 4, alpha = 0.6, label = "posterior")

# If we use the Bernouilli-defined likelihood, we have to pay more attention here...
# ax.plot(grid_points, likelihood['Y']**3, 'b--', lw = 4, alpha = 0.6, label = "Bern. likelihood")
ax.plot(grid_points, alt_lik, 'b-', lw = 4, alpha = 0.6, label = "norm. likelihood")

plt.vlines(full_bayesian_prediction, 0, max_y_val, color = 'k', linestyles = 'solid', label = 'Bayesian')
plt.vlines(map_solution, 0, max_y_val, color = 'g', linestyles = 'dashed', label = 'MAP')
plt.vlines(ma_priori_solution, 0, max_y_val, color = 'r', linestyles = 'dashdot', label = 'Prior mode')
plt.vlines(mle_solution, 0, max_y_val, color = 'b', linestyles = 'dotted', label = 'MLE')

ax.legend()

# Obtain the posterior
plt.show()
