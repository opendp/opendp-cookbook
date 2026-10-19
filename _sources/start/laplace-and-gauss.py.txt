# # Just adding Laplace and Gaussian noise
#
# Using higher-level APIs helps to ensure the correctness of your analysis,
# but sometimes you just want to add noise.

# +
import opendp.prelude as dp

dp.enable_features("contrib")
# -

space = dp.atom_domain(T=float, nan=False), dp.absolute_distance(T=float)
laplace_mechanism = space >> dp.m.then_laplace(scale=1.0)
dp_value = laplace_mechanism(100.0)
print(dp_value)

# Ta-da! We've added calibrated Laplace noise to 100.0.
# Gaussian noise is also possible:

gaussian_mechanism = space >> dp.m.then_gaussian(scale=1.0)
dp_value = gaussian_mechanism(100.0)
print(dp_value)

# If you have any doubts about whether Laplace or Gaussian noise is appropriate,
# or how to choose an appropriate scale,
# the [Context API](https://docs.opendp.org/en/stable/getting-started/index.html)
# will handle those details for you.
