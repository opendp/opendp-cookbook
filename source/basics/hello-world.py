# # Hello World
#
# This is a very basic example of **markdown** interleaved with code.

# +
import opendp.prelude as dp
dp.enable_features("contrib")
# -

space = dp.atom_domain(
    T=float, nan=False
), dp.absolute_distance(T=float)
laplace_mechanism = space >> dp.m.then_laplace(scale=1.0)
dp_value = laplace_mechanism(100.0)
print(dp_value)

# Ta-da! We've added calibrated noise.
