# # Many Gaussian Samples at Once
#
# Sometimes you need many samples of Gaussian noise: to noise a long vector of statistics,
# or to study the noise itself.
# Calling a scalar mechanism in a Python loop works, but it is slow.
# This recipe shows a faster way, and explains what OpenDP does when you ask for integer or float noise.
#
# It is based on [a discussion on the OpenDP repository](https://github.com/opendp/opendp/discussions/2135).
#
# ## One call instead of a loop
#
# Here is the slow way: a scalar mechanism, called once for each sample.

# +
import time

import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")

n_samples = 10_000

scalar_gaussian = dp.m.make_gaussian(
    dp.atom_domain(T=int), dp.absolute_distance(T=int), scale=1.0
)

start = time.perf_counter()
loop_samples = [scalar_gaussian(0) for _ in range(n_samples)]
loop_seconds = time.perf_counter() - start
print(f"loop:   {loop_seconds * 1000:.0f} ms")
# -

# The fast way is to give the mechanism a vector domain, and pass all the inputs in one call.
# The loop then runs in Rust, so you pay the cost of crossing from Python into the library only once.

# +
vector_gaussian = dp.m.make_gaussian(
    dp.vector_domain(dp.atom_domain(T=int)), dp.l2_distance(T=int), scale=1.0
)

start = time.perf_counter()
vector_samples = vector_gaussian([0] * n_samples)
vector_seconds = time.perf_counter() - start
print(
    f"vector: {vector_seconds * 1000:.0f} ms ({loop_seconds / vector_seconds:.0f}x faster)"
)
# -

# The exact speedup depends on your machine, but the vector call is usually several times faster.
# The samples are not drawn in parallel: the gain comes only from making one call instead of many.
#
# Both ways give samples from the same distribution:

# +
for name, samples in [("loop", loop_samples), ("vector", vector_samples)]:
    print(
        f"{name:>6}: mean {np.mean(samples):+.3f}, standard deviation {np.std(samples):.3f}"
    )
# -

# The two mechanisms also have the same privacy cost for an input distance of 1,
# measured in zero-concentrated DP ($\rho$):

print("scalar:", scalar_gaussian.map(1))
print("vector:", vector_gaussian.map(1))

# But the distance means something different in each case.
# For the vector mechanism, it is the L2 distance between the whole input vectors.
# If one person can change many entries of the vector, the distance is larger than 1,
# and you must pass that larger distance to `map` to get the right privacy cost.
#
# ## Integer noise is discrete Gaussian noise
#
# With an integer domain, as above, the samples are integers:
# OpenDP adds noise from the discrete Gaussian distribution.

print(vector_samples[:10])

# `make_gaussian` always uses the discrete Gaussian internally, even for floats.
# For a float input, OpenDP first rounds it to a grid of points spaced $2^k$ apart,
# adds discrete Gaussian noise on that grid, and converts the result back to a float.
#
# By default, $k = -1074$: the grid spacing is the smallest gap between doubles,
# so the rounding is exact and the result looks like continuous Gaussian noise.

# +
float_domain = dp.vector_domain(dp.atom_domain(T=float, nan=False))
float_gaussian = dp.m.make_gaussian(float_domain, dp.l2_distance(T=float), scale=1.0)

start = time.perf_counter()
float_samples = float_gaussian([0.0] * n_samples)
float_seconds = time.perf_counter() - start
print(f"floats, default k: {float_seconds * 1000:.0f} ms")
print(float_samples[:3])
# -

# ## Faster float noise with a coarser grid
#
# Sampling on such a fine grid is slower than sampling integers.
# You can choose a coarser grid with the `k` argument.
# But then rounding the input to the grid can move each entry,
# which raises the sensitivity, and OpenDP needs to know the vector length to account for it:

try:
    dp.m.make_gaussian(float_domain, dp.l2_distance(T=float), scale=1.0, k=-50)
except dp.OpenDPException as e:
    print(e.message)

# So we give the domain a fixed size, and compare several grid spacings:

# +
sized_float_domain = dp.vector_domain(
    dp.atom_domain(T=float, nan=False), size=n_samples
)

print(f"{'k':>6} {'time':>8} {'rho at distance 1':>20} {'distinct values':>16}")
for k in [-1074, -50, -20, -10, -5]:
    gaussian = dp.m.make_gaussian(
        sized_float_domain, dp.l2_distance(T=float), scale=1.0, k=k
    )
    start = time.perf_counter()
    samples = gaussian([0.0] * n_samples)
    milliseconds = (time.perf_counter() - start) * 1000
    print(
        f"{k:>6} {milliseconds:>6.0f} ms {gaussian.map(1.0):>20.15f} {len(set(samples)):>16}"
    )
# -

# With $k = -50$ the sampling is already much faster, and the privacy cost goes up only in the 14th decimal place.
# With a very coarse grid such as $k = -5$, the cost rises sharply, and the noise takes few distinct values.
# The increase also grows with the vector length, so check `map` for your own size.
#
# ## Takeaways
#
# - To draw many samples, use a vector domain and one call, not a Python loop.
# - The privacy cost of the vector mechanism is for the L2 distance between whole vectors.
# - Integer inputs get discrete Gaussian noise directly. Float inputs are rounded to a grid of spacing $2^k$ first.
# - The default grid is exact. A coarser grid (such as $k = -50$) is faster, needs a known vector length,
#   and costs slightly more privacy.
