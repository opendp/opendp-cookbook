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
# Timings vary from run to run, so we report the best of three.

# +
import time

import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")

n_samples = 1_000


def best_time_ms(function, repeats=3):
    """Runs `function` `repeats` times, and returns its last output and the fastest time in milliseconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = function()
        times.append(time.perf_counter() - start)
    return output, min(times) * 1000


scalar_gaussian = dp.m.make_gaussian(
    dp.atom_domain(T=int), dp.absolute_distance(T=int), scale=1.0
)

loop_samples, loop_ms = best_time_ms(
    lambda: [scalar_gaussian(0) for _ in range(n_samples)]
)
print(f"loop:   {loop_ms:.1f} ms")
# -

# The fast way is to give the mechanism a vector domain, and pass all the inputs in one call.
# The loop then runs in Rust, so you pay the cost of crossing from Python into the library only once.

# +
vector_gaussian = dp.m.make_gaussian(
    dp.vector_domain(dp.atom_domain(T=int)), dp.l2_distance(T=int), scale=1.0
)

vector_samples, vector_ms = best_time_ms(lambda: vector_gaussian([0] * n_samples))
print(f"vector: {vector_ms:.1f} ms ({loop_ms / vector_ms:.0f}x faster)")
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

# ## What the privacy cost means
#
# Each mechanism reports its privacy cost in zero-concentrated DP ($\rho$), for a given input distance:

print("one scalar call:", scalar_gaussian.map(1))
print("one vector call:", vector_gaussian.map(1))

# The numbers match, but they describe different things.
# The scalar cost is for **one call**. If the loop above had noised 1,000 statistics about the same people,
# the 1,000 releases would compose, and the total cost could be up to $1,000 \times 0.5 = 500$.
#
# The vector cost covers the whole vector at once. Its input distance is the L2 distance between the input vectors:
# the square root of the sum of the squared changes to each entry.
# If one person can change 4 entries by 1 each, the distance is $\sqrt{4} = 2$, and the cost is higher:

print("vector call, distance 2:", vector_gaussian.map(2))

# Pass the largest distance one person can cause to `map`, not just 1.
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

float_samples, float_ms = best_time_ms(lambda: float_gaussian([0.0] * n_samples))
print(f"floats, default k: {float_ms:.1f} ms")
print(float_samples[:3])
# -

# ## Faster float noise with a coarser grid
#
# Sampling on such a fine grid is slower than sampling integers.
# You can choose a coarser grid with the `k` argument.
# But then rounding the input to the grid can move each entry,
# which raises the sensitivity. For a vector, OpenDP needs to know its length to account for this:

try:
    dp.m.make_gaussian(float_domain, dp.l2_distance(T=float), scale=1.0, k=-50)
except dp.OpenDPException as e:
    print(e.message)

# So we give the domain a fixed size, and compare several grid spacings:

# +
sized_float_domain = dp.vector_domain(
    dp.atom_domain(T=float, nan=False), size=n_samples
)

print(f"{'k':>6} {'time':>9} {'rho at distance 1':>20} {'distinct values':>16}")
for k in [-1074, -50, -20, -10, -5]:
    gaussian = dp.m.make_gaussian(
        sized_float_domain, dp.l2_distance(T=float), scale=1.0, k=k
    )
    samples, milliseconds = best_time_ms(lambda: gaussian([0.0] * n_samples))
    print(
        f"{k:>6} {milliseconds:>6.1f} ms {gaussian.map(1.0):>20.15f} {len(set(samples)):>16}"
    )
# -

# With $k = -50$ the sampling is already much faster, and the privacy cost goes up only in the 14th decimal place.
# A grid coarser than that is no faster, and it costs more privacy.
# With a very coarse grid such as $k = -5$, the cost rises sharply, and the noise takes few distinct values.
# The increase also grows with the vector length, so check `map` for your own size.
#
# ## Takeaways
#
# - To draw many samples, use a vector domain and one call, not a Python loop.
# - A scalar mechanism's cost is per call, and repeated calls compose.
#   A vector mechanism's cost is for the L2 distance between whole vectors.
# - Integer inputs get discrete Gaussian noise directly. Float inputs are rounded to a grid of spacing $2^k$ first.
# - The default grid is exact. A coarser grid (such as $k = -50$) is faster and costs slightly more privacy.
#   For a vector, it also needs a known length.
