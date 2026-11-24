# # Streaming large files with Polars
#
# OpenDP uses Polars to compute its queries.
# By default, Polars loads all the columns a query needs into memory before it computes anything,
# but Polars also has a streaming engine. This reads the data in batches,
# so you can query a file that is larger than memory.
#
# Use streaming when your file is close to the size of your machine's memory, or larger.
# If the data fits easily, the default engine is fine.
#
# Most DP queries can use the streaming engine including
# counts, sums, means, grouping, and most transformations of the data.
# Quantile queries are [not currently supported](https://github.com/opendp/opendp/issues/3289).
#
# Currently, streaming requires `.lazy()` or the `"honest-but-curious"` flag.
# Native support for streaming in `OnceFrame.collect()` is tracked in
# [this issue](https://github.com/opendp/opendp/issues/3138).
#
# This recipe shows how to stream a DP query,
# how much memory it saves, and what to look out for.
#
# ## Making a large test file
#
# We make up a CSV file with 3 million rows.
# That fits in memory easily, but it is big enough to show the difference.
# We write it in chunks, so making the file doesn't use much memory itself.

# +
import re
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")

rng = np.random.default_rng(42)
states = ["CA", "NY", "TX", "WA"]
n_rows = 3_000_000
chunk_size = 500_000

temp_dir = tempfile.TemporaryDirectory()
csv_path = Path(temp_dir.name) / "people.csv"
with csv_path.open("w") as csv_file:
    for row_number in range(0, n_rows, chunk_size):
        pl.DataFrame(
            {
                "state": rng.choice(states, chunk_size),
                "age": rng.integers(18, 90, chunk_size),
                "income": rng.integers(0, 200_000, chunk_size),
            }
        ).write_csv(csv_file, include_header=row_number == 0)

print(f"{csv_path.stat().st_size / 2**20:.0f} MB on disk")
# -

# `pl.scan_csv` doesn't read the file.
# It returns a `LazyFrame`, which is a plan to read the file later.
# (For Parquet files, use `pl.scan_parquet`.)
# OpenDP accepts a `LazyFrame` as its data, so nothing is loaded yet.
#
# The margins describe what we know about the data without looking at it.
# `max_length` says the table has at most 3 million rows, in total and in each state.
# `invariant="keys"` says the list of states is public, so it doesn't need to be protected.

context = dp.Context.compositor(
    data=pl.scan_csv(csv_path),
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=1.0),
    split_evenly_over=1,
    margins=[
        dp.polars.Margin(by=[], max_length=n_rows),
        dp.polars.Margin(by=["state"], invariant="keys", max_length=n_rows),
    ],
)

# ## Streaming a query
#
# `query.release()` returns a `OnceFrame`,
# and `OnceFrame.collect()` always uses the default in-memory engine.
# To use the streaming engine, extract the plan from the `OnceFrame` with `.lazy()`,
# then call `.collect(engine="streaming")` on the plan.
#
# `.lazy()` needs the `"honest-but-curious"` feature, because it removes protections,
# and you have to keep the privacy promises yourself:
#
# - Don't use the `OnceFrame` after calling `.lazy()`.
#   It can still be collected, and that would be another release that the budget wouldn't track.
# - Collect the plan from `.lazy()` only once.
#   Each call to `.collect()` is a new noisy release, but again the budget is not tracked.
# - Don't look at the order of the output rows, because the order can reveal information about the data.
#   Don't add steps to the plan that depend on row order (like `head` or `with_row_index`),
#   and shuffle the output once you have collected it, as the
#   `lazy()` docs [suggest](https://docs.opendp.org/en/stable/api/python/opendp.extras.polars.html#opendp.extras.polars.OnceFrame.lazy).
#
# Each release below is a count and a mean income for each state.
# `fill_null(0)` is needed because OpenDP can't know that the file has no missing values.

# +
dp.enable_features("honest-but-curious")


def release_plan():
    """Releases a count and a mean income for each state, and returns only the plan."""
    query = (
        context.query()
        .group_by("state")
        .agg(
            dp.len(signed=True),
            pl.col("income").fill_null(0).dp.mean((0, 200_000)),
        )
    )
    return query.release().lazy()


plan = release_plan()
# -

# ## Checking that the plan streams
#
# Before you collect, you can check which parts of the plan the streaming engine will run.
# `show_graph` with `plan_stage="physical"` describes the steps without running them.
# Here we pick out the name of each step.
# A step called `in-memory-map` would mean that Polars falls back to the in-memory engine for that part.


# +
def physical_steps(plan):
    """Returns the names of the steps the streaming engine would run for this plan."""
    graph = plan.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    return set(re.findall(r'\[label="([^"\\]+)', graph))


physical_steps(plan)
# -

# Every step here runs in the streaming engine.
# When a step isn't supported, Polars falls back quietly, without a warning.
# The query still gives the right answer, but it may use more memory than you expect.
#
# ## How much memory it saves
#
# To see how much memory each engine takes,
# we look at the peak memory of the whole Python process (its "resident set size").
# The peak never goes down, so we collect with the streaming engine first,
# then with the in-memory engine, and print the peak before and after each one.
# The `resource` module only exists on Linux and macOS.


# +
def peak_mb():
    """Returns the peak memory of this process so far, in megabytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, and macOS reports bytes.
    return int(peak / 2**20 if sys.platform == "darwin" else peak / 2**10)


def timed_collect(plan, engine):
    """Collects the plan once with the given engine, prints time and peak memory, and shuffles the rows."""
    before = peak_mb()
    start = time.perf_counter()
    released = plan.collect(engine=engine)
    seconds = time.perf_counter() - start
    after = peak_mb()
    print(f"{engine}: {seconds:.2f} s")
    print(f"peak memory {before} MB -> {after} MB; {after - before} MB increase")
    return released.sample(fraction=1, shuffle=True)


# Note that for this demo we're collecting twice from the same plan,
# without any budget error! Turning on `honest-but-curious` allows us to do this.
# In the real world, this double-dipping would be a privacy violation.
streamed = timed_collect(plan, "streaming")
in_memory = timed_collect(plan, "in-memory")
# -

# The in-memory engine starts from the streaming peak,
# so the difference between its two numbers is the least it needed beyond what streaming used.
# When we ran this, streaming raised the peak by about 90 MB,
# and the in-memory engine raised it by over 200 MB more.
# Your numbers will be different, and streaming may also be faster.
#
# The gap grows with the size of the file:
# the in-memory engine needs room for every value it reads,
# while the streaming engine only needs room for a few batches at a time.
#
# Here is the streamed release.
# The in-memory release looks the same, but with different noise.

streamed.sort("state")

# Finally, we delete the test file.

temp_dir.cleanup()

# ## Takeaways
#
# - Stream when your data is close to the size of your memory, or larger.
# - Give OpenDP a `LazyFrame` from `pl.scan_csv` or `pl.scan_parquet`, so the file isn't loaded up front.
# - To stream, use `release().lazy().collect(engine="streaming")`.
#   Collect only once, drop the `OnceFrame`, and shuffle the output.
# - Check the plan with `show_graph(engine="streaming", plan_stage="physical")` to see whether anything falls back to memory.
# - Counts, sums, means and grouping stream well.
#   Here the in-memory engine needed 2.5 to 3.2 times as much extra memory.
# - Quantiles don't currently stream, but you can still collect them in memory.
