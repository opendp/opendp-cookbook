# # Polars streaming (for data that does not fit in memory)
#
# Polars supports [streaming](https://docs.pola.rs/user-guide/concepts/streaming/):
# Instead of reading the data all at once, it is read and processed in batches.
# In some cases this may be faster, and it also allows data to be processed that may not fit in memory.
#
# We'll first demonstrate Polars streaming without OpenDP,
# then at OpenDP, and finally deomonstrate an OpenDP query that can not be handled with streaming.

# ## Polars, without streaming, without OpenDP

import opendp.prelude as dp
import polars as pl

# +
scan_kwargs = {
    # Just so this doesn't need to be repeated on every example.
    "has_header": False,
    "new_columns": ["age", "sex", "educ", "race", "income", "married"],
    # Some incomes are written like "1e+05", so they can't be read as integers.
    "schema_overrides": {"income": pl.Float64},
}

data = pl.scan_csv(dp.examples.get_california_pums_path(), **scan_kwargs)

mean_age = data.select("age").mean().collect()

print(mean_age)
# -

# ## Polars with streaming, without OpenDP

# +
data = pl.scan_csv(dp.examples.get_california_pums_path(), **scan_kwargs)

query = data.select("age").mean()
# -

# At this point, we can inspect the query:
# This example is trivial, but in real examples it may point out processing bottlenecks.
#
# If you have graphviz installed, [show_graph](https://docs.pola.rs/user-guide/concepts/streaming/#inspecting-a-streaming-query) provides more information.

print(query.explain(format="tree", engine="streaming"))

# Proceed to run the query:

mean_age = query.collect(engine="streaming")
print(mean_age)

# ## Polars with streaming, with OpenDP
#
# Let's add OpenDP to the mix:

# +
dp.enable_features("contrib")
data = pl.scan_csv(dp.examples.get_california_pums_path(), **scan_kwargs)
context = dp.Context.compositor(
    data=data,
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=1.0),
    split_evenly_over=1,
    margins=[
        dp.polars.Margin(
            max_length=10_000,
        ),
    ],
)
dp_query = context.query().select(
    pl.col.age.dp.mean(bounds=(0, 100)),
)
dp_mean_age = dp_query.release().collect(engine="streaming")
print(dp_mean_age)
# -
