# # Floating-Point Budgets: Why Ten Queries at ε = 0.1 Don't Fit in ε = 1
#
# Say you have a privacy budget of $\epsilon = 1$, and you plan ten queries at $\epsilon = 0.1$ each.
# On paper that uses the budget exactly. In OpenDP, the tenth query is refused.
# This recipe shows why, and three ways to plan a budget so that every query fits.
#
# The question first came up in [opendp/opendp#2131](https://github.com/opendp/opendp/issues/2131).
#
# ## The surprise
#
# We use the California PUMS sample and ask ten questions:
# how many people are at least 20 years old, at least 25, and so on up to 65.
# Each count gets $\epsilon = 0.1$.

# +
import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")

data = pl.scan_csv(
    dp.examples.get_california_pums_path(),
    has_header=False,
    new_columns=["age", "sex", "educ", "race", "income", "married"],
).select("age")

thresholds = range(20, 70, 5)


def count_at_least(context, age, epsilon=None):
    query = context.query(epsilon=epsilon) if epsilon is not None else context.query()
    return (
        query.filter(pl.col("age") >= age)
        .select(dp.len(signed=True))
        .release()
        .collect()
        .item()
    )


context = dp.Context.compositor(
    data=data,
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=1.0),
)

for age in thresholds:
    try:
        count = count_at_least(context, age, epsilon=0.1)
        print(
            f"age >= {age}: {count:4d}   privacy loss so far: {context.current_privacy_loss()!r}"
        )
    except dp.OpenDPException as e:
        print(f"age >= {age}: refused. {e.message}")
# -

# The running total drifts upward: 0.30000000000000004 after three queries, 0.9000000000000005 after nine.
# After ten it would be 1.0000000000000007, which is more than 1, so OpenDP refuses the last query.
#
# ## Why it happens
#
# Computers store numbers like 0.1 in binary, and most decimal fractions have no exact binary form.
# The number Python calls `0.1` is the closest double to 0.1, and it is slightly larger:

# +
from fractions import Fraction

print(Fraction(0.1))
print(Fraction(0.1) > Fraction(1, 10))
# -

# When ordinary Python adds two floats, it rounds the result to the nearest double, which can go either way.
# OpenDP instead rounds every step of its privacy accounting **up**.
# Rounding down could report a smaller privacy loss than was really spent, and then the guarantee would be false.
# Rounding up can only overstate the loss, so the guarantee always holds.
# The cost is that small errors add up in one direction, and an exact plan ends up a hair over budget.
#
# The excess here is less than $10^{-15}$, far too small to matter for privacy.
# But the budget check is strict, so it still blocks the query.
#
# ## Fix 1: tell the Context how you will split the budget
#
# If you know up front how many queries you will run, pass `split_evenly_over`
# (or `split_by_weights` for unequal parts).
# The Context then finds the largest equal parts whose rounded-up total still fits:

# +
context = dp.Context.compositor(
    data=data,
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=1.0),
    split_evenly_over=len(thresholds),
)
print("each query gets epsilon =", context.d_mids[0])

counts = [count_at_least(context, age) for age in thresholds]
print("all", len(counts), "queries answered")
# -

# Each query gets 0.09999999999999998 instead of 0.1.
# That shrinks each part by about 2 parts in $10^{16}$, so the noise is the same for any practical purpose.
#
# ## Fix 2: spend what is left on the last query
#
# If you decide on queries as you go, you can't split in advance.
# Instead, give the last query whatever budget remains, as reported by the Context:

# +
context = dp.Context.compositor(
    data=data,
    privacy_unit=dp.unit_of(contributions=1),
    privacy_loss=dp.loss_of(epsilon=1.0),
)

*first, last = thresholds
for age in first:
    count_at_least(context, age, epsilon=0.1)

remaining = context.remaining_privacy_loss()
print("remaining budget:", remaining)
count_at_least(context, last, epsilon=remaining)
print("privacy loss after the last query:", context.current_privacy_loss())
# -

# Ask for the remaining budget before the last query.
# Once the budget is used up, OpenDP will not report it any more.
#
# ## Fix 3: search for the largest part that fits
#
# Splits can also go over budget when you build compositions yourself.
# This is the case from the original issue: one part of a larger budget is $\epsilon = 1$,
# and you split it into 0.1 and 0.9.
# Here we use randomized response as the mechanism, as the issue did.

# +
m_rr = dp.m.make_randomized_response_bool(prob=0.55)


def make_split(last_epsilon):
    return dp.c.make_adaptive_composition(
        input_domain=m_rr.input_domain,
        input_metric=m_rr.input_metric,
        output_measure=m_rr.output_measure,
        d_in=1,
        d_mids=[0.1, last_epsilon],
    )


print("0.1 + 0.9 costs", make_split(0.9).map(1))
# -

# The rounded-up total is just over 1, so a composition with a budget of exactly 1 would refuse this split.
# `dp.binary_search_param` finds the largest value of the last part whose total is still at most 1:

# +
last_epsilon = dp.binary_search_param(make_split, d_in=1, d_out=1.0)
print("largest last part:", last_epsilon)
print("0.1 + that costs", make_split(last_epsilon).map(1))
# -

# This is also how Fix 1 works inside the Context: it uses a binary search to scale all the parts down together.
#
# ## Which splits are exact?
#
# Not every split goes over. Fractions whose denominator is a power of two (halves, quarters, eighths, ...)
# have an exact binary form, so they add up with no rounding at all.
# Other splits may or may not go over, depending on how the rounding falls:

# +
splits = {
    "2 x 1/2": [0.5] * 2,
    "4 x 1/4": [0.25] * 4,
    "3 x 1/3": [1 / 3] * 3,
    "0.2 + 0.3 + 0.5": [0.2, 0.3, 0.5],
    "10 x 0.1": [0.1] * 10,
}
for name, d_mids in splits.items():
    composition = dp.c.make_adaptive_composition(
        input_domain=m_rr.input_domain,
        input_metric=m_rr.input_metric,
        output_measure=m_rr.output_measure,
        d_in=1,
        d_mids=d_mids,
    )
    total = composition.map(1)
    print(
        f"{name:>16}: total {total!r:<20} {'fits' if total <= 1.0 else 'over budget'}"
    )
# -

# You can't tell in advance which splits are safe, so it is better not to rely on luck.
#
# ## Takeaways
#
# - OpenDP rounds its privacy accounting up, so that the reported loss is never smaller than the real loss.
#   The price is that a plan that adds up exactly on paper can go over budget by a tiny amount.
# - If you know how many queries you will run, use `split_evenly_over` or `split_by_weights`.
# - If you choose queries as you go, give the last query `context.remaining_privacy_loss()`.
# - When you build compositions by hand, use `dp.binary_search_param` to find the largest part that fits.
# - Each fix gives up less than $10^{-15}$ of the budget, which makes no practical difference to the noise.
