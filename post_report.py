import os
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from slack_sdk import WebClient

client = WebClient(token=os.environ.get("SLACK_TOKEN"))

channel_id = "D07U250J0DT"  # Channel id


# Load the dataset
dataset = load_dataset("qgallouedec/trl-metrics", "models", split="train")

trl_tags = [
    "alignprop",
    "bco",
    "cpo",
    "ddpo",
    "dpo",
    "gkd",
    "iterative-sft",
    "kto",
    "online-dpo",
    "orpo",
    "ppo",
    "reward-trainer",
    "rloo",
    "sft",
    "xpo",
]

data = {"date": [], "no trainer tag": []}
for trl_tag in trl_tags:
    data[trl_tag] = []

# Process each model entry in the dataset
for entry in dataset:
    created_at = entry["created_at"].date()  # Get the creation date
    model_tags = entry["tags"]

    data["date"].append(created_at)

    # Determine the category
    at_least_one_tag = False
    for trl_tag in trl_tags:
        if trl_tag in model_tags:
            data[trl_tag].append(1)
            at_least_one_tag = True
        else:
            data[trl_tag].append(0)
    if not at_least_one_tag:
        data["no trainer tag"].append(1)
    else:
        data["no trainer tag"].append(0)

# Convert the processed data into a pandas DataFrame
df = pd.DataFrame(data)

# Aggregate data over time
df = df.groupby("date").sum().cumsum()

# Filter data for the last years
df = df[df.index > date(year=2024, month=1, day=1)]

# Sort by the final number of models
df = df.sort_values(by=df.index[-1], axis=1, ascending=False)

# Keep only the 5 most popular tags, and merge the rest into a single category "other"
top_tags = df.columns[:5]
df["other"] = df.drop(columns=top_tags).sum(axis=1)
df = df[top_tags.append(pd.Index(["other"]))]

# Place None in the last position
df = df[[col for col in df.columns if col != "no trainer tag"] + ["no trainer tag"]]

# Get the values and labels
values = [df[col].values for col in df.columns]
labels = df.columns

colors = plt.cm.rainbow([i / len(labels) for i in range(len(labels))])
plt.figure(figsize=(10, 6))
plt.stackplot(df.index, *values, labels=labels, alpha=0.8, colors=colors)
plt.legend(loc="upper left")
plt.xlabel("Date")
plt.ylabel("Cumulative Number of Models")
plt.title("Cumulative Number of Created Models Over Time")
plt.tight_layout()
image_path = "models_over_time.png"
plt.savefig(image_path)

with open(image_path, "rb") as file:
    response = client.files_upload_v2(file=file, channels=[channel_id])


# Helper function to filter data based on time range
def filter_date_range(df, date_col, start_date, end_date):
    return df[(df[date_col] >= start_date) & (df[date_col] < end_date)]


# Get the current time
now = datetime.now()

# Define the time periods for short-term (2 weeks)
one_week_ago = now - timedelta(weeks=1)
two_weeks_ago = now - timedelta(weeks=2)
three_weeks_ago = now - timedelta(weeks=3)
four_weeks_ago = now - timedelta(weeks=4)
five_weeks_ago = now - timedelta(weeks=5)

# Define the time periods for long-term (4 months)
four_months_ago = now - timedelta(weeks=16)
eight_months_ago = now - timedelta(weeks=32)

# Load datasets
issues_df = load_dataset("qgallouedec/trl-metrics", "issues", split="train").to_pandas()
comments_df = load_dataset("qgallouedec/trl-metrics", "issue_comments", split="train").to_pandas()
stargazers_df = load_dataset("qgallouedec/trl-metrics", "stargazers", split="train").to_pandas()
downloads_df = load_dataset("qgallouedec/trl-metrics", "pypi_downloads", split="train").to_pandas()
models_df = load_dataset("qgallouedec/trl-metrics", "models", split="train").to_pandas()
models_likes_df = load_dataset("qgallouedec/trl-metrics", "models_likes", split="train").to_pandas()

# Convert stargazers_df "starred_at" and models_df "created_at" to naive datetime
stargazers_df["starred_at"] = stargazers_df["starred_at"].dt.tz_convert(None)
models_df["created_at"] = models_df["created_at"].dt.tz_convert(None)
models_likes_df["liked_at"] = models_likes_df["liked_at"].dt.tz_convert(None)


def calculate_issues_stats(start_date, end_date):
    # Filter issues in the given period
    issues = filter_date_range(issues_df, "created_at", start_date, end_date)

    # Get the first comment for each issue
    first_comments = comments_df.groupby("issue_number").first().reset_index()

    # Merge to find the response time
    issues = issues.merge(
        first_comments[["issue_number", "created_at"]],
        left_on="number",
        right_on="issue_number",
        how="left",
        suffixes=("", "_first_comment"),
    )

    # Calculate the response time in days
    issues["response_time"] = (issues["created_at_first_comment"] - issues["created_at"]).dt.total_seconds() / (3600 * 24)

    # Count issues not answered within a week
    unanswered_issues_count = len(issues[issues["response_time"].isna() | (issues["response_time"] > 7)])
    return unanswered_issues_count


def calculate_stargazers_stats(start_date, end_date):
    return len(filter_date_range(stargazers_df, "starred_at", start_date, end_date))


def calculate_downloads_stats(start_date, end_date):
    return downloads_df[(downloads_df["day"] >= start_date.date()) & (downloads_df["day"] < end_date.date())][
        "num_downloads"
    ].sum()


def calculate_models_stats(start_date, end_date):
    return len(filter_date_range(models_df, "created_at", start_date, end_date))


def calculate_models_likes_stats(start_date, end_date):
    return len(filter_date_range(models_likes_df, "liked_at", start_date, end_date))


# Short-Term (2 Weeks) Stats
downloads_short_term = calculate_downloads_stats(two_weeks_ago, now)
downloads_short_term_prev = calculate_downloads_stats(four_weeks_ago, two_weeks_ago)
stargazers_short_term = calculate_stargazers_stats(two_weeks_ago, now)
stargazers_short_term_prev = calculate_stargazers_stats(four_weeks_ago, two_weeks_ago)
models_short_term = calculate_models_stats(two_weeks_ago, now)
models_short_term_prev = calculate_models_stats(four_weeks_ago, two_weeks_ago)
models_likes_short_term = calculate_models_likes_stats(two_weeks_ago, now)
models_likes_short_term_prev = calculate_models_likes_stats(four_weeks_ago, two_weeks_ago)
unanswered_issues_short_term = calculate_issues_stats(three_weeks_ago, one_week_ago)
unanswered_issues_short_term_prev = calculate_issues_stats(five_weeks_ago, three_weeks_ago)

# Long-Term (4 Months) Stats
downloads_long_term = calculate_downloads_stats(four_months_ago, now)
downloads_long_term_prev = calculate_downloads_stats(eight_months_ago, four_months_ago)
stargazers_long_term = calculate_stargazers_stats(four_months_ago, now)
stargazers_long_term_prev = calculate_stargazers_stats(eight_months_ago, four_months_ago)
models_long_term = calculate_models_stats(four_months_ago, now)
models_long_term_prev = calculate_models_stats(eight_months_ago, four_months_ago)
models_likes_long_term = calculate_models_likes_stats(four_months_ago, now)
models_likes_long_term_prev = calculate_models_likes_stats(eight_months_ago, four_months_ago)
unanswered_issues_long_term = calculate_issues_stats(four_months_ago, now)
unanswered_issues_long_term_prev = calculate_issues_stats(eight_months_ago, four_months_ago)

# Total Stats
stargazers_total = stargazers_df.shape[0]
downloads_total = downloads_df["num_downloads"].sum()
models_total = models_df.shape[0]
models_likes_total = models_likes_df.shape[0]


# Calculate Relative Changes
def calculate_relative_change(current, previous):
    return (current - previous) / previous * 100 if previous != 0 else float("inf")


downloads_short_term_change = calculate_relative_change(downloads_short_term, downloads_short_term_prev)
stargazers_short_term_change = calculate_relative_change(stargazers_short_term, stargazers_short_term_prev)
models_short_term_change = calculate_relative_change(models_short_term, models_short_term_prev)
models_likes_short_term_change = calculate_relative_change(models_likes_short_term, models_likes_short_term_prev)
unanswered_issues_short_term_change = calculate_relative_change(
    unanswered_issues_short_term, unanswered_issues_short_term_prev
)
downloads_long_term_change = calculate_relative_change(downloads_long_term, downloads_long_term_prev)
stargazers_long_term_change = calculate_relative_change(stargazers_long_term, stargazers_long_term_prev)
models_long_term_change = calculate_relative_change(models_long_term, models_long_term_prev)
models_likes_long_term_change = calculate_relative_change(models_likes_long_term, models_likes_long_term_prev)
unanswered_issues_long_term_change = calculate_relative_change(unanswered_issues_long_term, unanswered_issues_long_term_prev)

downloads_short_term_emoji = "🔴" if downloads_short_term_change < 0 else "🟢"
stargazers_short_term_emoji = "🔴" if stargazers_short_term_change < 0 else "🟢"
models_short_term_emoji = "🔴" if models_short_term_change < 0 else "🟢"
models_likes_short_term_emoji = "🔴" if models_likes_short_term_change < 0 else "🟢"
unanswered_issues_short_term_emoji = "🔴" if unanswered_issues_short_term_change > 0 else "🟢"
downloads_long_term_emoji = "🔴" if downloads_long_term_change < 0 else "🟢"
stargazers_long_term_emoji = "🔴" if stargazers_long_term_change < 0 else "🟢"
models_long_term_emoji = "🔴" if models_long_term_change < 0 else "🟢"
models_likes_long_term_emoji = "🔴" if models_likes_long_term_change < 0 else "🟢"
unanswered_issues_long_term_emoji = "🔴" if unanswered_issues_long_term_change > 0 else "🟢"


### Generate the Report
today = now.strftime("%Y-%m")

report = f"""
:trl:
*Monthly [TRL](https://github.com/huggingface/trl) Metrics Report*
:date: {today}


*:ultrafast_parrot: Short-Term* (Last 2 Weeks)

- {downloads_short_term_emoji} PyPI downloads: {downloads_short_term / 1_000:.1f}K (prev: {downloads_short_term_prev / 1_000:.1f}K, change: {downloads_short_term_change:+.2f}%)
- {stargazers_short_term_emoji} New GH :star:: {stargazers_short_term} (prev: {stargazers_short_term_prev}, change: {stargazers_short_term_change:+.2f}%)
- {models_short_term_emoji} New models on :hugging_face: Hub: {models_short_term} (prev: {models_short_term_prev}, change: {models_short_term_change:+.2f}%)
- {models_likes_short_term_emoji} New likes on :hugging_face: Hub: {models_likes_short_term} (prev: {models_likes_short_term_prev}, change: {models_likes_short_term_change:+.2f}%)
- {unanswered_issues_short_term_emoji} Issues not answered within a week: {unanswered_issues_short_term} (prev: {unanswered_issues_short_term_prev}, change: {unanswered_issues_short_term_change:+.2f}%)

*:60fps_parrot: Long-Term* (Last 4 Months)

- {downloads_long_term_emoji} PyPI downloads: {downloads_long_term / 1_000_000:.1f}M (prev: {downloads_long_term_prev/1_000_000:.1f}M, change: {downloads_long_term_change:+.2f}%, total: {downloads_total/1_000_000:.1f}M)
- {stargazers_long_term_emoji} New GH :star:: {stargazers_long_term} (prev: {stargazers_long_term_prev}, change: {stargazers_long_term_change:+.2f}%, total: {stargazers_total})
- {models_long_term_emoji} New models on :hugging_face: Hub: {models_long_term} (prev: {models_long_term_prev}, change: {models_long_term_change:+.2f}%, total: {models_total})
- {models_likes_long_term_emoji} New likes on :hugging_face: Hub: {models_likes_long_term} (prev: {models_likes_long_term_prev}, change: {models_likes_long_term_change:+.2f}%, total: {models_likes_total})
- {unanswered_issues_long_term_emoji} Issues not answered within a week: {unanswered_issues_long_term} (prev: {unanswered_issues_long_term_prev}, change: {unanswered_issues_long_term_change:+.2f}%)

Further comments and analysis :arrow_down:
[Code for generating this report](https://github.com/qgallouedec/trl-monitoring)
"""

print(report)

client.chat_postMessage(channel=channel_id, text=report)
