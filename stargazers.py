import os
from datetime import datetime

import pyarrow as pa
import requests
from datasets import Dataset


def get_stargazers(owner, repo, token):
    # Initialize the count and the page number
    page = 1
    stargazers = []
    while True:
        # Construct the URL for the stargazers with pagination
        stargazers_url = f"https://api.github.com/repos/{owner}/{repo}/stargazers?page={page}&per_page=100"

        # Send the request to GitHub API with appropriate headers
        headers = {"Accept": "application/vnd.github.v3.star+json", "Authorization": "token " + token}
        response = requests.get(stargazers_url, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch stargazers with status code {response.status_code}: {response.text}")

        stargazers_page = response.json()

        if not stargazers_page:  # Exit the loop if there are no more stargazers to process
            break

        stargazers.extend(stargazers_page)
        page += 1  # Move to the next page

    return stargazers


gh_token = os.environ.get("GITHUB_PAT")
stargazers = get_stargazers("huggingface", "trl", gh_token)
stargazers = {key: [stargazer[key] for stargazer in stargazers] for key in stargazers[0].keys()}
dataset = Dataset.from_dict(stargazers)


def clean(example):
    starred_at = datetime.strptime(example["starred_at"], "%Y-%m-%dT%H:%M:%SZ")
    starred_at = pa.scalar(starred_at, type=pa.timestamp("s", tz="UTC"))
    return {"starred_at": starred_at, "user": example["user"]["login"]}


dataset = dataset.map(clean, remove_columns=dataset.column_names)
dataset.push_to_hub("qgallouedec/trl-metrics", config_name="stargazers", token=os.environ.get("HF_TOKEN"))
