import os
from datetime import datetime
import time

import requests
from datasets import Dataset
from tqdm import tqdm

token = os.environ.get("GITHUB_PAT")


def get_full_response(url, headers, params=None):
    page = 1
    output = []
    params = params or {}
    while True:
        params = {**params, "page": page, "per_page": 100}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch issues: {response.text}")

        batch = response.json()
        if len(batch) == 0:
            break
        output.extend(batch)
        page += 1
        time.sleep(0.1)
    return output


issues_url = f"https://api.github.com/repos/huggingface/trl/issues"
gh_token = os.environ.get("GITHUB_PAT")
headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
issues = get_full_response(issues_url, headers, params={"state": "all"})

issues_dataset_dict = {
    "number": [],
    "title": [],
    "user": [],
    "state": [],
    "created_at": [],
    "closed_at": [],
    "comments_count": [],
}
comments_dataset_dict = {
    "user": [],
    "created_at": [],
    "body": [],
    "issue_number": [],
}
for issue in tqdm(issues):
    # Extract relevant information
    issue_number = issue["number"]
    title = issue["title"]
    created_at = datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    comments_count = issue["comments"]
    comments_url = issue["comments_url"]

    comments = get_full_response(comments_url, headers=headers)
    for comment in comments:
        comments_dataset_dict["user"].append(comment["user"]["login"])
        comments_dataset_dict["created_at"].append(datetime.strptime(comment["created_at"], "%Y-%m-%dT%H:%M:%SZ"))
        comments_dataset_dict["body"].append(comment["body"])
        comments_dataset_dict["issue_number"].append(issue_number)

    issues_dataset_dict["number"].append(issue_number)
    issues_dataset_dict["title"].append(title)
    issues_dataset_dict["user"].append(issue["user"]["login"])
    issues_dataset_dict["state"].append(issue["state"])
    issues_dataset_dict["created_at"].append(datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ"))
    issues_dataset_dict["closed_at"].append(
        datetime.strptime(issue["closed_at"], "%Y-%m-%dT%H:%M:%SZ") if issue["closed_at"] else None
    )
    issues_dataset_dict["comments_count"].append(comments_count)

issues_dataset = Dataset.from_dict(issues_dataset_dict)
comments_dataset = Dataset.from_dict(comments_dataset_dict)

issues_dataset.push_to_hub("qgallouedec/trl-metrics", config_name="issues", token=os.environ.get("HF_TOKEN"))
comments_dataset.push_to_hub("qgallouedec/trl-metrics", config_name="issue_comments", token=os.environ.get("HF_TOKEN"))
