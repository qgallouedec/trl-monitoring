import asyncio
import os
from datetime import datetime

import aiohttp
from datasets import Dataset
from tqdm.asyncio import tqdm_asyncio


gh_token = os.environ.get("GITHUB_PAT")
headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}

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


async def fetch_json(session, url, params=None):
    """Fetch JSON data asynchronously."""
    params = params or {}
    output = []
    page = 1
    while True:
        params["page"] = page
        async with session.get(url, headers=headers, params=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch data from {url}: {await response.text()}")
            batch = await response.json()
            if not batch:
                break
            output.extend(batch)
            page += 1
    return output


async def fetch_issue_data(session, issue):
    """Fetch comments for a given issue and append both issue and comment data."""
    issue_number = issue["number"]
    created_at = datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    closed_at = datetime.strptime(issue["closed_at"], "%Y-%m-%dT%H:%M:%SZ") if issue["closed_at"] else None

    issues_dataset_dict["number"].append(issue_number)
    issues_dataset_dict["title"].append(issue["title"])
    issues_dataset_dict["user"].append(issue["user"]["login"])
    issues_dataset_dict["state"].append(issue["state"])
    issues_dataset_dict["created_at"].append(created_at)
    issues_dataset_dict["closed_at"].append(closed_at)
    issues_dataset_dict["comments_count"].append(issue["comments"])

    comments = await fetch_json(session, issue["comments_url"])
    for comment in comments:
        comments_dataset_dict["user"].append(comment["user"]["login"])
        comments_dataset_dict["created_at"].append(datetime.strptime(comment["created_at"], "%Y-%m-%dT%H:%M:%SZ"))
        comments_dataset_dict["body"].append(comment["body"])
        comments_dataset_dict["issue_number"].append(issue_number)


async def main():
    issues_url = "https://api.github.com/repos/huggingface/trl/issues"
    async with aiohttp.ClientSession() as session:
        issues = await fetch_json(session, issues_url, params={"state": "all"})

        tasks = [fetch_issue_data(session, issue) for issue in issues]
        await tqdm_asyncio.gather(*tasks)


# Run the async main function to populate datasets
asyncio.run(main())

# Create and push datasets to the Hub
issues_dataset = Dataset.from_dict(issues_dataset_dict)
comments_dataset = Dataset.from_dict(comments_dataset_dict)

issues_dataset.push_to_hub("qgallouedec/trl-metrics", config_name="issues", token=os.environ.get("HF_TOKEN"))
comments_dataset.push_to_hub("qgallouedec/trl-metrics", config_name="issue_comments", token=os.environ.get("HF_TOKEN"))
