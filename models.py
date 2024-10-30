import asyncio
import os
from datetime import datetime

import aiohttp
import pyarrow as pa
from datasets import Dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from tqdm.asyncio import tqdm

api = HfApi()
models = api.list_models(tags="trl")
dataset_list = [
    {
        "id": model.id,
        "created_at": model.created_at,
        "likes": model.likes,
        "downloads": model.downloads,
        "tags": model.tags,
    }
    for model in models
]
dataset_dict = {key: [d[key] for d in dataset_list] for key in dataset_list[0].keys()}
dataset = Dataset.from_dict(dataset_dict)
dataset.push_to_hub("qgallouedec/trl-metrics", config_name="models", token=os.environ.get("HF_TOKEN"))


liked_list = []


async def fetch_likes(session, model_dict):
    """Fetch likes data asynchronously for a given model."""
    if model_dict["likes"] == 0:
        return  # Skip models with zero likes

    model_id = model_dict["id"]
    url = f"https://huggingface.co/api/models/{model_id}/likers?expand%5B%5D=likeAt"
    async with session.get(url) as response:
        response.raise_for_status()
        likes_data = await response.json()

        # Process each like entry
        for like in likes_data:
            user = like["user"]
            liked_at = like["likedAt"]
            liked_at = datetime.strptime(liked_at, "%Y-%m-%dT%H:%M:%S.%fZ")
            liked_at = pa.scalar(liked_at, type=pa.timestamp("s", tz="UTC"))
            liked_list.append({"user": user, "model_id": model_id, "liked_at": liked_at})


async def main(dataset_list):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_likes(session, model_dict) for model_dict in dataset_list]
        # Use tqdm to display progress
        await tqdm.gather(*tasks)


# Run the async main function
asyncio.run(main(dataset_list))

dataset_dict = {key: [d[key] for d in liked_list] for key in liked_list[0].keys()}
dataset = Dataset.from_dict(dataset_dict)
dataset = dataset.sort("liked_at")
dataset.push_to_hub("qgallouedec/trl-metrics", config_name="models_likes", token=os.environ.get("HF_TOKEN"))
