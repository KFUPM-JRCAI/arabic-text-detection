import json
import random
from pathlib import Path


RANDOM_SEED = 42
NUM_POSTS_TO_SAMPLE = 100


def sample_and_save_posts(
    posts_path,
    num_samples=NUM_POSTS_TO_SAMPLE,
    seed=RANDOM_SEED,
    output_path=None,
):
    with open(posts_path, 'r', encoding='utf-8') as f:
        posts = json.load(f)

    rng = random.Random(seed)
    sampled_posts = rng.sample(posts, num_samples)
    sampled_indices = [posts.index(post) for post in sampled_posts]

    print(f"Total posts available: {len(posts)}")
    print(f"Sampled {len(sampled_posts)} posts with seed {seed}")
    print(f"Sample indices (first 10): {sampled_indices[:10]}")

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "random_seed": seed,
                "num_samples": num_samples,
                "sampled_indices": sampled_indices,
                "sampled_posts": sampled_posts
            }, f, ensure_ascii=False, indent=2)

        print(f"Sampled posts reference saved to: {output_path}")

    return sampled_posts, sampled_indices
