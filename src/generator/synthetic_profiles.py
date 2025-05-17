# python3 -m venv .venv
# whenever you work on this project you’ll run source .venv/bin/activate first

# to run the generator `python3 synthetic_profiles.py -n 5000 -o ./data/test_profiles.json --seed 42`

#!/usr/bin/env python3
"""
synthetic_profiles.py

Generate a JSON file of synthetic dating-app profiles with:
 - id, name, age, gender
 - city, latitude, longitude
 - interests, looking_for, bio
"""

import random
import json
import argparse
from faker import Faker

GENDERS = ["male", "female", "nonbinary"]
INTERESTS_POOL = [
    "hiking", "travel", "cooking", "music", "reading", "gaming", "sports",
    "yoga", "photography", "movies", "dancing", "painting", "fitness",
    "technology", "art", "coffee", "tea", "running", "biking", "fashion",
    "gardening", "writing", "meditation", "animals", "foodie", "theater"
]
LOOKING_FOR_POOL = ["relationship", "friendship", "casual", "networking"]

def generate_profile(pid: int, fake: Faker) -> dict:
    gender = random.choice(GENDERS)
    name = fake.name_male() if gender == "male" else fake.name_female()
    age = random.randint(18, 65)

    # location
    city = fake.city()
    latitude = float(fake.latitude())
    longitude = float(fake.longitude())

    # interests & what they’re looking for
    interests = random.sample(INTERESTS_POOL, k=random.randint(3, 6))
    looking_for = random.sample(LOOKING_FOR_POOL, k=random.randint(1, 2))

    # bio: weave in a couple interests + hometown
    bio = (
        f"{name.split()[0]} is a {age}-year-old from {city} who loves "
        f"{', '.join(interests[:-1])}, and {interests[-1]}. "
        f"Looking for {', '.join(looking_for)}."
    )

    return {
        "id": pid,
        "name": name,
        "age": age,
        "gender": gender,
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "interests": interests,
        "looking_for": looking_for,
        "bio": bio
    }

def generate_profiles(count: int, seed: int = None) -> list:
    if seed is not None:
        random.seed(seed)
    fake = Faker()
    profiles = [generate_profile(i, fake) for i in range(1, count + 1)]
    return profiles

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dating-app profiles"
    )
    parser.add_argument(
        "-n", "--num", type=int, default=1000,
        help="how many profiles to generate (default: 1000)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./data/profiles.json",
        help="output JSON file path"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for reproducibility"
    )
    args = parser.parse_args()

    profiles = generate_profiles(args.num, seed=args.seed)
    with open(args.output, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Generated {len(profiles)} profiles ▶ {args.output}")

if __name__ == "__main__":
    main()
