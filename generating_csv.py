from faker import Faker
import pandas as pd
import random

fake = Faker()

# Number of samples per sentiment
n_samples = 50  # adjust for demo size

def generate_review(sentiment):
    """Generate a fake review based on sentiment."""
    base_text = fake.sentence(nb_words=random.randint(5, 15))
    if sentiment == "POSITIVE":
        modifiers = ["excellent", "fantastic", "amazing", "highly recommend", "loved it", "best ever"]
        return f"{base_text} {random.choice(modifiers)}"
    elif sentiment == "NEGATIVE":
        modifiers = ["terrible", "worst", "disappointing", "hate it", "never again", "regret"]
        return f"{base_text} {random.choice(modifiers)}"
    else:  # NEUTRAL
        modifiers = ["okay", "average", "fine", "acceptable", "not bad", "mediocre"]
        return f"{base_text} {random.choice(modifiers)}"

# Generate dataset
data = []
for _ in range(n_samples):
    data.append({"text": generate_review("POSITIVE"), "label": "POSITIVE"})
    data.append({"text": generate_review("NEGATIVE"), "label": "NEGATIVE"})
    data.append({"text": generate_review("NEUTRAL"), "label": "NEUTRAL"})

# Shuffle dataset
random.shuffle(data)

# Create DataFrame and save CSV
df = pd.DataFrame(data)
df.to_csv("faker_demo_sentiment.csv", index=False)
print("CSV 'faker_demo_sentiment.csv' created successfully!")
