# organize_train_with_split.py

import os
import shutil

if os.path.exists('data'):
    shutil.rmtree('data')
    print("Deleted old data folder")

print("Creating directories...")
os.makedirs('data/train/cat', exist_ok=True)
os.makedirs('data/train/dog', exist_ok=True)
os.makedirs('data/val/cat', exist_ok=True)
os.makedirs('data/val/dog', exist_ok=True)

# Your Kaggle training folder
source_dir = '/Users/ahmeddogar/Documents/DogCat/train/train'

# Get all files
all_files = sorted(os.listdir(source_dir))
cat_files = [f for f in all_files if f.startswith('cat.')]
dog_files = [f for f in all_files if f.startswith('dog.')]

print(f"Found {len(cat_files)} cat images")
print(f"Found {len(dog_files)} dog images")

# Split: 80% train, 20% validation
# Cats: 12,500 total -> 10,000 train, 2,500 val
train_split = 0.8
cat_split_idx = int(len(cat_files) * train_split)
dog_split_idx = int(len(dog_files) * train_split)

cat_train = cat_files[:800]
cat_val = cat_files[800:1000]
dog_train = dog_files[:800]
dog_val = dog_files[800:1000]

print(f"\nSplit:")
print(f"  Training - Cats: {len(cat_train)}, Dogs: {len(dog_train)}")
print(f"  Validation - Cats: {len(cat_val)}, Dogs: {len(dog_val)}")

# Copy training cats
print("\nCopying training cats...")
for i, filename in enumerate(cat_train):
    if i % 1000 == 0:
        print(f"  Progress: {i}/{len(cat_train)}")
    src = os.path.join(source_dir, filename)
    dst = os.path.join('data/train/cat', filename)
    shutil.copy2(src, dst)

# Copy training dogs
print("Copying training dogs...")
for i, filename in enumerate(dog_train):
    if i % 1000 == 0:
        print(f"  Progress: {i}/{len(dog_train)}")
    src = os.path.join(source_dir, filename)
    dst = os.path.join('data/train/dog', filename)
    shutil.copy2(src, dst)

# Copy validation cats
print("Copying validation cats...")
for filename in cat_val:
    src = os.path.join(source_dir, filename)
    dst = os.path.join('data/val/cat', filename)
    shutil.copy2(src, dst)

# Copy validation dogs
print("Copying validation dogs...")
for filename in dog_val:
    src = os.path.join(source_dir, filename)
    dst = os.path.join('data/val/dog', filename)
    shutil.copy2(src, dst)

print("\n" + "="*50)
print("Done!")
print("="*50)
print(f"\nFinal structure:")
print(f"data/train/cat: {len(cat_train)} images")
print(f"data/train/dog: {len(dog_train)} images")
print(f"data/val/cat: {len(cat_val)} images")
print(f"data/val/dog: {len(dog_val)} images")
print(f"\nTotal training: {len(cat_train) + len(dog_train)}")
print(f"Total validation: {len(cat_val) + len(dog_val)}")
