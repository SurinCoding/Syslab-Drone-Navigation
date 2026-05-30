import os
import re
import difflib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from PIL import Image

def black_white_threshold(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    binary = np.where(gray < 35, 0, 255).astype(np.uint8)

    binary_rgb = np.stack([binary, binary, binary], axis=-1)

    return Image.fromarray(binary_rgb)

def normalize_filename_for_match(path_or_name):
    name = os.path.basename(str(path_or_name)).strip()

    
    if "-" in name:# REMOVE LABEL-STUDIO PREFIX
        first, rest = name.split("-", 1)
        if len(first) <= 12:
            name = rest

    stem, ext = os.path.splitext(name)
    # print(stem)

    ext = ext.lower()
    # print(ext)
    stem = stem.replace("_", " ") #REPLACE _ W/ " "
    # print(stem)


    #SUB 2 _2 to (2) IFF appears at end
    stem = re.sub(r"[\s_]+(\d+)$", r" (\1)", stem)
    # print(stem)
    # return



    # print(re.subr"(\d{4})\s+(\d{1,3}\s+\d{2}\s+\d{2}\s+[AP]M(?:\s+\(\d*\))?)$")

    #ADD Comma after Year
    stem = re.sub(
        r"(\d{4})\s+(\d{1,2}\s+\d{2}\s+\d{2}\s+[AP]M(?:\s+\(\d+\))?)$",
        r"\1, \2",
        stem,
        flags=re.IGNORECASE
    )

    # REMOVE PUNCT
    stem = stem.replace(",", " ")
    stem = re.sub(r"\s+", " ", stem).strip().lower()
    # print (ext)
    # print(stem)
    return stem + ext


def build_image_index(image_root):
    image_index = {}
    all_files = []

    for fname in os.listdir(image_root):
        full_path = os.path.join(image_root, fname)

        if os.path.isfile(full_path):
            norm = normalize_filename_for_match(fname)
            # print(norm)


            image_index.setdefault(norm, []).append(full_path)
            all_files.append(fname)

    return image_index, all_files

#CLEANING IMAGES, due to how label-studio works, how images were stored in dropbox would not be found, unless using matching techniques below. 3 diff methods that led to ALL images being found, none being missing
def resolve_image_path_from_index(image_path_from_csv, image_root, image_index, all_files, cutoff=0.72):
    raw_name = os.path.basename(str(image_path_from_csv))
    norm_csv = normalize_filename_for_match(raw_name)

    #1 EXACT MATCH (Normalized)
    if norm_csv in image_index:
        return image_index[norm_csv][0], f"exact-normalized: {norm_csv}"

    #2 Fuzzy matched (Normalized)
    normalized_choices = list(image_index.keys())
    close_norm = difflib.get_close_matches(norm_csv, normalized_choices, n=1, cutoff=cutoff)
    # print(close_norm)

    if close_norm:
        best_norm = close_norm[0]
        return image_index[best_norm][0], f"fuzzy-normalized: {best_norm}"

    #3 fuzzy match on original name
    cleaned_csv_for_display = raw_name
    if "-" in cleaned_csv_for_display:
        first, rest = cleaned_csv_for_display.split("-", 1)
        # print(rest)
        if len(first) <= 12:
            cleaned_csv_for_display = rest
    cleaned_csv_for_display = cleaned_csv_for_display.replace("_", " ")

    close_file = difflib.get_close_matches(cleaned_csv_for_display, all_files, n=1, cutoff=cutoff)
    if close_file:
        best_file = close_file[0]
        # print(best_file)
        return os.path.join(image_root, best_file), f"fuzzy-original: {best_file}"

    return None, f"no match for normalized='{norm_csv}'"


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    

    except Exception:
        return False


class CVData(Dataset):
    def __init__(self, pandasDataset, image_root, transform=None, image_index=None, all_files=None):
        self.df = pandasDataset.reset_index(drop=True).copy()

        self.df["choice"] = self.df["choice"].astype(str).str.strip()

        storeUniqueValues = {
            "Forward": 0,
            "Left": 1,
            "Right": 2,
            "Down": 3,
            "Up": 4,
            "Left Up": 5,
            "Left Down": 6,
            "Right Down": 7,
            "Right U": 8,
            "Right Up": 8 #Added extra in case labeled Right Up instead of Right U
        }

        self.df["label"] = self.df["choice"].map(storeUniqueValues)

        bad_labels = self.df[self.df["label"].isna()]
        if len(bad_labels) > 0:
            # print("\nUnmapped choice values found:")
            # print(bad_labels["choice"].value_counts())

            print("\nDropping rows with unmapped labels...\n")
            self.df = self.df.dropna(subset=["label"]).reset_index(drop=True)

        self.df["label"] = self.df["label"].astype(int)

        self.image_root = image_root
        self.transform = transform

        if image_index is None or all_files is None:
            self.image_index, self.all_files = build_image_index(image_root)

        else:
            self.image_index = image_index
            self.all_files = all_files

    def __len__(self):
        return len(self.df)

    def resolve_image_path(self, image_path_from_csv):
        return resolve_image_path_from_index( image_path_from_csv, self.image_root, self.image_index, self.all_files)
    #RETURNS THE IMAGE based off index we created before

    def __getitem__(self, idx):
        image_path_from_csv = str(self.df.loc[idx, "image"])
        local_path, matched = self.resolve_image_path(image_path_from_csv)

        if local_path is None: #SHOULDN"T OCCUR
            raise FileNotFoundError(
                f"Image not found.\n"
                f"CSV value: {image_path_from_csv}\n"
                f"Match result: {matched}\n"
                f"Image folder: {self.image_root}"
            )

        try:#SHOULDN"T RUN
            with Image.open(local_path) as img:
                image = img.convert("RGB")
        except Exception as e: #FIND MATCHED PATH IF FAILED
            raise RuntimeError(
                f"Failed to open image.\n"
                f"CSV value: {image_path_from_csv}\n"
                f"Resolved path: {local_path}\n"
                f"Match result: {matched}\n"
                f"Error type: {type(e).__name__}\n"
                f"Error: {e}"
            )

        label = int(self.df.loc[idx, "label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class ConvNeuralNet(nn.Module): #SAME AS OUR CNN_FINAL
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.max_pool1(x)

        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.max_pool2(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x


def filter_existing_rows(df, image_folder, image_index, all_files): #not necesarry to run every single time
    kept_rows = []
    missing_rows = []

    for i in range(len(df)):
        # if i % 500 == 0:
            # print(f"Checked {i}/{len(df)} rows...")

        raw = str(df.iloc[i]["image"])
        local_path, matched = resolve_image_path_from_index(
            raw, image_folder, image_index, all_files
        )

        if local_path is None:
            missing_rows.append((raw, matched))
        else:
            kept_rows.append(i)

    filtered_df = df.iloc[kept_rows].reset_index(drop=True)

    # print("Total rows in CSV:", len(df))
    # print("Rows with image found:", len(filtered_df))
    # print("Rows missing image:", len(missing_rows))

    # if len(missing_rows) > 0:
    #     print("\nFirst 20 missing files:")
    #     for raw, matched in missing_rows[:20]:
    #         print("CSV value:", raw)
    #         print("Match result:", matched)
    #         print("-" * 50)

    return filtered_df, missing_rows



csv_path = r"H:\My Drive\Syslab\MergedDataFinal.csv" #YOU ARE REQUIRED TO CHANGE IT TO FIT YOUR CSV
image_folder = r"C:\Users\nickb\Dropbox\Labeled_images" #SAME HERE, CHANGE TO WHERE IT IS LOCATED

dataUsed = pd.read_csv(csv_path)
dataUsed["choice"] = dataUsed["choice"].astype(str).str.strip()

#BUILD INDEX, real images
image_index, all_files = build_image_index(image_folder)
print(f"Indexed {len(all_files)} files from image folder.")


trainingData, testingData = train_test_split(
    dataUsed,
    test_size=0.2,
    random_state=42,
    # stratify=dataUsed["choice"]
)


all_transforms = transforms.Compose([transforms.Lambda(lambda img: black_white_threshold(img)),transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean =[.5, 0.5, 0.5], std = [0.5, 0.5, .5])])

train_set = CVData( trainingData,image_root=image_folder, transform=all_transforms, image_index=image_index, all_files=all_files )

test_set = CVData( testingData,image_root=image_folder, transform=all_transforms, image_index=image_index, all_files=all_files)

# print("Training rows after cleanup:", len(train_set))
# print("Testing rows after cleanup:", len(test_set))

learning_rate = 0.001
batch_size = 64
num_classes = 9 #Forward, right, right up etc
iterations = 25 #NUM OF EPOCHS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

ourCN = ConvNeuralNet(num_classes).to(device) #SET OUR CNN

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    ourCN.parameters(),
    lr=learning_rate,
    weight_decay=0.005,
    momentum=0.9
)

#ACTUAL TRAINING PORTION
for epoch in range(iterations):
    ourCN.train()

    for images, labels in train_loader:
        images = images.to(device)

        labels = labels.to(device)

        outputs = ourCN(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{iterations}, Loss = {loss.item():.4f}")

#TRAINING
ourCN.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = ourCN(images)
        randomVar, classLabel = torch.max(predictions.data, 1)

        total += labels.size(0)
        correct += (classLabel == labels).sum().item()

    print("Training Accuracy =", correct / total)

#Testing ACCURACY
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = ourCN(images)
        randomVar, classLabel = torch.max(predictions.data, 1)

        total += labels.size(0)
        correct += (classLabel == labels).sum().item()

    print("Test Accuracy =", correct / total)

torch.save(ourCN.state_dict(), "CNNweightsFINAL.pth") #CHANGE TO WHAT YOU WANT YOUR CNN WEIGHTS FILE TO BE CALLED
# print("\nModel saved as CNNweightsBW.pth")
# print("Cleaned CSV saved as MergedData.csv")