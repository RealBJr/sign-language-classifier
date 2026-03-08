# sign-language-classifier
In the context of Concordia's COMP-472, this project tackles communication barriers faced by the deaf community in public spaces, where limited sign language proficiency can cause exclusion. Building on neural translation research, it develops a classifier for faster, more natural interaction than written communication.

## 2 Ways of Running the Notebook:
- Colab (click on the link at the top of the notebook) (slow for data extraction)
- Locally (follow the steps below) (faster for data extraction)

## Local Setup Guide

### Prerequisites

- **Python 3.10+**
- **VSCode** with the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- A **Kaggle account** (free) for downloading datasets

### 1. Clone the Repository

```bash
git clone https://github.com/RealBJr/sign-language-classifier.git
cd sign-language-classifier
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### 3. Install Dependencies

Install the packages:

```bash
pip install -r requirements.txt
```

Your requirements.txt should contain:

```
torch
torchvision
opencv-python
kagglehub
matplotlib
pandas
```

### 4. Set Up Kaggle Credentials

The notebook uses kagglehub to download datasets. It reads credentials from a kaggle.json file.

1. Go to [kaggle.com](https://www.kaggle.com) -> click your profile icon -> **Settings**
2. Scroll to the **API** section -> click **Create New Token**
3. This downloads a kaggle.json file that looks like:

```json
{"username":"your_username","key":"your_api_key_here"}
```
4. Place this file at:
   - ../.kaggle/kaggle.json (Under the .kaggle folder in the repo)

### 5. Open the Notebook in VSCode

1. Open the project folder in VSCode
2. Open model.ipynb
3. In the top-right corner, click **Select Kernel** -> choose your .venv Python interpreter
4. If prompted about the Jupyter extension, install it

### 6. Running the Notebook

The notebook has two parallel tracks — **Colab** sections and **Local** sections. When running locally:

- **Skip** sections labeled (Collab):
  - 1) Environment Setup (Collab)
  - 2) Data (Collab)

- **Run** sections labeled (Local):
  - 1) Environment Setup (Local) — downloads datasets via kagglehub, saves to a local SignLanguageProject/ folder, and verifies GPU access
  - 2) Data (Local) — processes all three datasets, creates DataLoaders, generates an info table, and visualizes class distributions and sample images

- **Section 3 onward** (Training, Evaluation) is shared between both environments — run as normal.