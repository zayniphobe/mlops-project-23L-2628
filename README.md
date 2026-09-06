# 23L-2628_MLOps_Assg1
Assignment 1 for the course MLOps (FAST NUCES). This is a minimal, reproducible MLOps workflow for house price prediction, built with scikit-learn and version-controlled with Git/GitHub.

## Project Structure

```
├── data/                     # raw dataset (ignored by git)
│   └── dataset.csv
├── src/
│   └── train_23L_2628.py   # training script
├── model/                    # trained model output (ignored by git)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/zayniphobe/23L-2628_MLOps_Assg1.git
   cd 23L-2628_MLOps_Assg1
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .
   source 23L-2628_MLOps_Assg1/bin/activate      # macOS/Linux
   23L-2628_MLOps_Assg1\Scripts\activate         # Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Place your dataset at `data/dataset.csv`.

## Running the Training Script

From the project root:

```
python src/train_23L_2628.py
```

This will:
- Load `data/dataset.csv`
- Train a `RandomForestRegressor`
- Print MAE and R² evaluation metrics
- Save the trained model to `model/model_STUDENT_ID.pkl`

## Additional Notes

- `data/` and `model/` are excluded from version control via `.gitignore` to
  keep the repository lightweight. Only source code and configuration files
  (`.py`, `.txt`, `.md`, `.gitignore`) are committed.
