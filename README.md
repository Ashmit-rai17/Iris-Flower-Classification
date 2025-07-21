# Iris Flower Classification Web App

This project builds a machine learning model to classify Iris flower species using the classic Iris dataset, and provides a Streamlit web app for interactive predictions.

## Project Structure

- `data/` — Contains the raw and processed Iris dataset.
- `models/` — Stores the trained machine learning model.
- `app/` — Contains the Streamlit web application.
- `main.py` — Script for data preprocessing, EDA, and model training.
- `requirements.txt` — List of required Python packages.

## Setup Instructions

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download and preprocess the data, run EDA, and train the model**

   ```bash
   python main.py
   ```
   This will:
   - Download and preprocess the Iris dataset
   - Save EDA visualizations in `data/`
   - Train a Random Forest model and save it in `models/`

3. **Run the Streamlit web app**

   ```bash
   cd app
   streamlit run iris_app.py
   ```

4. **Use the app**
   - Enter the flower measurements to predict the species.

## Notes
- EDA plots are saved in `data/` as PNG files.
- The trained model is saved as `models/iris_rf_model.joblib`.

## Requirements
See `requirements.txt` for all dependencies. 