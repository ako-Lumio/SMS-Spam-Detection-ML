# SMS Spam Detection Using Machine Learning

## Project Overview
This project uses Machine Learning to classify SMS messages as **Spam** or **Ham (Not Spam)**. It provides a simple web interface built using Streamlit, allowing users to enter a message and receive a prediction.

## Objectives
- Detect unwanted spam messages.
- Preprocess and analyse SMS text data.
- Train and compare multiple Machine Learning algorithms.
- Build an interactive spam detection application.

## Dataset
**SMS Spam Collection Dataset**

Source: [Kaggle – SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

Download `spam.csv` and place it inside the `data` folder.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- TF-IDF Vectorization

## Machine Learning Algorithms
1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Classifier (Linear SVC)

## Project Structure

```text
SMS-Spam-Detection/
├── data/
├── models/
├── app.py
├── train.py
├── check_data.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models

```bash
python train.py
```

### 3. Launch the application

```bash
python -m streamlit run app.py
```

## Features
- SMS message classification
- Comparison of three ML algorithms
- Model evaluation and visualisation
- Interactive Streamlit interface

## Author
Amrendar Kaur Oberoi

B.Tech – Artificial Intelligence and Data Science