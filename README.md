🏥 Disease Symptom Prediction

Predicts probable diseases from patient-reported symptoms by benchmarking five ML classifiers — built with Python and scikit-learn.


📋 Description:
This project takes binary symptom inputs and predicts the most likely disease using a supervised machine learning pipeline. Five classification algorithms are trained and evaluated head-to-head to identify which model generalizes best on unseen patient data. It demonstrates a complete ML workflow from data preprocessing through model evaluation.
Technologies: Python · Pandas · NumPy · scikit-learn

⚠️ For educational purposes only. Not intended for real medical diagnosis.

⚙️ Installation
1. Clone the repository
git clone https://github.com/Amnaa-Zahidd-30/Machine-Learning.git
cd Machine-Learning

2. Install dependencies
pip install pandas numpy scikit-learn

Requirements: Python 3.7+


🚀 Usage:
Run the full training and evaluation pipeline
python "Disease-Symptom_Prediction ML.py"
The script will:

Load and preprocess dataset_training.csv
Train all five classifiers
Print accuracy, precision, recall, and F1-score for each model


✨ Features:

Benchmarks 5 classifiers: Logistic Regression, Random Forest, SVM, KNN, SGD
Complete preprocessing pipeline — missing value handling, feature encoding, train/test split
Weighted evaluation metrics per classifier for fair multi-class comparison
Structured symptom dataset with binary feature flags as inputs
Modular code — easy to swap in new classifiers or datasets


🤝 Contributing
Contributions are welcome!

Fork the repository
Create a new branch: git checkout -b feature/your-feature
Commit your changes: git commit -m 'Add your feature'
Push and open a Pull Request

📬 Contact
amnazahid894@gmail.com
