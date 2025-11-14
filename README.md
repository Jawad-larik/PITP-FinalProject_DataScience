🎓 AI Student Performance Predictor

Predicting Academic Performance with AI

This project is a multi-page Streamlit web application that uses a machine learning model to predict a student's academic performance as Low, Medium, or High. It analyzes factors like test scores, parental education, and test preparation to generate a prediction and explains the reasoning behind it using SHAP.

✨ Key Features

🤖 AI-Powered Predictions: Uses a Random Forest Classifier trained on student data, achieving 96.5% accuracy.

💡 Model Explanations: Integrated SHAP (SHapley Additive exPlanations) to show which factors (e.g., "writing_score" or "lunch") had the biggest impact on each prediction.

📊 Multi-Page Dashboard: A clean, multi-page interface for navigation:

Predict: An interactive form to input student data and get a prediction.

Data Exploration: Visualizes global feature importance and the correlation between test scores.

About: Contains detailed project information.

🎨 Modern UI/UX: Features a custom-styled, animated gradient title, a light/dark mode toggle, and a consistent footer.

📷 App Preview

(Add your screenshots here! Just drag and drop them into your GitHub repo and update the paths.)

Prediction Page

Data Exploration Page





The main prediction page with interactive inputs.

The data exploration page showing feature importance.

🛠️ Tech Stack

Python: Core programming language.

Streamlit: For building the interactive web application.

Scikit-learn: For building the ColumnTransformer pipeline and training the RandomForestClassifier.

Pandas: For data manipulation and a background dataset for SHAP.

SHAP: For model explainability.

Joblib: For saving and loading the trained model pipeline.

Matplotlib & Seaborn: For creating visualizations in the "Data Exploration" page.

📂 File Structure

The project uses a modern multi-page app structure:

AI-Student-Performance-Predictor/
│
├── 1_🚀_Predict.py         # Main app page for predictions
├── style.py               # Shared CSS and footer function
├── requirements.txt       # All Python dependencies
│
├── model_train_file.ipynb # Jupyter Notebook used for model training
├── student_performance_pipeline.joblib  # The pre-trained model file
├── study_performance.csv  # The dataset used for training
│
├── pages/                   # Folder for all other app pages
│   ├── 2_📊_Data_Exploration.py # Data insights page
│   └── 3_ℹ️_About.py           # Project details page
│
└── README.md                # You are here


🚀 How to Run Locally

Follow these steps to run the app on your local machine.

1. Clone the Repository

git clone [https://github.com/YourUsername/AI-Student-Performance-Predictor.git](https://github.com/YourUsername/AI-Student-Performance-Predictor.git)
cd AI-Student-Performance-Predictor


2. Create a Virtual Environment (Recommended)

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies
This project requires the Python packages listed in requirements.txt.

pip install -r requirements.txt


4. Run the Streamlit App
The command will automatically find the pages/ folder and build the sidebar navigation.

streamlit run 1_🚀_Predict.py


Open your browser and go to http://localhost:8501 to use the app!

🧠 Model & Data

Dataset: The model was trained on the Student Performance in Exams dataset from Kaggle.

Model: RandomForestClassifier (within an sklearn.pipeline.Pipeline).

Performance: Achieved 96.5% accuracy on the 20% test split.

Key Features: The model identified the three test scores (writing_score, reading_score, math_score) as the most significant predictors of overall performance.

👨‍💻 Author

Your Name Here

GitHub: [@Jawad-Larik](https://github.com/Jawad-larik)

LinkedIn: [Jawad_larik01]([https://www.linkedin.com/in/jawad-larik01))
