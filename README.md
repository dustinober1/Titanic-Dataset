# 🚢 Titanic Survival Prediction - Machine Learning Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning project analyzing the famous Titanic dataset to predict passenger survival rates. This project demonstrates advanced data science techniques, from exploratory data analysis to interactive prediction systems.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Model Performance](#model-performance)
- [Interactive Demo](#interactive-demo)
- [Technologies Used](#technologies-used)
- [Key Insights](#key-insights)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project provides a complete end-to-end machine learning solution for predicting Titanic passenger survival. The analysis reveals fascinating insights into how social class, gender, age, and family dynamics influenced survival during one of history's most famous maritime disasters.

**Live Demo:** Try the interactive predictor to see if you would have survived the Titanic!

## ✨ Key Features

### 📊 Comprehensive Data Analysis
- **12-section EDA** with professional visualizations
- **Statistical analysis** of survival patterns
- **Historical context** and social implications
- **25+ visualizations** including heatmaps, correlation matrices, and distribution plots

### 🤖 Advanced Machine Learning
- **5 algorithm comparison** (Random Forest, Gradient Boosting, SVM, Logistic Regression, Naive Bayes)
- **Hyperparameter optimization** with GridSearchCV
- **Cross-validation** for robust performance estimates
- **Feature importance analysis** with detailed explanations

### 🎮 Interactive Components
- **Real-time prediction interface** for user input
- **Command-line predictor tool** for quick testing
- **Detailed explanations** of prediction factors
- **Multiple example scenarios** included

### 🔧 Production-Ready Code
- **Error handling** and input validation
- **Model persistence** with pickle serialization
- **Clean, documented code** following best practices
- **Modular design** for easy extension

## 📁 Project Structure

```
Titanic-Dataset/
├── README.md                     # This file
├── Titanic-Dataset.csv          # Original dataset
├── Titanic_EDA.ipynb            # Comprehensive exploratory data analysis
├── Titanic_ML_Predictor.ipynb   # Machine learning model and prediction system
├── titanic_predictor.py         # Command-line prediction tool
├── titanic_survival_model.pkl   # Trained model (generated after running)
└── requirements.txt             # Python dependencies
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/titanic-ml-analysis.git
   cd titanic-ml-analysis
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv titanic_env
   source titanic_env/bin/activate  # On Windows: titanic_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

### Required Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## 💻 Usage

### 1. Exploratory Data Analysis
Open and run `Titanic_EDA.ipynb` to explore:
- Dataset overview and statistics
- Missing value analysis
- Survival pattern visualization
- Feature correlation analysis
- Historical insights and context

### 2. Machine Learning Model
Run `Titanic_ML_Predictor.ipynb` to:
- Train and compare multiple ML models
- Perform hyperparameter optimization
- Evaluate model performance
- Use the interactive prediction interface

### 3. Command-Line Predictor
For quick predictions without Jupyter:
```bash
python titanic_predictor.py
```

### 4. Interactive Prediction Example
```python
# In Jupyter notebook after running ML cells
create_prediction_interface()

# Or use the function directly
probability, prediction = predict_survival(
    pclass=1,        # First class
    sex='female',    # Female
    age=25,          # 25 years old
    sibsp=0,         # No siblings/spouse
    parch=0,         # No parents/children
    fare=80,         # High fare
    embarked='S'     # Southampton
)
print(f"Survival probability: {probability:.1%}")
```

## 🔬 Methodology

### Data Preprocessing
1. **Missing Value Treatment**
   - Age: Imputed using median by title and class
   - Embarked: Filled with mode (Southampton)
   - Cabin: Created binary feature for presence

2. **Feature Engineering**
   - **Family size analysis** (optimal: 2-4 people)
   - **Title extraction** from names (Mr, Mrs, Miss, Master, Rare)
   - **Age groups** (Child, Teen, Young Adult, Adult, Senior)
   - **Fare categories** (Very Low to Very High)
   - **Economic indicators** (fare per person, high fare binary)

3. **Categorical Encoding**
   - Label encoding for ordinal features
   - One-hot encoding considerations for nominal features

### Model Selection Process
1. **Baseline comparison** of 5 algorithms
2. **Cross-validation** (5-fold) for robust evaluation
3. **Hyperparameter tuning** with GridSearchCV
4. **Feature importance** analysis
5. **Final model selection** based on CV accuracy

### Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **ROC-AUC**: Model's ability to distinguish classes
- **Precision/Recall**: Detailed performance analysis
- **Confusion Matrix**: Classification breakdown

## 📈 Results

### Model Performance
| Model | Accuracy | ROC-AUC | CV Score |
|-------|----------|---------|----------|
| **Random Forest** | **83.8%** | **0.84** | **81.5%** |
| Gradient Boosting | 82.1% | 0.83 | 80.8% |
| Logistic Regression | 80.4% | 0.82 | 79.9% |
| SVM | 79.9% | 0.81 | 79.2% |
| Naive Bayes | 78.2% | 0.79 | 77.8% |

### Feature Importance (Random Forest)
1. **Sex** (27.4%) - Gender was the strongest predictor
2. **Fare** (17.0%) - Economic status significantly impacted survival
3. **Title** (12.4%) - Social titles reflected survival patterns
4. **Age** (11.2%) - Age influenced evacuation priority
5. **Class** (10.0%) - Passenger class determined ship location

## 🎮 Interactive Demo

The project includes an interactive prediction system where users can input their personal characteristics:

**Example Predictions:**
- **Wealthy young woman** (1st class, female, age 25): ✅ **98.8% survival**
- **Working class father** (3rd class, male, age 30, family): ❌ **12.2% survival**
- **Middle class mother** (2nd class, female, age 35, children): ✅ **85.8% survival**
- **Poor young man** (3rd class, male, age 22, alone): ❌ **7.5% survival**

## 🛠 Technologies Used

- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment
- **Git** - Version control

## 🔍 Key Insights

### Historical Findings
1. **"Women and children first" protocol strictly followed**
   - Female survival rate: 74.2%
   - Male survival rate: 18.9%
   - Children (0-12): 58.0% survival rate

2. **Social class created survival hierarchy**
   - 1st Class: 62.9% survival
   - 2nd Class: 47.3% survival
   - 3rd Class: 24.2% survival

3. **Economic inequality directly impacted survival**
   - High fare passengers (>$100): 75.9% survival
   - Low fare passengers (<$10): 25% survival
   - 6x difference in average fare between classes

4. **Family dynamics influenced evacuation success**
   - Solo travelers: 30.4% survival
   - Small families (2-4): 57.8% survival
   - Large families (5+): 16.1% survival

### Technical Insights
- **Gender was 60% stronger predictor than class**
- **Random Forest achieved best performance** through ensemble methods
- **Feature engineering improved accuracy by ~5%**
- **Cross-validation prevented overfitting** in model selection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **Kaggle** for providing the Titanic dataset
- **scikit-learn** community for excellent ML tools
- **Pandas** and **NumPy** for powerful data manipulation
- **Matplotlib** and **Seaborn** for beautiful visualizations
