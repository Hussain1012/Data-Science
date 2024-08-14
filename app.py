import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import streamlit as st

# File paths
train_file_path = 'C:/Users/lenovo/Desktop/Excelr/Data Science/Data Science Projects/Data Science Solved Assignments/Titanic_train.csv'
test_file_path = 'C:/Users/lenovo/Desktop/Excelr/Data Science/Data Science Projects/Data Science Solved Assignments/Titanic_test.csv'

# Load the datasets
try:
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

# Data Exploration
st.write("### Data Information")
st.write(train_df.info())

st.write("### Data Description")
st.write(train_df.describe())

# Visualizations
st.write("### Histograms")
fig, ax = plt.subplots(figsize=(15, 10))
train_df.hist(ax=ax)
st.pyplot(fig)

st.write("### Boxplot of Age vs Survival")
fig, ax = plt.subplots()
sns.boxplot(x='Survived', y='Age', data=train_df, ax=ax)
st.pyplot(fig)

st.write("### Pairplot")
fig = sns.pairplot(train_df, hue='Survived')
st.pyplot(fig)

# Data Preprocessing
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Encode categorical variables using one-hot encoding
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)

# Ensure both train and test datasets have the same columns
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    if col != 'Survived':  # Don't add the target column to test dataset
        test_df[col] = 0

# Align the order of columns in the test set with the train set
test_df = test_df[train_df.drop(columns=['Survived']).columns]

# Drop unnecessary columns
X = train_df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'])
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_df_scaled = scaler.transform(test_df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId']))

model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_val)
y_pred_prob = model.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_prob)

st.write("### Model Evaluation Metrics")
st.write(f'**Accuracy:** {accuracy}')
st.write(f'**Precision:** {precision}')
st.write(f'**Recall:** {recall}')
st.write(f'**F1-Score:** {f1}')
st.write(f'**ROC AUC Score:** {roc_auc}')

fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
st.write("### ROC Curve")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label='ROC curve')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
st.pyplot(fig)

# Interpretation of Coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
st.write("### Model Coefficients")
st.write(coefficients)

# Streamlit Deployment
st.title('Titanic Survival Prediction')

Pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
Age = st.slider('Age', 0, 100, 25)
SibSp = st.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
Parch = st.slider('Number of Parents/Children Aboard', 0, 6, 0)
Fare = st.slider('Fare', 0, 500, 30)
Sex_male = st.selectbox('Sex', ['Male', 'Female']) == 'Male'
Embarked_Q = st.selectbox('Port of Embarkation (Q = Queenstown)', ['Yes', 'No']) == 'Yes'
Embarked_S = st.selectbox('Port of Embarkation (S = Southampton)', ['Yes', 'No']) == 'Yes'

input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Sex_male': [Sex_male],
    'Embarked_Q': [Embarked_Q],
    'Embarked_S': [Embarked_S]
})

# Ensure the input data has the same columns as the training data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

if prediction[0] == 1:
    st.write("The passenger is likely to survive.")
else:
    st.write("The passenger is unlikely to survive.")
