import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff

# load the dataset
data, meta = arff.loadarff('data/Autism-Adult-Data.arff')
df = pd.DataFrame(data)

# decode byte strings to regular strings
for column in df.select_dtypes([object]):
    df[column] = df[column].str.decode('utf-8')

# handle missing values
df.replace('?', pd.NA, inplace=True)  # Replace '?' with NaN
df.dropna(inplace=True)

# rename columns for clarity
df.rename(columns={
    'A1_Score': 'Question_1', 'A2_Score': 'Question_2', 'A3_Score': 'Question_3',
    'A4_Score': 'Question_4', 'A5_Score': 'Question_5', 'A6_Score': 'Question_6',
    'A7_Score': 'Question_7', 'A8_Score': 'Question_8', 'A9_Score': 'Question_9',
    'A10_Score': 'Question_10',
    'jundice': 'Born_with_jaundice',
    'austim': 'Family_member_with_ASD',
    'contry_of_res': 'Country_of_residence',
    'used_app_before': 'Used_screening_app',
    'result': 'Screening_result',
    'relation': 'Relation_to_test',
    'Class/ASD': 'ASD_Class'
}, inplace=True)

# encode boolean and binary values
binary_columns = ['Born_with_jaundice', 'Family_member_with_ASD', 'Used_screening_app', 'ASD_Class']
df[binary_columns] = df[binary_columns].applymap(lambda x: 1 if x.lower() == 'yes' else 0)

# drop redundant columns
df.drop(columns=['age_desc'], inplace=True)  # 'age_desc' seems redundant given 'age'

# convert numeric columns to proper types
df['age'] = pd.to_numeric(df['age'])
df['Screening_result'] = pd.to_numeric(df['Screening_result'])
behavioral_questions = [f'Question_{i}' for i in range(1, 11)]
for col in behavioral_questions:
    df[col] = pd.to_numeric(df[col])

# visualize data
# age distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# gender distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='gender', data=df)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# correlation heatmap
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# pair plot for questions
sns.pairplot(
    df,
    vars=behavioral_questions,
    hue='ASD_Class',
    diag_kind='kde',
    corner=True,
    height=0.9
)
plt.suptitle('Pair plot of Behavioral Questions', y=1.02)
plt.show()
