import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Load the dataset
df = pd.read_csv("DataAnalyst.csv")

# Print basic info
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())
# Drop unwanted columns
df.drop(['Unnamed: 0', 'Founded', 'Competitors'], axis=1, inplace=True, errors='ignore')
df.rename(columns={
    "Job Title": "job_title",
    "Salary Estimate": "salary_estimate",
    "Job Description": "job_description",
    "Company Name": "company_name",
    "Location": "location",
    "Headquarters": "headquarters",
    "Size": "size",
    "Type of ownership": "type_of_ownership",
    "Industry": "industry",
    "Sector": "sector",
    "Revenue": "revenue",
    "Easy Apply": "easy_apply"
}, inplace=True)
# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())
# Extract min and max salary from text like "$37K-$66K"
df[['MinSalary', 'MaxSalary']] = df['salary_estimate'].str.extract(r'\$(\d+)K-\$(\d+)K')
df['MinSalary'] = pd.to_numeric(df['MinSalary'], errors='coerce')
df['MaxSalary'] = pd.to_numeric(df['MaxSalary'], errors='coerce')

# Create average salary
df['average_salary'] = (df['MinSalary'] + df['MaxSalary']) / 2

# Drop original salary column
df.drop(['salary_estimate', 'MinSalary', 'MaxSalary'], axis=1, inplace=True)
df['job_title'] = df['job_title'].str.lower()

df['job_title'] = df['job_title'].replace({
    'sr. data analyst': 'senior data analyst',
    'sr data analyst': 'senior data analyst',
    'data analyst iii': 'senior data analyst',
    'data analyst i': 'junior data analyst',
    'data analyst ii': 'middle data analyst'
})
df.to_csv("cleaned_data.csv", index=False)
print("\nCleaned data saved as 'cleaned_data.csv'")
# phase 3 exploratrory data analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['average_salary'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Average Salary")
plt.xlabel("Average Salary (in $1000s)")
plt.ylabel("Job Count")
plt.grid(True)
plt.show()
top_jobs = df['job_title'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_jobs.values, y=top_jobs.index, palette="crest")
plt.title("Top 10 Job Titles")
plt.xlabel("Number of Jobs")
plt.ylabel("Job Title")
plt.grid(True)
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(x='job_title', y='average_salary', data=df)
plt.xticks(rotation=45)
plt.title("Salary by Job Title")
plt.xlabel("Job Title")
plt.ylabel("Average Salary")
plt.grid(True)
plt.show()
top_locations = df.groupby('location')['average_salary'].mean().sort_values(ascending=False).head(10).reset_index()

fig = px.bar(top_locations, x='average_salary', y='location', orientation='h',
             title='Top 10 Locations by Avg Salary', color='average_salary')
fig.update_layout(xaxis_title='Avg Salary (USD)', yaxis_title='Location', showlegend=False)
fig.show()
filtered_size = df[df['size'].isin(['1 to 50 employees', '51 to 200 employees',
                                    '201 to 500 employees', '501 to 1000 employees',
                                    '1001 to 5000 employees', '5001 to 10000 employees',
                                    '10000+ employees'])]

plt.figure(figsize=(12, 6))
sns.countplot(y='size', data=filtered_size, order=filtered_size['size'].value_counts().index)
plt.title("Company Size Distribution")
plt.xlabel("Count")
plt.ylabel("Company Size")
plt.grid(True)
plt.show()
top_sectors = df['sector'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_sectors.values, y=top_sectors.index, palette="magma")
plt.title("Top 10 Sectors Hiring Data Analysts")
plt.xlabel("Number of Jobs")
plt.ylabel("Sector")
plt.grid(True)
plt.show()
# Only select numeric columns for correlation
numeric_df = df.select_dtypes(include='number')

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix (Numeric Columns Only)")
plt.show()


# phase 4 : ML Modelling
# -----------------------------
# PHASE 4: SALARY PREDICTION MODEL
# -----------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 🔹 STEP 1: Feature Engineering – Skill Extraction
df['has_python'] = df['job_description'].str.contains('python', case=False, na=False).astype(int)
df['has_excel'] = df['job_description'].str.contains('excel', case=False, na=False).astype(int)
df['tech_skill_score'] = df['has_python'] + df['has_excel']

# 🔹 STEP 2: Encode Categorical Features
features_to_encode = ['size', 'type_of_ownership', 'industry', 'sector']
df_encoded = pd.get_dummies(df[features_to_encode], drop_first=True)

# 🔹 STEP 3: Prepare Final Dataset for Modeling
# Combine numerical and encoded features
df_model = pd.concat([df[['Rating', 'tech_skill_score', 'average_salary']], df_encoded], axis=1)

# Drop rows with missing values
df_model.dropna(inplace=True)

# 🔹 STEP 4: Split Data
X = df_model.drop('average_salary', axis=1)
y = df_model['average_salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 STEP 5: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 STEP 6: Predict and Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Improved Mean Absolute Error (MAE): ${mae:.2f}")
print(f"📈 Improved R² Score: {r2:.2f}")
