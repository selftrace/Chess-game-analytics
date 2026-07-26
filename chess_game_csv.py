import pandas as pd
import zipfile

zipped_file_path = '/content/games.csv.zip'
csv_file_name = 'games.csv'

with zipfile.ZipFile(zipped_file_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')

df = pd.read_csv(f'/content/{csv_file_name}')


print("\nFirst 5 rows of the dataset:")
display(df.head())

rows, cols = df.shape
print(f"Dataset contains {rows} rows and {cols} columns.")

print("\nColumn names and their data types:")
display(df.info())

missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_info = pd.DataFrame({
    'Missing values': missing_values,
    'Percentage': missing_percentage
})

display(missing_info[missing_info['Missing values'] > 0].sort_values(by='Percentage', ascending=False))

if missing_info['Missing values'].sum() == 0:
    print("No missing values found in the dataset.")

duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

if duplicate_rows > 0:
    print("\nDuplicate rows found. Recommending to drop duplicates.")

else:
    print("No duplicate rows found.")

if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate rows have been removed.")
    rows, cols = df.shape
    print(f"New dataset contains {rows} rows and {cols} columns.")

print("\nDescriptive statistics for numerical columns:")
display(df.describe())

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("\nIdentifying distributions, ranges, and potential outliers for numerical columns:")
for col in numerical_cols:
    print(f"\n--- Column: {col} ---")
    print(f"Range: {df[col].min()} to {df[col].max()}")
    print(f"Mean: {df[col].mean():.2f}")
    print(f"Median: {df[col].median():.2f}")
    print(f"Standard Deviation: {df[col].std():.2f}")

    print(f"Skewness: {df[col].skew():.2f}")
    print(f"Kurtosis: {df[col].kurt():.2f}")

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

    if not outliers.empty:
        print(f"Potential outliers detected (outside 1.5*IQR): {len(outliers)} records. Min outlier: {outliers.min()}, Max outlier: {outliers.max()}")
    else:
        print("No significant outliers detected based on IQR.")

categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

print("\nHighlighting important categorical variable frequencies:")
for col in categorical_cols:
    print(f"\n--- Column: {col} ---")

    if df[col].nunique() > 50:
        print(f"Top 10 most frequent values out of {df[col].nunique()} unique values:")
        display(df[col].value_counts().head(10))
    else:
        print(f"Value counts for {df[col].nunique()} unique values:")
        display(df[col].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

numerical_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation matrix of numerical variables')
plt.show()

print("\nKey correlations to note:")

strong_correlations = correlation_matrix[((correlation_matrix.abs() > 0.7) & (correlation_matrix.abs() < 1.0))].stack().sort_values(ascending=False)
if not strong_correlations.empty:
    display(strong_correlations)
else:
    print("No strong correlations (absolute value > 0.7) found between distinct numerical variables.")

print("\nAverage white and black ratings by victory status:")
display(df.groupby('victory_status')[['white_rating', 'black_rating']].mean())

print("\nAverage white and black ratings by winner:")
display(df.groupby('winner')[['white_rating', 'black_rating']].mean())

print("\nAverage number of turns by victory status:")
display(df.groupby('victory_status')['turns'].mean())

print("\nTop 10 most frequent opening names and their characteristics:")
top_openings = df['opening_name'].value_counts().head(10).index
display(df[df['opening_name'].isin(top_openings)].groupby('opening_name')[['turns', 'white_rating', 'black_rating']].mean().sort_values(by='turns', ascending=False))

df['created_at_dt'] = pd.to_datetime(df['created_at'], unit='ms')
df['last_move_at_dt'] = pd.to_datetime(df['last_move_at'], unit='ms')

print("Converted 'created_at' and 'last_move_at' to datetime objects.")
display(df[['created_at_dt', 'last_move_at_dt']].head())

plt.figure(figsize=(12, 6))
df['created_at_dt'].dt.to_period('M').value_counts().sort_index().plot(kind='line', marker='o')
plt.title('Number of games created over time (monthly)')
plt.xlabel('Date')
plt.ylabel('Number of games')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
df.set_index('created_at_dt').resample('M')[['white_rating', 'black_rating']].mean().plot(kind='line', marker='o')
plt.title('Average white and black ratings over time (monthly)')
plt.xlabel('Date')
plt.ylabel('Average rating')
plt.grid(True)
plt.show()

# histograms

numerical_cols_for_hist = ['turns', 'white_rating', 'black_rating', 'opening_ply']

fig, axes = plt.subplots(len(numerical_cols_for_hist), 1, figsize=(10, 5 * len(numerical_cols_for_hist)))

for i, col in enumerate(numerical_cols_for_hist):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# box plots

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.boxplot(x='victory_status', y='turns', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Turns by victory status')

sns.boxplot(x='winner', y='white_rating', data=df, ax=axes[0, 1])
axes[0, 1].set_title('White rating by winner')

sns.boxplot(x='winner', y='black_rating', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Black rating by winner')

sns.boxplot(x='rated', y='turns', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Turns for rated vs. unrated games')

plt.tight_layout()
plt.show()

categorical_cols_for_bar = ['rated', 'victory_status', 'winner', 'increment_code']

fig, axes = plt.subplots(len(categorical_cols_for_bar), 1, figsize=(10, 5 * len(categorical_cols_for_bar)))

for i, col in enumerate(categorical_cols_for_bar):
    sns.countplot(y=df[col], order=df[col].value_counts().index, ax=axes[i])
    axes[i].set_title(f'Frequency of {col}')
    axes[i].set_xlabel('Count')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 7))
df['opening_name'].value_counts().head(10).plot(kind='barh')
plt.title('Top 10 most frequent opening names')
plt.xlabel('Number of games')
plt.ylabel('Opening name')
plt.gca().invert_yaxis()
plt.show()

