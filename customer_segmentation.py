import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("styles.csv", on_bad_lines="skip")

# Select useful columns
df = df[['gender',
         'masterCategory',
         'subCategory',
         'season',
         'usage']]

# Remove missing values
df = df.dropna()

# Convert text to numbers
le = LabelEncoder()

df['gender_enc'] = le.fit_transform(df['gender'])
df['category_enc'] = le.fit_transform(df['masterCategory'])
df['subcat_enc'] = le.fit_transform(df['subCategory'])
df['season_enc'] = le.fit_transform(df['season'])
df['usage_enc'] = le.fit_transform(df['usage'])

# Features for clustering
X = df[['gender_enc',
        'category_enc',
        'subcat_enc',
        'season_enc',
        'usage_enc']]

# Create 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)

df['Cluster'] = kmeans.fit_predict(X)

# Save output
df.to_csv("segmented_customers.csv", index=False)

print("Segmentation Completed!")
print(df[['gender','masterCategory','usage','Cluster']].head())