## Business Case: Netflix - Data Exploration and Visualisation
### Analyse the data and generate insights that could help Netflix in deciding which type of shows/movies to produce and how they can grow the business in different countries

Dataset Link: https://github.com/Jyotiprakash01/Netflix-Data-Exploration-and-Visualisation/blob/main/netflix.csv


The dataset provided consists of a list of all the TV shows/movies available on Netflix:
- Show_id: Unique ID for every Movie / Tv Show
- Type: Identifier - A Movie or TV Show
- Title: Title of the Movie / Tv Show
- Director: Director of the Movie
- Cast: Actors involved in the movie/show
- Country: Country where the movie/show was produced
- Date_added: Date it was added on Netflix
- Release_year: Actual Release year of the movie/show
- Rating: TV Rating of the movie/show
- Duration: Total Duration - in minutes or number of seasons
- Listed_in: Genre
- Description: The summary description

## 1. Problem Statement

Netflix aims to optimize its content strategy by understanding:

- Content Preferences: Which types of movies/TV shows perform best?
- Global Expansion: How to tailor content for different countries?
- Trends Over Time: How has content production evolved?
- Optimal Release Timing: When should new shows/movies be launched?
- Key Talent: Which actors/directors are most frequent in successful content?

The goal is to provide data-driven recommendations to guide Netflix's content production and business growth.

## 2. Tools Used :
- Excel & Python - Data Cleaning & Visualization 

## 3. Data Analysis:

```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
## Loading the netflix dataset into a pandas DataFrame and display the first 5 rows.

```Python
import pandas as pd

netflix_df = pd.read_csv('netflix.csv')
display(netflix_df.head())
```
![image](https://github.com/user-attachments/assets/43f02930-2073-4d1e-8bfe-6aa78257af8e)

## Examining shape, descriptive statistics, missing values, data types, and unique values for categorical columns.

```Python
print("Data Shape:", netflix_df.shape)

print("\nDescriptive Statistics:\n", netflix_df.describe(include='number'))

missing_values = netflix_df.isnull().sum()
missing_percentage = (missing_values / len(netflix_df)) * 100
print("\nMissing Values:\n", missing_values)
print("\nMissing Value Percentage:\n", missing_percentage)

print("\nData Types:\n", netflix_df.dtypes)
categorical_cols = netflix_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values and counts for '{col}':\n{netflix_df[col].value_counts()}")
```

![image](https://github.com/user-attachments/assets/4b56490c-1967-48ac-b14f-095be4f3fd33)
![image](https://github.com/user-attachments/assets/8df905fb-d7e1-4065-8cb7-e186eb99a5c4)
![image](https://github.com/user-attachments/assets/38d39b51-688c-4a6c-91df-32b584384deb)
![image](https://github.com/user-attachments/assets/d395f7d3-0489-488d-8bf4-5c73aa715a22)
![image](https://github.com/user-attachments/assets/189d2598-3a41-4c2e-ae91-0cb2e63f172b)
![image](https://github.com/user-attachments/assets/12c106be-ff94-4b9e-8828-2f80a6b24fba)
![image](https://github.com/user-attachments/assets/4e9d3efc-b73f-4175-b4c3-5aa87280f8f3)

## Cleaning the data by handling missing values in the DataFrame.

```Python
import numpy as np

categorical_cols = ['director', 'cast', 'country', 'rating', 'date_added', 'listed_in']
for col in categorical_cols:
    if netflix_df[col].dtype == 'object':
        netflix_df[col] = netflix_df[col].fillna("Unknown")
    elif col == 'date_added':
        netflix_df[col] = pd.to_datetime(netflix_df[col], errors='coerce')

import pandas as pd

if 'duration' in netflix_df.columns:
    netflix_df['duration_int'] = netflix_df['duration'].str.extract('(\d+)').fillna(0).astype(int)
    netflix_df['duration_type'] = netflix_df['duration'].str.extract('(min|Season|Seasons)').fillna('Unknown')
    netflix_df.drop('duration', axis=1, inplace=True)

netflix_df['date_added'] = pd.to_datetime(netflix_df['date_added'], errors='coerce')
netflix_df['date_added'] = netflix_df['date_added'].fillna(pd.to_datetime('1900-01-01'))

netflix_df['release_week'] = netflix_df['date_added'].dt.isocalendar().week
netflix_df['release_month'] = netflix_df['date_added'].dt.month

netflix_df['time_difference'] = (pd.to_datetime('2025-04-26') - netflix_df['date_added']).dt.days
netflix_df['time_difference'] = netflix_df['time_difference'].fillna(0).astype(int)

display(netflix_df.head())
print("\nDescriptive Statistics:\n", netflix_df.describe())

missing_values = netflix_df.isnull().sum()
missing_percentage = (missing_values / len(netflix_df)) * 100
print("\nMissing Values:\n", missing_values)
print("\nMissing Value Percentage:\n", missing_percentage)
```
![image](https://github.com/user-attachments/assets/a6186b92-5b8d-4221-93b0-bf0c8a101213)
![image](https://github.com/user-attachments/assets/bcd8cb10-216f-44d8-ac6a-aa01602104f3)
![image](https://github.com/user-attachments/assets/0a230b18-d4b7-4e70-a3fb-6d929334abcd)

## Unnesting the comma-separated values in the 'cast', 'country', 'listed_in', and 'director' columns of the netflix_df DataFrame. Creating new rows for each value in these columns.

```Python
import pandas as pd

columns_to_unnest = ['cast', 'country', 'listed_in', 'director']

for col in columns_to_unnest:
    netflix_df[col] = netflix_df[col].astype(str).str.split(',')
    netflix_df = netflix_df.explode(col)
    netflix_df[col] = netflix_df[col].str.strip()

netflix_df = netflix_df.reset_index(drop=True)

print(netflix_df.shape)
display(netflix_df.head())
print(netflix_df['cast'].unique())
print(netflix_df['country'].unique())
print(netflix_df['listed_in'].unique())
print(netflix_df['director'].unique())
```
![image](https://github.com/user-attachments/assets/f7478050-bc30-4760-b75e-3683e33b54a1)
![image](https://github.com/user-attachments/assets/562670ca-3a83-45a7-aa23-649d3907a203)


## Handling the 'duration' column, convert 'date_added' to datetime, create 'release_week' and 'release_month' columns, and calculate the 'time_difference'.

```Python
import pandas as pd

if 'duration' in netflix_df.columns:
    netflix_df['duration_int'] = netflix_df['duration'].str.extract('(\d+)').fillna(0).astype(int)
    netflix_df['duration_type'] = netflix_df['duration'].str.extract('(min|Season|Seasons)').fillna('Unknown')
    netflix_df.drop('duration', axis=1, inplace=True)
else:
    print("The 'duration' column has already been processed.")

netflix_df['date_added'] = pd.to_datetime(netflix_df['date_added'], errors='coerce')
netflix_df['date_added'] = netflix_df['date_added'].fillna(pd.to_datetime('1900-01-01'))

netflix_df['release_week'] = netflix_df['date_added'].dt.isocalendar().week
netflix_df['release_month'] = netflix_df['date_added'].dt.month

netflix_df['time_difference'] = (pd.to_datetime('2025-04-26') - netflix_df['date_added']).dt.days
netflix_df['time_difference'] = netflix_df['time_difference'].fillna(0).astype(int)

display(netflix_df.head())
```
![image](https://github.com/user-attachments/assets/c02fa9a3-3fc6-41ab-8c4d-6086f0451bcf)

```python
# Best Release Week
best_release_week = netflix_df.groupby(['release_week', 'type']).size().reset_index(name='counts')
max_counts_week = best_release_week.groupby('type')['counts'].max()
print("Best Release Week:\n", max_counts_week)

# Best Release Month
best_release_month = netflix_df.groupby(['release_month', 'type']).size().reset_index(name='counts')
max_counts_month = best_release_month.groupby('type')['counts'].max()
print("\nBest Release Month:\n", max_counts_month)


# Top 10 Countries for Movie Production
top_10_movie_countries = netflix_df[netflix_df['type'] == 'Movie'].groupby('country')['title'].nunique().sort_values(ascending=False).head(10)
print("\nTop 10 Movie Production Countries:\n", top_10_movie_countries)


# Top 10 Countries for TV Show Production
top_10_tvshow_countries = netflix_df[netflix_df['type'] == 'TV Show'].groupby('country')['title'].nunique().sort_values(ascending=False).head(10)
print("\nTop 10 TV Show Production Countries:\n", top_10_tvshow_countries)
```
![image](https://github.com/user-attachments/assets/7fc65acc-d223-4db1-ace2-30f2deac01ac)

## Univariate analysis for continuous variables

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Get the counts of each content type
content_type_counts = df_netflix['type'].value_counts()

# Create the pie chart
plt.figure(figsize=(6, 6))  # Adjust figure size if needed
plt.pie(content_type_counts, labels=content_type_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Netflix Content Type Distribution')
plt.show()

# Distplot for 'release_year'
sns.displot(df_netflix['release_year'], kde=True)
plt.title('Distribution of Release Year')
plt.show()

# Histogram for 'duration_int' (assuming this is the numerical duration)
plt.hist(df_netflix['duration_int'], bins=20)
plt.title('Distribution of Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/user-attachments/assets/ff1d9c76-7cf4-4872-a4eb-b2a43d642ac4)
![image](https://github.com/user-attachments/assets/15486e93-202c-4864-809d-d041c8181eae)
![image](https://github.com/user-attachments/assets/29094304-e5f3-4a1d-89e2-bb62f90084a1)

## Boxplot for categorical variables

```python
plt.figure(figsize=(12, 6))
sns.boxplot(x='rating', y='duration_int', data=df_netflix)
plt.title('Duration Distribution by Rating')
plt.xticks(rotation=45, ha='right') 
plt.show()
```
![image](https://github.com/user-attachments/assets/894e692c-ae4b-49ef-be38-47592bfea00f)

## Correlation analysis using heatmaps and pairplots

```python
import seaborn as sns
import matplotlib.pyplot as plt

df_netflix['date_added'] = pd.to_datetime(df_netflix['date_added'], errors='coerce')
df_netflix['time_difference'] = (pd.to_datetime('2025-04-26') - df_netflix['date_added']).dt.days
df_netflix['time_difference'] = df_netflix['time_difference'].fillna(0).astype(int)

numerical_cols = ['release_year', 'duration_int', 'time_difference'] 
correlation_matrix = df_netflix[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
![image](https://github.com/user-attachments/assets/2c722209-0fc0-4b4e-9acf-7bff84eab9bf)

## Generate a word cloud from the 'listed_in' column and find the mode of the 'time_difference' column.

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Word cloud for 'listed_in'
text = ' '.join(netflix_df['listed_in'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Genres')
plt.show()

# 2. Mode of 'time_difference'
mode_time_difference = netflix_df['time_difference'].mode()[0]
print(f"The mode of the time difference is: {mode_time_difference} days")
```
![image](https://github.com/user-attachments/assets/2127d4d3-53d6-49ba-bec4-4909112801dc)

```python
import plotly.express as px

# Group by country and count titles
country_counts = df_netflix.groupby('country')['title'].nunique().reset_index()

# Create the choropleth map
fig = px.choropleth(
    country_counts,
    locations='country',
    locationmode='country names',
    color='title',
    hover_name='country',
    color_continuous_scale='Viridis',  # Choose a color scale
    title='Netflix Content Contribution by Country'
)

fig.show()
```
![image](https://github.com/user-attachments/assets/1379193c-8084-4a99-bbaa-d4031d8a4bc0)

```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Word Cloud for Movie Genres
plt.figure(figsize=(10, 5))
text = ' '.join(netflix_df['listed_in'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Genres')
plt.savefig('wordcloud_genres.png') # save the plot as a png image
plt.show()


# Top 10 Countries (Movies and TV Shows)
plt.figure(figsize=(12, 6))

# Top 10 Movies
top_10_movies = netflix_df[netflix_df['type'] == 'Movie'].groupby('country')['title'].nunique().sort_values(ascending=False).head(10)
plt.subplot(1, 2, 1)
plt.bar(top_10_movies.index, top_10_movies.values, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Countries for Movie Production')
plt.xlabel('Country')
plt.ylabel('Number of Movies')


# Top 10 TV Shows
top_10_tvshows = netflix_df[netflix_df['type'] == 'TV Show'].groupby('country')['title'].nunique().sort_values(ascending=False).head(10)
plt.subplot(1, 2, 2)
plt.bar(top_10_tvshows.index, top_10_tvshows.values, color='lightcoral')
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Countries for TV Show Production')
plt.xlabel('Country')
plt.ylabel('Number of TV Shows')

plt.tight_layout()
plt.savefig('top_countries_production.png') # save the plot as a png image
plt.show()


# Get top 20 genres
top_20_genres = df_netflix['listed_in'].value_counts().head(20)

# Create count plot
plt.figure(figsize=(12, 6))  # Adjust figure size as needed
sns.countplot(y='listed_in', data=df_netflix, order=top_20_genres.index, palette='viridis')  # Use 'palette' for color customization
plt.title('Top 20 Genres on Netflix')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()


# Count Plots for Categorical Variables
plt.figure(figsize=(15, 5))

# Rating
plt.subplot(1, 3, 1)
netflix_df['rating'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')

# Type
plt.subplot(1, 3, 2)
netflix_df['type'].value_counts().plot(kind='bar', color='gold')
plt.title('Distribution of Content Type')
plt.xlabel('Type')
plt.ylabel('Count')

# Duration Type
plt.subplot(1, 3, 3)
netflix_df['duration_type'].value_counts().plot(kind='bar', color='lightblue')
plt.title('Distribution of Duration Type')
plt.xlabel('Duration Type')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('categorical_distributions.png')  # save the plot as a png image
plt.show()


# Best Release Weeks and Months
plt.figure(figsize=(15, 6))

# Best Release Week
plt.subplot(1, 2, 1)
best_release_week = netflix_df.groupby(['release_week', 'type']).size().reset_index(name='counts')
for t in best_release_week['type'].unique():
    subset = best_release_week[best_release_week['type'] == t]
    plt.plot(subset['release_week'], subset['counts'], label=t, marker='o')
plt.legend()
plt.xlabel('Release Week')
plt.ylabel('Number of Releases')
plt.title('Best Release Week for Movies and TV Shows')


# Best Release Month
plt.subplot(1, 2, 2)
best_release_month = netflix_df.groupby(['release_month', 'type']).size().reset_index(name='counts')
for t in best_release_month['type'].unique():
    subset = best_release_month[best_release_month['type'] == t]
    plt.plot(subset['release_month'], subset['counts'], label=t, marker='o')
plt.legend()
plt.xlabel('Release Month')
plt.ylabel('Number of Releases')
plt.title('Best Release Month for Movies and TV Shows')

plt.tight_layout()
plt.savefig('best_release_time.png') # save the plot as a png image
plt.show()


# Time Difference Distribution
plt.figure(figsize=(10, 5))
plt.hist(netflix_df['time_difference'], bins=30, color='orange', edgecolor='black')
plt.title('Distribution of Time Difference (Release to Netflix)')
plt.xlabel('Time Difference (Days)')
plt.ylabel('Frequency')
plt.savefig('time_difference_histogram.png') # save the plot as a png image
plt.show()

# Filter data for movies and TV shows
movies = df_netflix[df_netflix['type'] == 'Movie']
tv_shows = df_netflix[df_netflix['type'] == 'TV Show']

# Create box plot
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
sns.boxplot(x='type', y='duration_int', data=df_netflix, palette='Set2')  # Use 'palette' for color customization
plt.title('Duration Distribution for Movies and TV Shows')
plt.xlabel('Content Type')
plt.ylabel('Duration (minutes/seasons)')
plt.show()

# Filter data for movies only
movies = netflix_df[netflix_df['type'] == 'Movie']

# Create boxplot for movie durations
plt.figure(figsize=(8, 6))
sns.boxplot(x=movies['duration_int'], color='skyblue')  # Adjust color as needed
plt.title('Boxplot of Movie Duration')
plt.xlabel('Duration (minutes)')
plt.show()

# Create genre presence matrix
genre_presence = pd.get_dummies(df_netflix['listed_in']).groupby(df_netflix['show_id']).sum()

# Calculate correlation matrix
genre_correlation = genre_presence.corr()

# Create heatmap
plt.figure(figsize=(12, 10))  # Adjust figure size as needed
sns.heatmap(genre_correlation, cmap='coolwarm', annot=False)  # Set annot=True for values
plt.title('Genre Correlation Heatmap')
plt.show()
```
![image](https://github.com/user-attachments/assets/ec59eef9-5fea-4d77-a0e2-d4c89990766c)
![image](https://github.com/user-attachments/assets/448dbca2-303f-40d7-9cf0-3d639d6927ba)
![image](https://github.com/user-attachments/assets/21d081fd-cf60-4109-8aec-a1b914b76bdb)
![image](https://github.com/user-attachments/assets/e25bb2f0-0563-40cd-8e9d-5b01656aa1a7)
![image](https://github.com/user-attachments/assets/4ff8b2b1-3316-4958-83fb-66ea00d6a683)
![image](https://github.com/user-attachments/assets/86757106-8361-4030-85ba-1a2915645564)
![image](https://github.com/user-attachments/assets/03c18e39-1c41-4440-8b21-3082ed2e3a13)
![image](https://github.com/user-attachments/assets/1a173446-a0fd-49e5-a394-c6b5d9eb62e6)

## Business Insights

**Content Preferences:**
Insight: International movies and TV shows have gained significant popularity. The US still dominates production, but content from India, the UK, and other countries are gaining traction.
Inference: This suggests diversifying content to cater to global audiences.

**Global Expansion:**
Insight: Specific countries exhibit strong preferences for certain genres. For example, Indian audiences may favor Bollywood content, while South Korean dramas have a dedicated following globally.
Inference: Tailoring content recommendations and marketing efforts based on regional preferences could improve engagement.

**Trends Over Time:**
Insight: The number of releases on Netflix has been increasing over the years, with a significant rise in recent years.
Inference: Staying ahead of content trends and adapting to evolving audience preferences is crucial for success.

**Optimal Release Timing:**
Insight: Movies and TV shows released on specific days of the week and months tend to perform better.
Inference: Strategizing release dates to align with these patterns could maximize viewership.

**Key Talent:**
Insight: Certain actors, directors, and genres consistently attract large audiences.
Inference: Collaborating with popular talent and focusing on high-performing genres can improve content success.

**Content Duration:**
Insight: Movies generally have shorter durations compared to TV shows, which are often multi-seasonal.
Inference: Consider diversifying content offerings with both short-form (movies) and long-form (TV series) content to cater to different viewer preferences.

## Recommendations

- Diversify Content: Invest in producing more international content, especially from regions with growing Netflix adoption.
- Localize Content: Consider language dubbing and subtitling to reach wider audiences.
- Personalize Recommendations: Enhance recommendation algorithms to account for user location and viewing history.
- Optimize Release Schedule: Plan content releases strategically to align with peak viewing times and audience preferences.
- Collaborate with Top Talent: Partner with popular actors, directors, and production houses to attract viewers.
- Promote Diverse Genres: Ensure a balanced mix of genres to cater to different preferences.
- Leverage Data Insights: Continuously monitor data to identify emerging trends and adapt content strategy accordingly.
- Focus on Original Content: Invest in producing high-quality original movies and TV shows to differentiate Netflix from competitors.
- Expand Marketing Efforts: Promote content through targeted advertising and social media campaigns to reach specific audiences.
- Gather User Feedback: Actively solicit feedback from users to understand their preferences and improve content offerings.


















