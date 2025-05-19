## **Case Background:** 
Netflix is known for its work in data science, AI, and ML, particularly for building strong recommendation models and algorithms that understand customer behavior and patterns. Suppose you are working in a data-driven job role, and you have a dataset of more than 9,000 movies. You need to solve the following questions to help the company make informed business decisions accordingly.

## **Exploratory Data Analysis using the guiding questions:** 
1. What is the most frequent genre of movies released on Netflix?
2. Which has highest votes in vote avg column?
3. What movie got the highest popularity? what's its genre?
4. What movie got the lowest popularity? what's its genre?
5. Which year has the most filmed movies?

## **Conclusion**
1. The most frequent genre of movies is Drama with a frequency of 3715/25552 which is roughly 14.5% of all genres
2. The popular vote category has 25.5% votes with 6250 rows. Drama is the most popular genre in the popular category
3. Spiderman-no way home has the highest popularity rate with the genres action, science fiction and adventure
4. The United States vs. Billie Holiday, and Threads were least popular, with the genres Music, Drama, History, War, Science Fiction
5. The year 2021 has the most filmed movies

## **Learnings:** 
1. Always save the dataset into a different dataframe to be able to compare dataframes
2. Make your own conclusions, check and recheck assumptions

## Analysis
    

### Step: Import relavant Libraries


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
```

### Step: Import data file


```python
df = pd.read_csv('mymoviedb.csv', lineterminator = '\n')

#lineterminator = '\n': Each row of your CSV ends with \n, so you're telling pandas: “Hey, each line ends with a newline — split the rows accordingly.”
```


```python
df.head() #Scan the first 5 rows of the dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Release_Date</th>
      <th>Title</th>
      <th>Overview</th>
      <th>Popularity</th>
      <th>Vote_Count</th>
      <th>Vote_Average</th>
      <th>Original_Language</th>
      <th>Genre</th>
      <th>Poster_Url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-12-15</td>
      <td>Spider-Man: No Way Home</td>
      <td>Peter Parker is unmasked and no longer able to...</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>en</td>
      <td>Action, Adventure, Science Fiction</td>
      <td>https://image.tmdb.org/t/p/original/1g0dhYtq4i...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-03-01</td>
      <td>The Batman</td>
      <td>In his second year of fighting crime, Batman u...</td>
      <td>3827.658</td>
      <td>1151</td>
      <td>8.1</td>
      <td>en</td>
      <td>Crime, Mystery, Thriller</td>
      <td>https://image.tmdb.org/t/p/original/74xTEgt7R3...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-02-25</td>
      <td>No Exit</td>
      <td>Stranded at a rest stop in the mountains durin...</td>
      <td>2618.087</td>
      <td>122</td>
      <td>6.3</td>
      <td>en</td>
      <td>Thriller</td>
      <td>https://image.tmdb.org/t/p/original/vDHsLnOWKl...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-11-24</td>
      <td>Encanto</td>
      <td>The tale of an extraordinary family, the Madri...</td>
      <td>2402.201</td>
      <td>5076</td>
      <td>7.7</td>
      <td>en</td>
      <td>Animation, Comedy, Family, Fantasy</td>
      <td>https://image.tmdb.org/t/p/original/4j0PNHkMr5...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-12-22</td>
      <td>The King's Man</td>
      <td>As a collection of history's worst tyrants and...</td>
      <td>1895.511</td>
      <td>1793</td>
      <td>7.0</td>
      <td>en</td>
      <td>Action, Adventure, Thriller, War</td>
      <td>https://image.tmdb.org/t/p/original/aq4Pwv5Xeu...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info() #Check Data Types of columns in the dataset
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9827 entries, 0 to 9826
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Release_Date       9827 non-null   object 
     1   Title              9827 non-null   object 
     2   Overview           9827 non-null   object 
     3   Popularity         9827 non-null   float64
     4   Vote_Count         9827 non-null   int64  
     5   Vote_Average       9827 non-null   float64
     6   Original_Language  9827 non-null   object 
     7   Genre              9827 non-null   object 
     8   Poster_Url         9827 non-null   object 
    dtypes: float64(2), int64(1), object(6)
    memory usage: 691.1+ KB
    


```python
df['Genre'].head() #Check the 1st 5 rows of genre column for irregulatiries
```




    0    Action, Adventure, Science Fiction
    1              Crime, Mystery, Thriller
    2                              Thriller
    3    Animation, Comedy, Family, Fantasy
    4      Action, Adventure, Thriller, War
    Name: Genre, dtype: object




```python
len(df) #Checking number of rows in the dataset
```




    9827




```python
df.duplicated().sum() #Checking if the dataset has any duplicates

# df.duplicated(): This checks every row in your DataFrame df and returns a boolean Series 
# — True if a row is a duplicate of a previous row, False otherwise.

# .sum(): Tells you how many rows are duplicated (Sum of true)
```




    0




```python
df.describe() #For Statistical summary of numeric columns
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Popularity</th>
      <th>Vote_Count</th>
      <th>Vote_Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9827.000000</td>
      <td>9827.000000</td>
      <td>9827.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.326088</td>
      <td>1392.805536</td>
      <td>6.439534</td>
    </tr>
    <tr>
      <th>std</th>
      <td>108.873998</td>
      <td>2611.206907</td>
      <td>1.129759</td>
    </tr>
    <tr>
      <th>min</th>
      <td>13.354000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>16.128500</td>
      <td>146.000000</td>
      <td>5.900000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21.199000</td>
      <td>444.000000</td>
      <td>6.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>35.191500</td>
      <td>1376.000000</td>
      <td>7.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5083.954000</td>
      <td>31077.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Moving Summary

**Observations:**
1. The datafrmae has 9827 rows and 9 columns
2. The dataset seems tidy with no NaNs or duplicates

**Modifications:**
1. Release_Date needs to be casted into data time to extract only the year (which is relavant to the analysis)
2. The columns Overview, Original_Language and Poster_Url are not relavant to the analysis
3. Vote_Average can be categorised for better analysis
4. Genre column has comma seperated values with a white space after each comma
5. Genre column should also be categorised for better analysis
   

### Step: Changing Data type of `Release_Date` column


```python
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
print(df['Release_Date'].dtype)

#Meaning of each line:
#df['Release_Date']
#→ You're selecting the column called Release_Date.

#pd.to_datetime(...)
#→ This function tries to convert strings (like '2023-01-01' or '01/01/2023') into actual datetime64 format 
# — basically, dates that pandas can understand and work with (for sorting, extracting years, etc.).

#errors='coerce'
#→ If a value cannot be converted to a proper date (like "unknown" or "abcd"), it will not crash.
#Instead, it will turn that value into NaT, which means Not a Time (similar to NaN but for datetime).
```

    datetime64[ns]
    


```python
df['Release_Date'] = df['Release_Date'].dt.year
df['Release_Date'].dtype

#df['Release_Date']: selects the Release_Date column from the DataFrame.

#.dt.year: extracts only the year from the datetime values.

#df['Release_Date'] = ...: overwrites the column with just the years.

#df['Release_Date'].dtype: checks the data type of the updated column (will now show int64).
```




    dtype('int32')




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9827 entries, 0 to 9826
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Release_Date       9827 non-null   int32  
     1   Title              9827 non-null   object 
     2   Overview           9827 non-null   object 
     3   Popularity         9827 non-null   float64
     4   Vote_Count         9827 non-null   int64  
     5   Vote_Average       9827 non-null   float64
     6   Original_Language  9827 non-null   object 
     7   Genre              9827 non-null   object 
     8   Poster_Url         9827 non-null   object 
    dtypes: float64(2), int32(1), int64(1), object(5)
    memory usage: 652.7+ KB
    


```python
df['Release_Date']
```




    0       2021
    1       2022
    2       2022
    3       2021
    4       2021
            ... 
    9822    1973
    9823    2020
    9824    2016
    9825    2021
    9826    1984
    Name: Release_Date, Length: 9827, dtype: int32



### Step: Removing the `Overview`, `Original_Language` and `Poster_Url` columns 


```python
df.columns
```




    Index(['Release_Date', 'Title', 'Overview', 'Popularity', 'Vote_Count',
           'Vote_Average', 'Original_Language', 'Genre', 'Poster_Url'],
          dtype='object')




```python
cols = ['Overview','Original_Language','Poster_Url']
```


```python
df.drop(cols, axis = 1, inplace = True)

#df.drop(...): drops rows or columns from df.

#cols: a list of column names to remove (e.g., ['Overview', 'Poster_Url']).

#axis=1: tells pandas to drop columns (axis 0 would be rows).

#inplace=True: applies the change directly to df without needing to assign it back. 

#Without inplace=True: you're making a copy and replacing the old one.
#With inplace=True: you're modifying the original directly.
#It’s like editing a document:
#Without inplace=True: you make edits in a new file, then save over the original.
#With inplace=True: you just edit the original file directly.
```


```python
df.columns
```




    Index(['Release_Date', 'Title', 'Popularity', 'Vote_Count', 'Vote_Average',
           'Genre'],
          dtype='object')



### Step: Categorising the `Vote_Average` column

The `Vote_Average` will be divided into 4 categories: `popular`, `average`, `below_average` and `not popular`


```python
def categorize(df, col, labels, new_col_name):
    edges = [
        df[col].describe()['min'],
        df[col].describe()['25%'],
        df[col].describe()['50%'],
        df[col].describe()['75%'],
        df[col].describe()['max']
    ]
    df[new_col_name] = pd.cut(df[col], bins=edges, labels=labels, duplicates='drop')
    return df

#This is defining a function called categorize. It takes:

#df: your DataFrame,

#col: the name of the column you want to categorize (as a string),

#labels: a list of names to give to each category (like ['low', 'medium', 'high']).

#new_col_name: to insert results into a new column instead of replacing an existing column.
```


```python
labels = ['Not Popular', 'Below Average', 'Average', 'Popular']
categorize(df, 'Vote_Average', labels, 'Vote_Average_Category')
df['Vote_Average_Category'].unique()

#Here the labels are being assigned to the bins created in the previous line of code
#Calling the categorise function defined earlier with the feilds required 
#Returning values of the column that are unique
```




    ['Popular', 'Below Average', 'Average', 'Not Popular', NaN]
    Categories (4, object): ['Not Popular' < 'Below Average' < 'Average' < 'Popular']




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Release_Date</th>
      <th>Title</th>
      <th>Popularity</th>
      <th>Vote_Count</th>
      <th>Vote_Average</th>
      <th>Genre</th>
      <th>Vote_Average_Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>Spider-Man: No Way Home</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>Action, Adventure, Science Fiction</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022</td>
      <td>The Batman</td>
      <td>3827.658</td>
      <td>1151</td>
      <td>8.1</td>
      <td>Crime, Mystery, Thriller</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>No Exit</td>
      <td>2618.087</td>
      <td>122</td>
      <td>6.3</td>
      <td>Thriller</td>
      <td>Below Average</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021</td>
      <td>Encanto</td>
      <td>2402.201</td>
      <td>5076</td>
      <td>7.7</td>
      <td>Animation, Comedy, Family, Fantasy</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>The King's Man</td>
      <td>1895.511</td>
      <td>1793</td>
      <td>7.0</td>
      <td>Action, Adventure, Thriller, War</td>
      <td>Average</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Vote_Average_Category'].value_counts()
```




    Vote_Average_Category
    Not Popular      2467
    Popular          2450
    Average          2412
    Below Average    2398
    Name: count, dtype: int64




```python
df.dropna(inplace= True)
df.isna().sum()
```




    Release_Date             0
    Title                    0
    Popularity               0
    Vote_Count               0
    Vote_Average             0
    Genre                    0
    Vote_Average_Category    0
    dtype: int64



### Step: Split the `Genre` field so that each movie-genre pair appears in a separate row, with only one genre per line.


```python
df['Genre'] = df['Genre'].str.split(', ')

df = df.explode('Genre').reset_index(drop = True)

df.head()

#This splits the Genre column ("Action, Adventure" or "Comedy, Drama") into a list of genres.
#For example:'Action, Adventure' → ['Action', 'Adventure']
#This makes each cell in the Genre column a list instead of a string.

# explode('Genre'): This takes each list in the Genre column and creates a new row for each item in the list,
#keeping the rest of the row data the same.

#reset_index(drop=True): After exploding, the row indices may look jumbled (e.g., 0, 0, 1, 2, 2, 2...).
#This resets the index to a clean sequence (0, 1, 2, 3...) and drops the old index.
#So if a movie had 3 genres, it now appears as 3 separate rows, each with one genre. 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Release_Date</th>
      <th>Title</th>
      <th>Popularity</th>
      <th>Vote_Count</th>
      <th>Vote_Average</th>
      <th>Genre</th>
      <th>Vote_Average_Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>Spider-Man: No Way Home</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>Action</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021</td>
      <td>Spider-Man: No Way Home</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>Adventure</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021</td>
      <td>Spider-Man: No Way Home</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>Science Fiction</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022</td>
      <td>The Batman</td>
      <td>3827.658</td>
      <td>1151</td>
      <td>8.1</td>
      <td>Crime</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022</td>
      <td>The Batman</td>
      <td>3827.658</td>
      <td>1151</td>
      <td>8.1</td>
      <td>Mystery</td>
      <td>Popular</td>
    </tr>
  </tbody>
</table>
</div>



### Step: Create categories out of the `Genre` field


```python
df['Genre'] = df['Genre'].astype('category')
df['Genre'].dtype
```




    CategoricalDtype(categories=['Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                      'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
                      'TV Movie', 'Thriller', 'War', 'Western'],
    , ordered=False, categories_dtype=object)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 25552 entries, 0 to 25551
    Data columns (total 7 columns):
     #   Column                 Non-Null Count  Dtype   
    ---  ------                 --------------  -----   
     0   Release_Date           25552 non-null  int32   
     1   Title                  25552 non-null  object  
     2   Popularity             25552 non-null  float64 
     3   Vote_Count             25552 non-null  int64   
     4   Vote_Average           25552 non-null  float64 
     5   Genre                  25552 non-null  category
     6   Vote_Average_Category  25552 non-null  category
    dtypes: category(2), float64(2), int32(1), int64(1), object(1)
    memory usage: 949.2+ KB
    


```python
df.nunique() 

#This function tells you how many unique values there are in each column of your DataFrame.
```




    Release_Date              100
    Title                    9415
    Popularity               8088
    Vote_Count               3265
    Vote_Average               73
    Genre                      19
    Vote_Average_Category       4
    dtype: int64



### Step: Create Visualisation


```python
sns.set_style('white') 

#sns → This refers to Seaborn,a Python visualization library built on top of matplotlib, used for making beautiful and informative charts easily.

#set_style() → This function sets the overall theme or style for your Seaborn plots. 
#It changes things like background color, grid lines, tick marks, etc.

#'whitegrid' → This is one of the available style options. It means:
#White background

#Light grey grid lines on the plot (good for numerical data)

#Other Options: 
#sns.set_style('darkgrid')    # Dark background with grid
#sns.set_style('white')       # White background, no grid
#sns.set_style('dark')        # Dark background, no grid
#sns.set_style('ticks')       # White background with ticks on axes
```

### Question 1: What is the most frequent genre of movies released on Netflix?


```python
df['Genre'].describe()
```




    count     25552
    unique       19
    top       Drama
    freq       3715
    Name: Genre, dtype: object




```python
sns.catplot(y='Genre', data = df, kind = 'count',
            order = df['Genre'].value_counts().index)
plt.title('Genre column distribution')
plt.show
df['Genre'].value_counts()            


#sns.catplot(...)
#→ This is a categorical plot function from Seaborn. 
#It’s used to visualize relationships between a categorical variable and some other variable (or count of occurrences).

#x='Genre' → Tells Seaborn to put the 'Genre' column on the x-axis. This means you're showing data for each unique genre.

#data=df → You're plotting from the DataFrame named df.

#kind='count' → This makes a count plot, meaning it counts how many times each genre appears and shows bars accordingly.
#(It’s the same as sns.countplot() under the hood.)

#order=df['Genre'].value_counts().index → This sorts the bars by frequency (highest to lowest).
```




    Genre
    Drama              3715
    Comedy             3006
    Action             2652
    Thriller           2473
    Adventure          1829
    Romance            1461
    Horror             1457
    Animation          1426
    Family             1405
    Fantasy            1295
    Science Fiction    1255
    Crime              1235
    Mystery             765
    History             426
    War                 307
    Music               291
    TV Movie            214
    Documentary         203
    Western             137
    Name: count, dtype: int64




    
![png](output_42_1.png)
    



```python
(df['Genre'].value_counts(normalize=True) * 100).round(2)
```




    Genre
    Drama              14.54
    Comedy             11.76
    Action             10.38
    Thriller            9.68
    Adventure           7.16
    Romance             5.72
    Horror              5.70
    Animation           5.58
    Family              5.50
    Fantasy             5.07
    Science Fiction     4.91
    Crime               4.83
    Mystery             2.99
    History             1.67
    War                 1.20
    Music               1.14
    TV Movie            0.84
    Documentary         0.79
    Western             0.54
    Name: proportion, dtype: float64



### Question 2: Which has highest votes in vote avg column?


```python
df['Vote_Average_Category'].value_counts()
(df['Vote_Average_Category'].value_counts(normalize=True)*100).round(2)
```




    Vote_Average_Category
    Average          25.88
    Popular          25.52
    Below Average    24.84
    Not Popular      23.76
    Name: proportion, dtype: float64




```python
sns.catplot(y='Vote_Average_Category', data = df, kind = 'count',
            order = df['Vote_Average_Category'].value_counts().index)
plt.title('Votes Distribution Chart')
plt.show()
```


    
![png](output_46_0.png)
    



```python
popular = df[df['Vote_Average_Category'] == 'Popular'] 
sns.catplot(y='Genre', data = popular, kind = 'count',
            order = df['Genre'].value_counts().index)
plt.title('Popular Category Split')
plt.show()

print(pd.DataFrame({
    'Count': df[df['Vote_Average_Category'] == 'Popular']['Genre'].value_counts(),
    'Percentage': (df[df['Vote_Average_Category'] == 'Popular']['Genre'].value_counts(normalize=True) * 100).round(2)
}))
```


    
![png](output_47_0.png)
    


                     Count  Percentage
    Genre                             
    Drama             1308       20.06
    Comedy             602        9.23
    Animation          544        8.34
    Action             527        8.08
    Adventure          449        6.89
    Romance            414        6.35
    Thriller           394        6.04
    Fantasy            379        5.81
    Family             356        5.46
    Crime              327        5.02
    Science Fiction    265        4.06
    Mystery            197        3.02
    History            173        2.65
    War                131        2.01
    Horror             129        1.98
    Music              118        1.81
    Documentary        105        1.61
    TV Movie            54        0.83
    Western             48        0.74
    

### Question 3: What movie got the highest popularity? what's its genre?


```python
df[df['Popularity']==df['Popularity'].max()]

# df['Popularity'].max() --> Finds the maximum value inside Popularity

# df['Popularity']==df['Popularity'].max() --> Checks each row if the popularity = maximum 

#df[df['Popularity']==df['Popularity'].max()] --> Filters those rows from the dataframe where popularity is maximum
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Release_Date</th>
      <th>Title</th>
      <th>Popularity</th>
      <th>Vote_Count</th>
      <th>Vote_Average</th>
      <th>Genre</th>
      <th>Vote_Average_Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>Spider-Man: No Way Home</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>Action</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021</td>
      <td>Spider-Man: No Way Home</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>Adventure</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021</td>
      <td>Spider-Man: No Way Home</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>Science Fiction</td>
      <td>Popular</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Or: 
df.sort_values(by='Popularity', ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Release_Date</th>
      <th>Title</th>
      <th>Popularity</th>
      <th>Vote_Count</th>
      <th>Vote_Average</th>
      <th>Genre</th>
      <th>Vote_Average_Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>Spider-Man: No Way Home</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>Action</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021</td>
      <td>Spider-Man: No Way Home</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>Science Fiction</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021</td>
      <td>Spider-Man: No Way Home</td>
      <td>5083.954</td>
      <td>8940</td>
      <td>8.3</td>
      <td>Adventure</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022</td>
      <td>The Batman</td>
      <td>3827.658</td>
      <td>1151</td>
      <td>8.1</td>
      <td>Crime</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022</td>
      <td>The Batman</td>
      <td>3827.658</td>
      <td>1151</td>
      <td>8.1</td>
      <td>Thriller</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022</td>
      <td>The Batman</td>
      <td>3827.658</td>
      <td>1151</td>
      <td>8.1</td>
      <td>Mystery</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2022</td>
      <td>No Exit</td>
      <td>2618.087</td>
      <td>122</td>
      <td>6.3</td>
      <td>Thriller</td>
      <td>Below Average</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2021</td>
      <td>Encanto</td>
      <td>2402.201</td>
      <td>5076</td>
      <td>7.7</td>
      <td>Family</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2021</td>
      <td>Encanto</td>
      <td>2402.201</td>
      <td>5076</td>
      <td>7.7</td>
      <td>Fantasy</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021</td>
      <td>Encanto</td>
      <td>2402.201</td>
      <td>5076</td>
      <td>7.7</td>
      <td>Animation</td>
      <td>Popular</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sorted = df.sort_values(by='Popularity', ascending=False)
sns.barplot(y='Title', x='Popularity', data=df_sorted.head(20))
plt.show()

#df.sort_values(by='Popularity', ascending=False): This sorts the DataFrame df by the 'Popularity' column in descending order (because ascending=False)
#This means that the rows with the highest popularity values will appear at the top of the DataFrame.

#sns.barplot(): This function from the seaborn library is used to create a bar plot.

#y='Title': The Title column of the DataFrame will be used for the labels on the y-axis.

#x='Popularity': The Popularity column will be used for the values on the x-axis.

#data=df_sorted.head(20): This specifies that you want to plot the first 20 rows of the sorted DataFrame df_sorted. 
#The head(20) function selects the top 20 rows, so you will plot the 20 entries with the highest popularity values.

```


    
![png](output_51_0.png)
    


### Question 4: What movie got the lowest popularity? what's its genre?


```python
df[df['Popularity']==df['Popularity'].min()]

# df['Popularity'].min() --> Finds the minimum value inside Popularity

# df['Popularity']==df['Popularity'].min() --> Checks each row if the popularity = minimum 

#df[df['Popularity']==df['Popularity'].min()] --> Filters those rows from the dataframe where popularity is minimum
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Release_Date</th>
      <th>Title</th>
      <th>Popularity</th>
      <th>Vote_Count</th>
      <th>Vote_Average</th>
      <th>Genre</th>
      <th>Vote_Average_Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25546</th>
      <td>2021</td>
      <td>The United States vs. Billie Holiday</td>
      <td>13.354</td>
      <td>152</td>
      <td>6.7</td>
      <td>Music</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>25547</th>
      <td>2021</td>
      <td>The United States vs. Billie Holiday</td>
      <td>13.354</td>
      <td>152</td>
      <td>6.7</td>
      <td>Drama</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>25548</th>
      <td>2021</td>
      <td>The United States vs. Billie Holiday</td>
      <td>13.354</td>
      <td>152</td>
      <td>6.7</td>
      <td>History</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>25549</th>
      <td>1984</td>
      <td>Threads</td>
      <td>13.354</td>
      <td>186</td>
      <td>7.8</td>
      <td>War</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>25550</th>
      <td>1984</td>
      <td>Threads</td>
      <td>13.354</td>
      <td>186</td>
      <td>7.8</td>
      <td>Drama</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>25551</th>
      <td>1984</td>
      <td>Threads</td>
      <td>13.354</td>
      <td>186</td>
      <td>7.8</td>
      <td>Science Fiction</td>
      <td>Popular</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Or:
df.sort_values(by='Popularity', ascending=True).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Release_Date</th>
      <th>Title</th>
      <th>Popularity</th>
      <th>Vote_Count</th>
      <th>Vote_Average</th>
      <th>Genre</th>
      <th>Vote_Average_Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25551</th>
      <td>1984</td>
      <td>Threads</td>
      <td>13.354</td>
      <td>186</td>
      <td>7.8</td>
      <td>Science Fiction</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>25546</th>
      <td>2021</td>
      <td>The United States vs. Billie Holiday</td>
      <td>13.354</td>
      <td>152</td>
      <td>6.7</td>
      <td>Music</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>25547</th>
      <td>2021</td>
      <td>The United States vs. Billie Holiday</td>
      <td>13.354</td>
      <td>152</td>
      <td>6.7</td>
      <td>Drama</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>25550</th>
      <td>1984</td>
      <td>Threads</td>
      <td>13.354</td>
      <td>186</td>
      <td>7.8</td>
      <td>Drama</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>25549</th>
      <td>1984</td>
      <td>Threads</td>
      <td>13.354</td>
      <td>186</td>
      <td>7.8</td>
      <td>War</td>
      <td>Popular</td>
    </tr>
    <tr>
      <th>25548</th>
      <td>2021</td>
      <td>The United States vs. Billie Holiday</td>
      <td>13.354</td>
      <td>152</td>
      <td>6.7</td>
      <td>History</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>25544</th>
      <td>2016</td>
      <td>The Offering</td>
      <td>13.355</td>
      <td>94</td>
      <td>5.0</td>
      <td>Thriller</td>
      <td>Not Popular</td>
    </tr>
    <tr>
      <th>25545</th>
      <td>2016</td>
      <td>The Offering</td>
      <td>13.355</td>
      <td>94</td>
      <td>5.0</td>
      <td>Horror</td>
      <td>Not Popular</td>
    </tr>
    <tr>
      <th>25543</th>
      <td>2016</td>
      <td>The Offering</td>
      <td>13.355</td>
      <td>94</td>
      <td>5.0</td>
      <td>Mystery</td>
      <td>Not Popular</td>
    </tr>
    <tr>
      <th>25542</th>
      <td>2020</td>
      <td>Violent Delights</td>
      <td>13.356</td>
      <td>8</td>
      <td>3.5</td>
      <td>Horror</td>
      <td>Not Popular</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sorted = df.sort_values(by='Popularity', ascending=True)
sns.barplot(y='Title', x='Popularity', data=df_sorted.head(20))
plt.show()

#df.sort_values(by='Popularity', ascending=True): This sorts the DataFrame df by the 'Popularity' column in asecnding order (because ascending=True)
#This means that the rows with the lowest popularity values will appear at the top of the DataFrame.

#sns.barplot(): This function from the seaborn library is used to create a bar plot.

#y='Title': The Title column of the DataFrame will be used for the labels on the y-axis.

#x='Popularity': The Popularity column will be used for the values on the x-axis.

#data=df_sorted.head(20): This specifies that you want to plot the first 20 rows of the sorted DataFrame df_sorted. 
#The head(20) function selects the top 20 rows, so you will plot the 20 entries with the lowest popularity values.

```


    
![png](output_55_0.png)
    


### Question 5: Which year has the most filmed movies?


```python
df['Release_Date'].hist(bins=range(df['Release_Date'].min(), df['Release_Date'].max() + 1))
plt.title("Movies by Year of Release")
plt.show

df['Release_Date'].value_counts().head(10).sort_values(ascending = False)

```




    Release_Date
    2021    1636
    2018    1384
    2017    1365
    2019    1271
    2016    1209
    2020    1121
    2015    1015
    2014     922
    2013     877
    2011     855
    Name: count, dtype: int64




    
![png](output_57_1.png)
    



```python

```
