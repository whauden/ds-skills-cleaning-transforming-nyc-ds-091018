
# Import Pandas


```python
import pandas as pd
```

# Load and Preview the Dataset


```python
df = pd.read_csv('lego_sets.csv')
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
      <th>ages</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>piece_count</th>
      <th>play_star_rating</th>
      <th>prod_desc</th>
      <th>prod_id</th>
      <th>prod_long_desc</th>
      <th>review_difficulty</th>
      <th>set_name</th>
      <th>star_rating</th>
      <th>theme_name</th>
      <th>val_star_rating</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6-12</td>
      <td>29.99</td>
      <td>2.0</td>
      <td>277.0</td>
      <td>4.0</td>
      <td>Catapult into action and take back the eggs fr...</td>
      <td>75823.0</td>
      <td>Use the staircase catapult to launch Red into ...</td>
      <td>Average</td>
      <td>Bird Island Egg Heist</td>
      <td>4.5</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>US</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6-12</td>
      <td>19.99</td>
      <td>2.0</td>
      <td>168.0</td>
      <td>4.0</td>
      <td>Launch a flying attack and rescue the eggs fro...</td>
      <td>75822.0</td>
      <td>Pilot Pig has taken off from Bird Island with ...</td>
      <td>Easy</td>
      <td>Piggy Plane Attack</td>
      <td>5.0</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>US</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6-12</td>
      <td>12.99</td>
      <td>11.0</td>
      <td>74.0</td>
      <td>4.3</td>
      <td>Chase the piggy with lightning-fast Chuck and ...</td>
      <td>75821.0</td>
      <td>Pitch speedy bird Chuck against the Piggy Car....</td>
      <td>Easy</td>
      <td>Piggy Car Escape</td>
      <td>4.3</td>
      <td>Angry Birds™</td>
      <td>4.1</td>
      <td>US</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12+</td>
      <td>99.99</td>
      <td>23.0</td>
      <td>1032.0</td>
      <td>3.6</td>
      <td>Explore the architecture of the United States ...</td>
      <td>21030.0</td>
      <td>Discover the architectural secrets of the icon...</td>
      <td>Average</td>
      <td>United States Capitol Building</td>
      <td>4.6</td>
      <td>Architecture</td>
      <td>4.3</td>
      <td>US</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12+</td>
      <td>79.99</td>
      <td>14.0</td>
      <td>744.0</td>
      <td>3.2</td>
      <td>Recreate the Solomon R. Guggenheim Museum® wit...</td>
      <td>21035.0</td>
      <td>Discover the architectural secrets of Frank Ll...</td>
      <td>Challenging</td>
      <td>Solomon R. Guggenheim Museum®</td>
      <td>4.6</td>
      <td>Architecture</td>
      <td>4.1</td>
      <td>US</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12261 entries, 0 to 12260
    Data columns (total 14 columns):
    ages                 12261 non-null object
    list_price           12261 non-null float64
    num_reviews          10641 non-null float64
    piece_count          12261 non-null float64
    play_star_rating     10486 non-null float64
    prod_desc            11884 non-null object
    prod_id              12261 non-null float64
    prod_long_desc       12261 non-null object
    review_difficulty    10206 non-null object
    set_name             12261 non-null object
    star_rating          10641 non-null float64
    theme_name           12258 non-null object
    val_star_rating      10466 non-null float64
    country              12261 non-null object
    dtypes: float64(7), object(7)
    memory usage: 1.3+ MB


# Feature Engingeering
As we'll see later, we'll often want to create new features for our data sets in order to improve the performance of various machine learning algorithms. Let's practice this with a few examples.

# Mean Price by Theme
Let's create a new column that lists the mean price for the theme that that particular lego set is from. This could prove useful for a regression algorithm that we'll be building later!

Here's a general outline:
    * Calculate average price per theme; use the groupby method, subset to price and calculate the mean
    * Create a dictionary of `{theme : avg_price}`
    * Make the new column; map the dictionary to the original theme_column and save the results to a new column


```python
#Your code here

#Groupby theme_name and calculate average price
grouped = df.groupby('theme_name')['list_price'].mean()#Your code here

#Can be helpful to preview your intermediate transformations
grouped.head()
```




    theme_name
    Angry Birds™                  21.021100
    Architecture                  65.082371
    BOOST                        196.572316
    Blue's Helicopter Pursuit     61.934648
    BrickHeadz                    14.868018
    Name: list_price, dtype: float64




```python
#Your code here

#Create Dictionary
theme_price_dict = dict(grouped)
#Create new column with dictionary
df['Theme_Avg_Price'] = df.theme_name.map(theme_price_dict)
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
      <th>ages</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>piece_count</th>
      <th>play_star_rating</th>
      <th>prod_desc</th>
      <th>prod_id</th>
      <th>prod_long_desc</th>
      <th>review_difficulty</th>
      <th>set_name</th>
      <th>star_rating</th>
      <th>theme_name</th>
      <th>val_star_rating</th>
      <th>country</th>
      <th>Theme_Avg_Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6-12</td>
      <td>29.99</td>
      <td>2.0</td>
      <td>277.0</td>
      <td>4.0</td>
      <td>Catapult into action and take back the eggs fr...</td>
      <td>75823.0</td>
      <td>Use the staircase catapult to launch Red into ...</td>
      <td>Average</td>
      <td>Bird Island Egg Heist</td>
      <td>4.5</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>US</td>
      <td>21.021100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6-12</td>
      <td>19.99</td>
      <td>2.0</td>
      <td>168.0</td>
      <td>4.0</td>
      <td>Launch a flying attack and rescue the eggs fro...</td>
      <td>75822.0</td>
      <td>Pilot Pig has taken off from Bird Island with ...</td>
      <td>Easy</td>
      <td>Piggy Plane Attack</td>
      <td>5.0</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>US</td>
      <td>21.021100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6-12</td>
      <td>12.99</td>
      <td>11.0</td>
      <td>74.0</td>
      <td>4.3</td>
      <td>Chase the piggy with lightning-fast Chuck and ...</td>
      <td>75821.0</td>
      <td>Pitch speedy bird Chuck against the Piggy Car....</td>
      <td>Easy</td>
      <td>Piggy Car Escape</td>
      <td>4.3</td>
      <td>Angry Birds™</td>
      <td>4.1</td>
      <td>US</td>
      <td>21.021100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12+</td>
      <td>99.99</td>
      <td>23.0</td>
      <td>1032.0</td>
      <td>3.6</td>
      <td>Explore the architecture of the United States ...</td>
      <td>21030.0</td>
      <td>Discover the architectural secrets of the icon...</td>
      <td>Average</td>
      <td>United States Capitol Building</td>
      <td>4.6</td>
      <td>Architecture</td>
      <td>4.3</td>
      <td>US</td>
      <td>65.082371</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12+</td>
      <td>79.99</td>
      <td>14.0</td>
      <td>744.0</td>
      <td>3.2</td>
      <td>Recreate the Solomon R. Guggenheim Museum® wit...</td>
      <td>21035.0</td>
      <td>Discover the architectural secrets of Frank Ll...</td>
      <td>Challenging</td>
      <td>Solomon R. Guggenheim Museum®</td>
      <td>4.6</td>
      <td>Architecture</td>
      <td>4.1</td>
      <td>US</td>
      <td>65.082371</td>
    </tr>
  </tbody>
</table>
</div>



# Extending Code with for Loops
Expand upon our previous example by writing a function that takes in a column to group by and a column to take the average of (in our previous example, theme_name and list_price) and creates a new column to our dataframe corresponding to the average value for the category to which that feature corresponds. 


```python
def avg_feat(cfeat, nfeat):
    new_col = '{}_Avg_{}'.format(cfeat, nfeat)
    grouped = df.groupby(cfeat)[nfeat].mean()
    df[new_col] = df[cfeat].map(dict(grouped))
```

# Applying your function
Now write a for loop that iterates over several category columns and several numerical columns, and apply your above function to create a new column of the average values for the categorical feature.


```python
df.columns
```




    Index(['ages', 'list_price', 'num_reviews', 'piece_count', 'play_star_rating',
           'prod_desc', 'prod_id', 'prod_long_desc', 'review_difficulty',
           'set_name', 'star_rating', 'theme_name', 'val_star_rating', 'country',
           'Theme_Avg_Price'],
          dtype='object')




```python
cat_feats = ['ages', 'review_difficulty', 'country']
num_feats = ['list_price', 'num_reviews', 'piece_count', 'play_star_rating', 'star_rating', 'val_star_rating']
for cfeat in cat_feats:
    for nfeat in num_feats:
        avg_feat(cfeat, nfeat)
print(df.columns)
df.head(2)
```

    Index(['ages', 'list_price', 'num_reviews', 'piece_count', 'play_star_rating',
           'prod_desc', 'prod_id', 'prod_long_desc', 'review_difficulty',
           'set_name', 'star_rating', 'theme_name', 'val_star_rating', 'country',
           'Theme_Avg_Price', 'ages_Avg_list_price', 'ages_Avg_num_reviews',
           'ages_Avg_piece_count', 'ages_Avg_play_star_rating',
           'ages_Avg_star_rating', 'ages_Avg_val_star_rating',
           'review_difficulty_Avg_list_price', 'review_difficulty_Avg_num_reviews',
           'review_difficulty_Avg_piece_count',
           'review_difficulty_Avg_play_star_rating',
           'review_difficulty_Avg_star_rating',
           'review_difficulty_Avg_val_star_rating', 'country_Avg_list_price',
           'country_Avg_num_reviews', 'country_Avg_piece_count',
           'country_Avg_play_star_rating', 'country_Avg_star_rating',
           'country_Avg_val_star_rating', 'Has_Trademark', 'Has_Registered'],
          dtype='object')





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
      <th>ages</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>piece_count</th>
      <th>play_star_rating</th>
      <th>prod_desc</th>
      <th>prod_id</th>
      <th>prod_long_desc</th>
      <th>review_difficulty</th>
      <th>set_name</th>
      <th>...</th>
      <th>review_difficulty_Avg_star_rating</th>
      <th>review_difficulty_Avg_val_star_rating</th>
      <th>country_Avg_list_price</th>
      <th>country_Avg_num_reviews</th>
      <th>country_Avg_piece_count</th>
      <th>country_Avg_play_star_rating</th>
      <th>country_Avg_star_rating</th>
      <th>country_Avg_val_star_rating</th>
      <th>Has_Trademark</th>
      <th>Has_Registered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6-12</td>
      <td>29.99</td>
      <td>2.0</td>
      <td>277.0</td>
      <td>4.0</td>
      <td>Catapult into action and take back the eggs fr...</td>
      <td>75823.0</td>
      <td>Use the staircase catapult to launch Red into ...</td>
      <td>Average</td>
      <td>Bird Island Egg Heist</td>
      <td>...</td>
      <td>4.529934</td>
      <td>4.232244</td>
      <td>47.252546</td>
      <td>14.564673</td>
      <td>421.816401</td>
      <td>4.320282</td>
      <td>4.507081</td>
      <td>4.238505</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6-12</td>
      <td>19.99</td>
      <td>2.0</td>
      <td>168.0</td>
      <td>4.0</td>
      <td>Launch a flying attack and rescue the eggs fro...</td>
      <td>75822.0</td>
      <td>Pilot Pig has taken off from Bird Island with ...</td>
      <td>Easy</td>
      <td>Piggy Plane Attack</td>
      <td>...</td>
      <td>4.490274</td>
      <td>4.235066</td>
      <td>47.252546</td>
      <td>14.564673</td>
      <td>421.816401</td>
      <td>4.320282</td>
      <td>4.507081</td>
      <td>4.238505</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 35 columns</p>
</div>



# Feature Engineering 2
Create a new column for whether or not the theme name contains a trademark (TM) designation.


```python
df.theme_name.value_counts(normalize=True)[:5]
```




    Star Wars™                   0.112335
    DUPLO®                       0.095122
    City                         0.089085
    Juniors                      0.079785
    THE LEGO® NINJAGO® MOVIE™    0.064937
    Name: theme_name, dtype: float64




```python
#Your code here
df['Has_Trademark'] = df.theme_name.str.contains('™')
df.Has_Trademark.value_counts(normalize=True)
```




    False    0.721406
    True     0.278594
    Name: Has_Trademark, dtype: float64



# Feature Engineering 3
Create a new column for whether or not the set name contains a registered (R) designation.


```python
#Your code here
df['Has_Registered'] = df.theme_name.str.contains('®')
df.Has_Registered.value_counts(normalize=True)
```




    False    0.744657
    True     0.255343
    Name: Has_Registered, dtype: float64



# Dealing with Null Values
In future algorithms and applications, having null values can be problematic. Due to this, dealing with null values is a common problem in data science. Below are a few options at your disposal.

# Subsetting the DataFrame
One option for dealing with null values is simply subseting your data to rows without missing values. You can subset a dataframe according to a criterion like this:


```python
subset = df[df.theme_name=='Angry Birds™']
subset.head()
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
      <th>ages</th>
      <th>list_price</th>
      <th>num_reviews</th>
      <th>piece_count</th>
      <th>play_star_rating</th>
      <th>prod_desc</th>
      <th>prod_id</th>
      <th>prod_long_desc</th>
      <th>review_difficulty</th>
      <th>set_name</th>
      <th>star_rating</th>
      <th>theme_name</th>
      <th>val_star_rating</th>
      <th>country</th>
      <th>Theme_Avg_Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6-12</td>
      <td>29.9900</td>
      <td>2.0</td>
      <td>277.0</td>
      <td>4.0</td>
      <td>Catapult into action and take back the eggs fr...</td>
      <td>75823.0</td>
      <td>Use the staircase catapult to launch Red into ...</td>
      <td>Average</td>
      <td>Bird Island Egg Heist</td>
      <td>4.5</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>US</td>
      <td>21.0211</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6-12</td>
      <td>19.9900</td>
      <td>2.0</td>
      <td>168.0</td>
      <td>4.0</td>
      <td>Launch a flying attack and rescue the eggs fro...</td>
      <td>75822.0</td>
      <td>Pilot Pig has taken off from Bird Island with ...</td>
      <td>Easy</td>
      <td>Piggy Plane Attack</td>
      <td>5.0</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>US</td>
      <td>21.0211</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6-12</td>
      <td>12.9900</td>
      <td>11.0</td>
      <td>74.0</td>
      <td>4.3</td>
      <td>Chase the piggy with lightning-fast Chuck and ...</td>
      <td>75821.0</td>
      <td>Pitch speedy bird Chuck against the Piggy Car....</td>
      <td>Easy</td>
      <td>Piggy Car Escape</td>
      <td>4.3</td>
      <td>Angry Birds™</td>
      <td>4.1</td>
      <td>US</td>
      <td>21.0211</td>
    </tr>
    <tr>
      <th>2528</th>
      <td>6-12</td>
      <td>31.1922</td>
      <td>2.0</td>
      <td>277.0</td>
      <td>4.0</td>
      <td>Catapult into action and take back the eggs fr...</td>
      <td>75823.0</td>
      <td>Use the staircase catapult to launch Red into ...</td>
      <td>Average</td>
      <td>Bird Island Egg Heist</td>
      <td>4.5</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>CA</td>
      <td>21.0211</td>
    </tr>
    <tr>
      <th>2529</th>
      <td>6-12</td>
      <td>19.4922</td>
      <td>2.0</td>
      <td>168.0</td>
      <td>4.0</td>
      <td>Launch a flying attack and rescue the eggs fro...</td>
      <td>75822.0</td>
      <td>Pilot Pig has taken off from Bird Island with ...</td>
      <td>Easy</td>
      <td>Piggy Plane Attack</td>
      <td>5.0</td>
      <td>Angry Birds™</td>
      <td>4.0</td>
      <td>CA</td>
      <td>21.0211</td>
    </tr>
  </tbody>
</table>
</div>



You can then chain the `.isnull()` method along with the `~` which negates an expression to remove null values. For example:


```python
print(len(df))
populated = df[~df.theme_name.isnull()] #The tilde (~) negates the conditional, turning all True values False and vice versa
print(len(populated))
```

    12261
    12258


# Removing Null Values
Practice subsetting the dataframe by removing all entries where the star_rating is not populated.


```python
#Your code here
subset = df[~df.star_rating.isnull()]
```

# Imputing Missing Values
Another option for corraling data with missing values is to impute an average (or other) value. For example, rather then dropping all rows where there is no star_rating, we could impute value such as the average star_rating for all sets, or the average star_rating for similar sets. You'll practice an initial example of that here.

# Update the Star Rating Column
Update the star_rating column for those entries where there is no value. Do this by filling in the average value.


```python
#Your code here
avg = df.star_rating.mean()
print(avg)
df.star_rating = df.star_rating.fillna(value=avg)
```

    4.514134009961459

