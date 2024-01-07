import pandas as pd

def price(row):
   if row['PRICE'] > 400:
      return 'HIGH'
   return 'LOW'

def clean_df(path):
   df = pd.read_csv(path, sep = '|')
   df = df[['NAME', 'PRICE', 'CUSINE_CATEGORY']]
   df = df[df.NAME != 'NAME']
   df.dropna(subset=['NAME'], inplace=True)
   df['CUISINE_LVL1'] = df['CUSINE_CATEGORY'].str.split(',').str[0]
   df['CUISINE_LVL2'] = df['CUSINE_CATEGORY'].str.split(',').str[1]
   df['CUISINE_LVL3'] = df['CUSINE_CATEGORY'].str.split(',').str[2]
   df.PRICE = pd.to_numeric(df.PRICE, errors= 'coerce')
   df['COST'] = df.apply(price, axis=1)
   df = df.drop(['CUSINE_CATEGORY', 'PRICE'], axis = 1)
   df['NAME'] = df['NAME'].str.title()
   df['COST'] = df['COST'].str.title()
   print('Saving clean file now...')
   df.to_csv('data/Zomato_Mumbai_Dataset_clean.csv')
   return df

if __name__ == '__main__':
    clean_df('data/Zomato_Mumbai_Dataset.csv')

