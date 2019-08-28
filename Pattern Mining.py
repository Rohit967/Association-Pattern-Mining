#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Please find the frequent pattern and association rules for the data set italy_retial.csv. 
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('italy_retail.csv')

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')

df2 = (df
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


#(1) Please create the binary presentation dataframe of the dataset. You can use pivot_table.
binary_rep = df2.applymap(encode_units)
binary_rep.drop('POSTAGE', inplace=True, axis=1)


#(2) Generate frequent itemsets with min_support=0.1
frequent_itemsets = apriori(binary_rep, min_support=0.1, use_colnames=True)


#(3) Generate association rules by using the frequent itemsets with min_threshold=1
ass_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
ass_rules.head()


#(4) Please describe the interesting relationship you find, and make your recommendations.
df3 = ass_rules[ (ass_rules['lift'] >= 6) &
       (ass_rules['confidence'] >= 0.8) ]

#combination with the products Lunch Bag cars blue and Lunch bag Woodland is having highest LIFT with confidence 1.
# So we can recommend combination of these two products for highers profits.
df2['LUNCH BAG CARS BLUE'].sum()
df2['LUNCH BAG WOODLAND'].sum()


#combination with the products TOY TIDY PINK POLKADOT and TOY TIDY SPACEBOY is having LIFT of 7.6 with confidence 1.
#We can increase the sales of TOY TIDY PINK POLKADOT by recommending it with TOY TIDY SPACEBOY.
df2['TOY TIDY PINK POLKADOT'].sum()
df2['TOY TIDY SPACEBOY'].sum()



#Additional:
#To check which products outlie by Quantity and InvoiceNo
import seaborn as sns
import matplotlib.pyplot as plt 
sns.lmplot('InvoiceNo', 'Quantity', df, hue='Description', fit_reg=False)
fig = plt.gcf()
fig.set_size_inches(12, 9)
plt.show()

