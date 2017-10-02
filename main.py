# coding=utf8
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

import parameters as pm 

RS = 20171002
np.random.seed(RS)

ROUNDS = 450
params = {
	'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'learning_rate': 0.04,
    'verbose': 0,
    'num_leaves': 2 ** 5,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': RS,
    'feature_fraction': 0.7,
    'feature_fraction_seed': RS,
    'max_bin': 100,
    'max_depth': 5,
    'num_rounds': ROUNDS
}



print("Started")
input_folder = './'
train_df = pd.read_csv(input_folder + 'train.csv', parse_dates=['timestamp'])
test_df  = pd.read_csv(input_folder + 'test.csv' , parse_dates=['timestamp'])
macro_df = pd.read_csv(input_folder + 'macro.csv', parse_dates=['timestamp'])

#fix outlier
train_df.drop(train_df[train_df["life_sq"] > 5000].index, inplace=True)

#fix wrong pricing of train_df
#train_df.drop(10905, inplace=True)


def clean(df):
    #fix build year
    df.loc[df[df["build_year"] < 1000].index, "build_year"] = df["build_year"].median()
    df.loc[df[df["build_year"] > 2018].index, "build_year"] = df["build_year"].median()
    #fix full sq & life sq
    df.loc[df[df["life_sq"] < 7].index, "life_sq"] = np.nan   
    df.loc[df[df["full_sq"] < 6].index, "full_sq"] = np.nan    
    df.loc[df[df["life_sq"] > df["full_sq"]].index, "full_sq"] = np.nan
    #fix max_floor
    df.loc[df[df["floor"] > df["max_floor"]].index, "max_floor"] = np.nan
    #fix kitch_sq
    df.loc[df[df["kitch_sq"] > df["full_sq"]].index, "life_sq"] = np.nan


    return df

train_df = clean(train_df)
test_df = clean(test_df)

train_y  = train_df.copy()
test_ids = test_df['id']

train_df.drop(['id', 'price_doc'], axis=1, inplace=True)
test_df.drop(['id'], axis=1, inplace=True)
print("Data: X_train: {}, X_test: {}".format(train_df.shape, test_df.shape))

# Predict product
test_df['product_type'] = test_df['product_type'].fillna(method = 'backfill')

train_inv = train_df.loc[train_df['product_type'] == 'Investment']
train_oro = train_df.loc[train_df['product_type'] == 'OwnerOccupier']
test_inv = test_df.loc[test_df['product_type'] == 'Investment']
test_oro = test_df.loc[test_df['product_type'] == 'OwnerOccupier']

train_y_inv = np.log(train_y.loc[train_inv.index.values.tolist()]['price_doc'].values)
train_y_oro = np.log(train_y.loc[train_oro.index.values.tolist()]['price_doc'].values)

df_inv = pd.concat([train_inv, test_inv])
df_oro = pd.concat([train_oro, test_oro])



def generate_set(df, train_df, isInv):
    #Lets try using only those from https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity
    macro_cols = ["timestamp","balance_trade","balance_trade_growth","eurrub","average_provision_of_build_contract","micex_rgbi_tr","micex_cbi_tr","deposits_rate","mortgage_value","mortgage_rate","income_per_cap","museum_visitis_per_100_cap","apartment_build"]
    df = df.merge(macro_df[macro_cols], on='timestamp', how='left')
    print("Merged with macro: {}".format(df.shape))
    
    #Dates...
    df['year'] = df.timestamp.dt.year
    df['month'] = df.timestamp.dt.month
    df['dow'] = df.timestamp.dt.dayofweek
    df.drop(['timestamp'], axis=1, inplace=True)
    
    #More featuers needed...
    
    df_num = df.select_dtypes(exclude=['object'])
    df_obj = df.select_dtypes(include=['object']).copy()
    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]
    
    df_values = pd.concat([df_num, df_obj], axis=1)
    
    if isInv:
        items = pm.items_inv
    else:
        items = pm.items_oro
        
    for item in items:
        # full_sq / material
        m_fs = df_values.groupby([item], as_index = False)['full_sq'].mean()
        m_fs.columns = [item, 'full_sq/' + item]
        df_values = df_values.merge(m_fs, how = 'left', on = item)
        df_values['full_sq/'+item] = df_values['full_sq'] / df_values['full_sq/'+item]
    
    df_values.drop(['full_sq'], axis=1, inplace=True)
    
    
    df_values['floor/max_floor'] = df_values['floor'] / df_values['max_floor']
    
    
    pos = train_df.shape[0]
    train_df = df_values[:pos]
    test_df  = df_values[pos:]
    del df, df_num, df_obj, df_values
    
    return train_df, test_df



train_inv_set, test_inv_set = generate_set(df_inv, train_inv, isInv = True)
train_oro_set, test_oro_set = generate_set(df_oro, train_oro, isInv = False)

#drop wrong pricing
train_y_oro_pd = pd.DataFrame(train_y_oro)
for price_ix in (pm.wpl_oro_gt + pm.wpl_oro_ls):
    train_oro_set.drop(price_ix, inplace=True)
    train_y_oro_pd.drop(price_ix, inplace=True)

train_y_oro = train_y_oro_pd.values.flatten()

train_y_inv_pd = pd.DataFrame(train_y_inv)
for price_ix in (pm.wpl_inv_gt + pm.wpl_inv_ls):
    train_inv_set.drop(price_ix, inplace=True)
    train_y_inv_pd.drop(price_ix, inplace=True)

train_y_inv = train_y_inv_pd.values.flatten()


def train_model(train_df, train_y, test_df):
    print("Training on: {}".format(train_df.shape, train_y.shape))
    
    train_lgb = lgb.Dataset(train_df, train_y)
    model = lgb.train(params, train_lgb, num_boost_round=ROUNDS)
    preds = model.predict(test_df)
    	
    print("Features importance...")
    gain = model.feature_importance('gain')
    ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    print(ft.head(25))
    
    
    score = lgb.cv(params, train_lgb, metrics = "l2_root")

    '''
    plt.figure()
    ft[['feature','gain']].head(25).plot(kind='barh', x='feature', y='gain', legend=False, figsize=(10, 20))
    plt.gcf().savefig('features_importance.png')
    '''
    return model, preds, score

model_inv, preds_inv, score_inv = train_model(train_inv_set, train_y_inv, test_inv_set)
model_oro, preds_oro, score_oro = train_model(train_oro_set, train_y_oro, test_oro_set)

rmse_inv =  rms = sqrt(mean_squared_error(train_y_inv, model_inv.predict(train_inv_set)))
print ("investment rmse score = {}".format(rmse_inv))
print ("investment cv score = {}".format(np.mean(score_inv['rmse-mean'])))

rmse_oro =  rms = sqrt(mean_squared_error(train_y_oro, model_oro.predict(train_oro_set)))
print ("owneroccupied rmse score = {}".format(rmse_oro))
print ("owneroccupied cv score = {}".format(np.mean(score_oro['rmse-mean'])))

temp = pd.concat([pd.DataFrame(np.exp(train_y_oro)), pd.DataFrame(np.exp(model_oro.predict(train_oro_set)))], axis = 1)
temp['diff'] = temp.iloc[:,0] - temp.iloc[:,1]
temp['diff %'] = temp['diff'] * 100.0 / temp.iloc[:,0]
temp['diff % 2'] = temp['diff'] * 100.0 / temp.iloc[:,1]
pd.DataFrame(temp[temp['diff %'] < -100].index).to_csv("oro_ls.csv", index = False)
pd.DataFrame(temp[temp['diff % 2'] > 100].index).to_csv("oro_gt.csv",index = False)


temp_inv = pd.concat([pd.DataFrame(np.exp(train_y_inv)), pd.DataFrame(np.exp(model_inv.predict(train_inv_set)))], axis = 1)
temp_inv['diff'] = temp_inv.iloc[:,0] - temp_inv.iloc[:,1]
temp_inv['diff %'] = temp_inv['diff'] * 100.0 / temp_inv.iloc[:,0]
temp_inv['diff % 2'] = temp_inv['diff'] * 100.0 / temp_inv.iloc[:,1]
pd.DataFrame(temp_inv[temp_inv['diff %'] < -100].index).to_csv("inv_ls.csv", index = False)
pd.DataFrame(temp_inv[temp_inv['diff % 2'] > 100].index).to_csv("inv_gt.csv",index = False)
preds_inv = pd.DataFrame(data = preds_inv, index = test_inv.index)
preds_oro = pd.DataFrame(data = preds_oro, index = test_oro.index)
preds = pd.concat([preds_inv, preds_oro]).sort_index().values.flatten()

print("Writing output...")
out_df = pd.DataFrame({"id":test_ids, "price_doc":np.exp(preds)})
out_df.to_csv("lgb_{}_{}.csv".format(ROUNDS, RS), index=False)
print(out_df.head(3))


print("Done.")
