import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pandas.tseries.offsets import MonthEnd
import psycopg2
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

conn = psycopg2.connect(
    "dbname='dwh' user='moksha' host='dwh.cnqejzhenq8g.us-east-2.rds.amazonaws.com' password='mokshalentesplus2021'")
conn.autocommit = True

#def item_counts_df(conn, country):
def item_counts_df(conn):
    query = "select campaign_id, campaign_name, mkt.campaigns.business_unit, mkt.campaigns.start, mkt.campaigns.end, sku from mkt.campaigns inner join mkt.discounts using(campaign_id) "
    campaign_data = pd.read_sql(query, conn)
    item_query = "select catalog.skus.sku, catalog.skus.business_unit, catalog.items.type from catalog.skus inner join catalog.items using(item_id) where catalog.skus.empresa = 'lentesplus'"
    item_data = pd.read_sql(item_query, conn)
    merged = pd.merge(campaign_data, item_data, on='sku', how='inner').drop_duplicates()
    campaigns_item_counts = merged.pivot_table(index='campaign_id', columns='type', aggfunc='size').fillna(0).drop(
        ['Cuidado y Belleza', 'Gafas'], axis=1).reset_index()
    campaigns_item_counts['lenses'] = campaigns_item_counts['Lentes de contacto'] + campaigns_item_counts[
        'Lentes de prueba']
    campaigns_item_counts = campaigns_item_counts.drop(['Lentes de contacto', 'Lentes de prueba'], axis=1)
    unique_campaigns = campaign_data.drop('sku', axis=1).drop_duplicates()
    campaigns = pd.merge(unique_campaigns, campaigns_item_counts, on='campaign_id', how='inner')
    campaigns = campaigns.rename(columns={'Gotas': 'drops', 'Solución': 'solutions'})
    campaigns = campaigns.loc[campaigns['business_unit'].isin(['AR', 'CL', 'MX', 'CO'])]
    # campaigns = campaigns.drop('sku', axis =1).drop_duplicates()
    return campaigns, item_data


#def target_var(conn, campaigns, country):
def target_var(conn, campaigns, country):
    # query = "select campaign_name, created_at, sku ,gmv_usd, gmv, discount_price, discount_rule from silver.sales_products where sales_products.empresa = 'lentesplus' and campaign_name <> 'Tier Pricing' and business_unit= %s"
    # sales = pd.read_sql(query, conn, params = [country]).dropna()
    query = "select campaign_name, created_at, sku ,gmv_usd, gmv, discount_price, discount_rule from silver.sales_products where sales_products.empresa = 'lentesplus' and campaign_name <> 'Tier Pricing'"
    sales = pd.read_sql(query, conn).dropna()
    merged = pd.merge(sales, campaigns, on='campaign_name', how='inner')
    merged_correct = merged.loc[
        (merged['created_at'] <= merged['end']) & (merged['created_at'] >= merged['start'])]
    targets = merged_correct.groupby('campaign_id')['gmv_usd'].sum().reset_index()
    return targets, sales


#def campaign_types(conn, country):
def campaign_types(conn):

    # query = "select campaign_id, tipo_descuento from mkt.campaigns where business_unit = %s"
    # types = pd.read_sql(query, conn, params = [country])
    query = "select campaign_id, tipo_descuento from mkt.campaigns"
    types = pd.read_sql(query, conn)
    types_modified = types['tipo_descuento'].str.lower()
    mod_types_bool = types_modified.str.contains('tachado') | types_modified.str.contains(
        'storewide') | types_modified.str.contains(
        'dto 10%') | types_modified.str.contains('descuento porcentual') | types_modified.str.contains(
        '20% en todo el portafolio') | types_modified.str.contains('20% todo el portafolio')
    types['tipo_descuento'] = mod_types_bool
    return types


#def rankings(conn, campaigns, sales, country):
def rankings(conn, campaigns, sales):
    query_ranks = "select to_char(created_at, 'YYYY-MM') as dates, sku, business_unit, row_number() over(partition by to_char(created_at, 'YYYY-MM'), business_unit order by sum(quantity_actual) desc) as ranking from silver.sales_products where sales_products.empresa = 'lentesplus' and coalesce(campaign_name, '') <> 'Tier Pricing' and to_char(created_at, 'YYYY-MM') >= '2020-03' group by 1, 2, 3 order by 1 desc"
    ranks = pd.read_sql(query_ranks, conn)
    merge_s_and_c = pd.merge(campaigns, sales, on='campaign_name', how='inner')
    merge_s_and_c = merge_s_and_c.loc[
        (merge_s_and_c['created_at'] >= merge_s_and_c['start']) & (merge_s_and_c['created_at'] <= merge_s_and_c['end'])]
    merge_s_and_c['month_and_year'] = pd.to_datetime(
        merge_s_and_c['start'].dt.year.astype(str) + '/' + merge_s_and_c['start'].dt.month.astype(str))
    merge_s_and_c['prev_month'] = merge_s_and_c['month_and_year'] - pd.DateOffset(months=1)
    merge_s_and_c['prev_month'][merge_s_and_c['prev_month'] == pd.to_datetime('2020/02/01')] = pd.to_datetime(
        '2020/03/01')
    ranks['dates'] = pd.to_datetime(ranks['dates'])
    merge_trial = pd.merge(merge_s_and_c, ranks, left_on=['prev_month', 'sku'], right_on=['dates', 'sku'])[
        ['campaign_id', 'sku', 'ranking']].drop_duplicates()
    grouped = merge_trial.loc[(merge_trial['ranking'] >= 1) & (merge_trial['ranking'] <= 10)].groupby('campaign_id')[
        'ranking'].count().reset_index()

    return grouped, merge_s_and_c, ranks



#def targets(conn, country):
def targets(conn):
    # query = "select country, mkt.targets.month, target,mkt.targets.type from mkt.targets where empresa = 'lentesplus' and (type = 'new customers' or type = 'recurrent customers') and business_unit = %s"
    # targets = pd.read_sql(query, conn, params = [country])
    query = "select country, mkt.targets.month, target,mkt.targets.type from mkt.targets where empresa = 'lentesplus' and (type = 'new customers' or type = 'recurrent customers')"
    targets = pd.read_sql(query, conn)
    target_2 = targets.groupby(['month', 'country'])['target'].sum().reset_index()
    target_2['month_only'] = target_2['month'].dt.month
    target_2['year_only'] = target_2['month'].dt.year
    target_2['month_and_year'] = target_2['year_only'].astype(str) + '/' + target_2['month_only'].astype(str)
    target_2['month_and_year'] = pd.to_datetime(target_2['month_and_year'])
    target_2 = target_2.drop(['month', 'month_only', 'year_only'], axis=1)
    return target_2


#def shares(conn, merge_s_and_c, country):
def shares(conn, merge_s_and_c):
    query = "select sku, to_char(created_at, 'YYYY-MM'), business_unit, sum(gmv_usd) / (sum(sum(gmv_usd)) over (partition by to_char(created_at, 'YYYY-MM'), business_unit)) as share from silver.sales_products where empresa = 'lentesplus' and coalesce(campaign_name, '') <> 'Tier Pricing' and to_char(created_at, 'YYYY-MM') >= '2020-03' group by sku, to_char(created_at, 'YYYY-MM'), business_unit order by share desc"
    sale_shares = pd.read_sql(query, conn)
    sale_shares['to_char'] = pd.to_datetime(sale_shares['to_char'])
    shared_merged = pd.merge(merge_s_and_c, sale_shares, left_on=['sku', 'prev_month'],
                             right_on=['sku', 'to_char'], how='inner')
    filtered = shared_merged[['campaign_id', 'business_unit_x', 'start', 'end', 'sku', 'share']].drop_duplicates()
    my_shares = filtered.groupby('campaign_id')['share'].sum().reset_index()
    return my_shares, sale_shares


def discount(conn, merge_s_and_c):
    discount_total = merge_s_and_c.assign(
        total_discount=merge_s_and_c['discount_price'] + merge_s_and_c['discount_rule'])
    discount_total['discount_pct'] = discount_total['total_discount'] / discount_total['gmv']
    discounts = discount_total.groupby(['sku', 'campaign_id'])['discount_pct'].mean().reset_index()
    campaign_discounts = discounts.groupby('campaign_id')['discount_pct'].mean().reset_index()
    return campaign_discounts

#def final_df(conn, country):
def final_df(conn):
    #item_output = item_counts_df(conn, country)
    item_output = item_counts_df(conn)
    campaigns = item_output[0]
    #target_output = target_var(conn, campaigns, country)
    target_output = target_var(conn, campaigns)
    target_variables = target_output[0]
    sales = target_output[1]
    #types = campaign_types(conn, country)
    types = campaign_types(conn)
    #ranking_output = rankings(conn, campaigns, sales, country)
    ranking_output = rankings(conn, campaigns, sales)
    top_10_counts = ranking_output[0]
    merge_s_and_c = ranking_output[1]
    #cmp_targets = targets(conn, country)
    cmp_targets = targets(conn)
    share_output = shares(conn, merge_s_and_c)
    cmp_shares = share_output[0]
    discounts = discount(conn, merge_s_and_c)
    final_v1 = pd.merge(campaigns, target_variables, on='campaign_id', how='inner')
    final_v2 = pd.merge(final_v1, types, on='campaign_id', how='inner')
    final_rank_10 = pd.merge(final_v2, top_10_counts, on='campaign_id', how='left')
    final_rank_10['ranking'] = final_rank_10['ranking'].fillna(0)
    final_trial = final_rank_10.assign(month=final_rank_10['start'].dt.month.astype(str))
    final_trial['year'] = final_trial['start'].dt.year.astype(str)
    final_trial['month_and_year'] = pd.to_datetime(final_trial['year'] + '/' + final_trial['month'])
    final_v3 = pd.merge(final_trial, cmp_targets, left_on=['month_and_year', 'business_unit'],
                        right_on=['month_and_year', 'country'],
                        how='inner')
    final_v3.drop(['month', 'year', 'month_and_year', 'country'], axis=1, inplace=True)
    final_v4 = pd.merge(final_v3, cmp_shares, on='campaign_id', how='inner')
    final_total = pd.merge(final_v4, discounts, on='campaign_id', how='inner')
    final_total['month'] = final_total['start'].dt.month
    final_total['year'] = final_total['start'].dt.year
    final = final_total.loc[final_total['end'] < pd.datetime.now()]
    final['duration'] = (final['end'] - final['start']).dt.days
    return final, share_output[1], ranking_output[2], item_output[1]

def model_fit(my_df, max_depth, min_samples_split, min_samples_leaf, n_estimators):

    features = ['drops', 'solutions', 'lenses', 'tipo_descuento', 'ranking', 'target', 'share', 'discount_pct', 'month',
                'year', 'duration']

    training_data = my_df[features]

    training_labels = my_df[['gmv_usd']]
    cat_cols = ['month', 'year']
    quant_cols = ['drops', 'solutions', 'lenses', 'ranking', 'target', 'share', 'discount_pct', 'duration']
    pl_cat = Pipeline([('one-hot', OneHotEncoder(handle_unknown='ignore'))])
    pl_num = Pipeline([('scale', StandardScaler())])
    preproc = ColumnTransformer(transformers=[('scaler', pl_num, quant_cols), ('one-hot', pl_cat, cat_cols)],
                                remainder='passthrough')

    pl_rf = Pipeline(steps=[('preprocessor', preproc), ('rf', RandomForestRegressor(max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf =  min_samples_leaf, n_estimators = n_estimators))])

    pl_rf.fit(training_data, training_labels)
    return pl_rf

def predictor(drops, solutions, lenses, tipo_descuento, ranking, target, share, discount_pct, month, year,
              duration, features_df, max_depth, min_samples_split, min_samples_leaf, n_estimators):
    df = pd.DataFrame(
        {'drops': drops, 'solutions': solutions, 'lenses': lenses, 'tipo_descuento': tipo_descuento, 'ranking': ranking,
         'target': target, 'share': share, 'discount_pct': discount_pct, 'month': month, 'year': year,
         'duration': duration}, index=[0])

    predictions = []
    for i in range(5):
        pl = model_fit(features_df, max_depth, min_samples_split, min_samples_leaf, n_estimators)
        predictions.append(pl.predict(df)[0])
    return predictions, {'min': min(predictions), 'max': max(predictions), 'mean': (sum(predictions) / len(predictions))}

def query(month, year, sku_list,share_data, rank_data, item_data):
    to_return = []
    if str(month) == '3' and str(year) == '2020':
        filter_on = '2020-03'
    else:
        date = pd.to_datetime(str(year) + '/' + str(month))
        prev_month_start = (date - pd.DateOffset(months=1)).date()
        prev_month = str(prev_month_start.month).zfill(2)
        prev_year = str(prev_month_start.year)
        filter_on = prev_year + '-' + prev_month
    rank_data = rank_data.loc[rank_data['dates'] == filter_on]

    rank_data_filtered = rank_data.loc[rank_data['sku'].isin(sku_list)]
    ranks_corrected = rank_data_filtered.loc[(rank_data_filtered['ranking'] >= 1) & (rank_data_filtered['ranking'] <= 10)]
    ranking = len(ranks_corrected)
    share_data = share_data.loc[share_data['to_char'] == filter_on]
    share_data_filtered = share_data.loc[share_data['sku'].isin(sku_list)]
    share = share_data_filtered['share'].sum()
    item_data_filtered = item_data.loc[item_data['sku'].isin(sku_list)].drop_duplicates()
    item_data_filtered['type'].replace({'Lentes de contacto': 'lenses', 'Lentes de prueba': 'lenses'}, inplace=True)
    my_series = item_data_filtered['type'].value_counts()
    to_return.append(ranking)
    to_return.append(share)
    if 'lenses' in my_series.index:
        lenses = my_series.loc['lenses']
        to_return.append(lenses)
    else:
        to_return.append(0)
    if 'Solución' in my_series.index:
        solutions = my_series.loc['Solución']
        to_return.append(solutions)
    else:
        to_return.append(0)
    if 'Gotas' in my_series.index:
        drops = my_series.loc['Gotas']
        to_return.append(drops)
    else:
        to_return.append(0)
    return to_return

def hyperparameters(model_df):
    #st.markdown("Checking for parameters")
    parameters = {
        'max_depth': [3, 5, 7, 10, 13, 15, 18, None],
        'min_samples_split': [2, 3, 5, 7, 10, 15, 20],
        'min_samples_leaf': [2, 3, 5, 7, 10, 15, 20],
        'n_estimators': [400, 800, 1000, 1400, 1800, 2000]
    }
    clf = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = parameters, n_iter= 20,  cv=3)
    clf.fit(model_df[['drops', 'solutions', 'lenses', 'tipo_descuento', 'ranking', 'target', 'share', 'discount_pct', 'month',
                'year', 'duration']], model_df[['gmv_usd']])
    #st.markdown(clf.best_params_)
    return clf.best_params_

def main():
    st.title('Campaign Sales Prediction App')
    st.subheader('Enter campaign features for predictions')
    country = st.selectbox('Country', ('CL', 'CO', 'MX', 'AR'))
    tipo_descuento = st.selectbox('Price Discount: Select 1 for yes and 0 for no', (0,1))
    target = st.text_input('Target Customers for the month', 650)
    #discount_pct = st.text_input('Discount Percent', 0.05)
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    month = st.select_slider('month of campaign', range(1, 13))
    year = st.text_input('campaign year', 2021)
    duration = st.text_input('duration', 10)

    if st.button('Download Data'):
        #model_outputs = final_df(conn, country)
        model_outputs = final_df(conn)
        model_df = model_outputs[0]
        share_data = model_outputs[1]
        rank_data = model_outputs[2]
        item_data = model_outputs[3]
        best_params = hyperparameters(model_df)
        param_df = pd.DataFrame(best_params, index = [0])
        param_df.to_csv('best_params.csv')
        model_df.to_csv('model_df.csv')
        share_data.to_csv('share_data.csv')
        rank_data.to_csv('rank_data.csv')
        item_data.to_csv('item_data.csv')

    if st.button('Get Prediction'):
        try:

            model_df = pd.read_csv('model_df.csv', index_col = [0])
            model_df = model_df.loc[model_df['business_unit']==country]
            share_data = pd.read_csv('share_data.csv', index_col=[0])
            share_data = share_data.loc[share_data['business_unit'] == country]
            rank_data = pd.read_csv('rank_data.csv', index_col = [0])
            rank_data = rank_data.loc[rank_data['business_unit'] == country]
            item_data = pd.read_csv('item_data.csv', index_col = [0])
            item_data = item_data.loc[item_data['business_unit'] == country]
            df = pd.read_csv(data_file)

            sku_list = list(df['sku'])

            query_output = query(month, year, sku_list, share_data, rank_data, item_data)

            drops = query_output[4]

            solutions = query_output[3]

            lenses = query_output[2]

            ranking = query_output[0]

            share = query_output[1]

            parameters = pd.read_csv('best_params.csv', index_col =[0])

            max_depth = parameters['max_depth'].iloc[0]

            min_samples_split = parameters['min_samples_split'].iloc[0]

            min_samples_leaf = parameters['min_samples_leaf'].iloc[0]

            n_estimators = parameters['n_estimators'].iloc[0]

            discount = df['discount'].mean()

            predictions = predictor(float(drops), float(solutions), float(lenses), bool(tipo_descuento), float(ranking),
                                    float(target), float(share), float(discount), int(month), int(year), int(duration),
                                    model_df, max_depth, min_samples_split, min_samples_leaf, n_estimators)

            st.success(predictions[0])
            st.success(predictions[1])
        except:
            st.markdown("Please download the data first!")




    return "Click predict to make predictions!"

if __name__ == "__main__":
    main()






