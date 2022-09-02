import pandas as pd
import numpy as np
def get_data(num = 123):
    num = str(num)
    input = pd.read_excel(f'./dataset/input_{num}.xlsx')[['corp_id','invoice_date','AT','status']]
    input = input[input['status']==1]
    inp = input[['corp_id','invoice_date']].drop_duplicates()
    output = pd.read_excel(f'./dataset/output_{num}.xlsx')[['corp_id','invoice_date','AT','status']]
    output = output[output['status']==1]
    outp = output[['corp_id','invoice_date']].drop_duplicates()

    input_sum = input.groupby(['corp_id','invoice_date']).sum()['AT']
    output_sum = output.groupby(['corp_id','invoice_date']).sum()['AT']

    inp['AT_in'] = np.array(input_sum)
    outp['AT_out'] = np.array(output_sum)

    invoice = pd.merge(left = inp,right = outp, on = ['corp_id','invoice_date'],how = 'outer')

    #inp = inp.resample('1M')
    return invoice

def re_agg(data):
    data = data.fillna(1e-8)
    from numpy import datetime64
    data = data[(data['invoice_date']>= datetime64('2017-01-01')) & (data['invoice_date'] < datetime64('2020-01-01'))]
    start_date = data['invoice_date'].min()
    peroid = pd.DataFrame(pd.date_range(data['invoice_date'].min(), periods=36, freq="M"),columns=['date'])

    ID = np.array(data['corp_id'].drop_duplicates())
    df = pd.DataFrame([])

    for id in ID:
        e1 = data[data['corp_id']==id]
        #e1.index = e1['invoice_date']
        e1 = e1[['AT_in','AT_out','invoice_date']]

        e1 = pd.merge(left = e1,right= peroid,left_on = 'invoice_date',right_on='date',how='right')
        e1.index = e1['date'] #type: ignore
        e1 = e1[['AT_in','AT_out']]
        e2 = e1
        e2['refund_in'] = e2[e2['AT_in']<0]['AT_in']
        e2['refund_out'] = e2[e2['AT_out']<0]['AT_out']
        e2 = e2.resample('1M').count()
        e1 = e1.resample('1M').sum()

        e1['ref_in'] = e2['refund_in']/(e2['AT_in']+1e-8)
        e1['ref_out'] = e2['refund_out']/(e2['AT_out']+1e-8)
        
        e1['sales'] = e1['AT_out'] - e1['AT_in']
        for i in range(1, 4, 1):
            e1[f'sales_lag_{i}'] = e1['sales'].transform(lambda x:x.shift(i)) #type: ignore
            e1[f'sales_lag_{i}_diff'] = e1['sales'].transform(lambda x:x.shift(i).diff()) #type: ignore
        e1['corp_id'] = id
        df = pd.concat([df,e1]) #type: ignore
        
    df['date'] = df.index
    df.index = range(len(df)) #type: ignore
    return df

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from math import nan
def get_tesors(data,input_length,label_length,total_length,batch_size):
    rate = pd.read_excel(f'./dataset/info_{batch_size}.xlsx',header = 0)
    features = [c for c in data.columns if c not in ['corp_id','date','AT_in','AT_out']]
    
    ratings = {'A':0,'B':1,'C':2,'D':3,nan:'unknown'}
    defaults = {'是':1,'否':0,nan:'unknown'}
    if batch_size == 123:
        rate['rating'],rate['default'] = [ratings[c] for c in rate['rating']],[defaults[c] for c in rate['default']]
    label = rate[['rating','default']]

    standardscaler = StandardScaler()
    X_ss = standardscaler.fit_transform(np.asarray(data[features]))

    num_features = len(features)
    X_ss = tf.reshape(tensor= tf.constant(X_ss,dtype=tf.float16),shape=[batch_size,total_length,num_features])
    input,test = X_ss[:,:2*input_length,:],X_ss[:,input_length:,:]
    label = tf.reshape(tensor= tf.constant(label,dtype=tf.float16),shape=[batch_size,label_length,2])

    return input,test,label


from sklearn.preprocessing import MinMaxScaler
def get_output(test,model):
    output = tf.squeeze(model(test))
    rate = np.array(np.array(output[:,0]),dtype=int)
    rate = np.where(rate>3,3,rate)
    rate = np.where(rate<0,0,rate)
    minmax = MinMaxScaler()
    dft= minmax.fit_transform(np.array(output[:,1]).reshape(-1, 1))
    result = pd.DataFrame({
    "rating": list(rate),
    "default": list(dft)
    })
    return result