#https://kyle-stahl-mn.com/stock-portfolio-optimization

from lib import sharedfunctions as fn

from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas
import datetime

ls_symbols = ['ASX-A2M','ASX-WBC','ASX-OZL','ASX-CBA','ASX-WPL','ASX-AEB']

bDate = '2017-12-31'
bDate = datetime.datetime.strptime(bDate, '%Y-%m-%d').date()

df_prices = pandas.DataFrame()

for symbol in ls_symbols:

    filepath = '\\Python\\data\\historical\\' +  symbol  +'.csv'

    df_hdata = pandas.read_csv(filepath)
    df_hdata['DATE'] = pandas.to_datetime(df_hdata['DATE']).apply(lambda x: x.date())

    del df_hdata['OPEN']
    del df_hdata['HIGH']
    del df_hdata['LOW']
    del df_hdata['CLOSE']
    del df_hdata['VOLUME']
    del df_hdata['SOURCE']

    df_hdata[symbol] = df_hdata['ADJ CLOSE']

    del df_hdata['ADJ CLOSE']

    if len(df_prices) == 0:

        df_prices = df_hdata

    else:

        #df = pandas.merge(df, df_hdata, how='outer',left_on=['DATE'],right_on=['DATE'])
        df_prices = pandas.merge(df_prices, df_hdata, on=['DATE'], how='inner')

train = df_prices.loc[(df_prices['DATE'] < bDate)]
test = df_prices.loc[(df_prices['DATE'] >= bDate)]

train = train.set_index('DATE')
test = test.set_index('DATE')

returns = train.pct_change()

returns = returns.iloc[1:] # Day 1 does not have a return so we remove it from the dataframe
returns = returns + 1

# Mean return for each stock (in array)
r = np.array(np.mean(returns, axis=0))


# Covariance matrix between stocks (in array)
S = np.array(returns.cov())

print(S)


# Vector of 1's equal in length to r
e = np.ones(len(r))

# Set the projected mean return for the portfolio
mu = 1+(0.07/252) # 7% rate annually per day

def objective(w):
    return np.matmul(np.matmul(w,S),w)

# Set initial weight values
w = np.random.random(len(r))

# Define Constraints
const = ({'type' : 'ineq' , 'fun' : lambda w: np.dot(w,r) - mu}, # returns - mu >= 0
         {'type' : 'eq' , 'fun' : lambda w: np.dot(w,e) - 1})    # sum(w) - 1 = 0

# Create Bounds
# Creates a tuple of tuples to pass to minimize
# to ensure all weights are betwen [0, inf]
non_neg = []
for i in range(len(r)):
    non_neg.append((0,None))
    
non_neg = tuple(non_neg)

# Run optimization with SLSQP solver
res = minimize(fun=objective, x0=w, method='SLSQP',constraints=const,bounds=non_neg)
w = res.x.round(6)
print(w)
print(w.sum())
print(list(returns.columns[w > 0.0]))

print('      jacobian: ')
print(res.jac)
#print('       elapsed: %s' % (time.time() - t))
print('       success: '+ str(res.success))
print('       message: '+ res.message)
print('objective func: '+ str(res.fun))
print(' # evaluations: '+ str(res.nfev))
print('  # iterations: '+ str(res.nit))


# Invest $100,000 on Jan. 1st 2017
num_shares = w * 100000 / test.iloc[0,]
np.dot(num_shares, test.iloc[0,])
no_short = test.dot(num_shares)


# Mean return for each stock (in array)
r = np.array(np.mean(returns, axis=0))

# Covariance matrix between stocks (in array)
S = np.array(returns.cov())

# Inverse of the covariance matrix
Si = np.linalg.inv(S)

# Vector of 1's equal in length to r
e = np.ones(len(r))

# a, b, c coefficients
a = np.matmul(np.matmul(r,Si),r)
b = np.matmul(np.matmul(r,Si),e)
c = np.matmul(np.matmul(e,Si),e) # same as Si.sum()

# Lambda1 and Lambda2 coefficients
l1 = (c*mu - b) / (a*c - b*b)
l2 = (-b*mu + a) / (a*c - b*b)

# Calculate weights
w = l1*(np.matmul(Si,r)) + l2*(np.matmul(Si,e)) 
print(w)
print(w.sum())
print(list(returns.columns[w != 0.0]))





