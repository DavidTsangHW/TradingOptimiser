#https://stackoverflow.com/questions/58636704/how-to-maximize-revenue-python

import pandas as pd
import numpy as np
from pulp import *

# Problem Data
df = pd.DataFrame({
    'Time': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Price': [44, 100, 40, 110, 77, 109, 65, 93, 89, 49]})

times = list(df.Time)
times_plus_1 = times + [times[-1] + 1]

# Instantiate maximisation problem
prob = LpProblem("numpy_constraints", LpMaximize)

# Create the problem vairables
# Cash in bank and stock-level at start of each interval
Cash = pulp.LpVariable.dicts("Cash", times_plus_1, cat='Continuous', lowBound=0)
Stock = pulp.LpVariable.dicts("Stock", times_plus_1, cat='Continuous', lowBound=0)

# Amount bought during interval
Buy = pulp.LpVariable.dicts("Buy", times, cat='Continuous')

# Add Objective to problem - cash at end of period modelled
prob += Cash[times_plus_1[-1]]

# Add constraints
# Start with a single dollar in the bank & no stock
prob += Cash[times[0]] == 1.0
prob += Stock[times[0]] == 0.0

# Cash & stock update rules
for t in times:
    prob += Cash[t+1] == Cash[t] - Buy[t]*df.Price[t]
    prob += Stock[t+1] == Stock[t] + Buy[t]

# Solve
prob.solve()

# Check when we bought when:
Buy_soln = np.array([Buy[t].varValue for t in times])
print("Buy_soln:")
print(Buy_soln)

Stock_soln = np.array([Stock[t].varValue for t in times_plus_1])
print("Stock_soln:")
print(Stock_soln)

Cash_soln = np.array([Cash[t].varValue for t in times_plus_1])
print("Cash_soln:")
print(Cash_soln)
