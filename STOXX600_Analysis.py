import numpy as np
import pandas as pd

def MA_strategy(MA, strategy, short_MA, rebalance_freq, rebalance_cost, long_short):
    "Print the return of a long only strategy and a MA strategy on the STOXX600"
    "From 2014-2021, based on MA period, rebalancing frequency and"
    "Whether the strategy is short or neutral when MA price is below observed price"
    "For short positions use -1 for netural use 0"
    # MA indicates the length of the moving average calculation
    # strategy = 1 for prefixed rebalance freq with position being long if price is below MA as MA is seen as intrinsic value and markets will mean revert
    # strategy = 2 for crossover strategy where a price above the MA price is seen as an upward trending price and position is then long
    # strategy = 3 for a golden/dead cross strategy where if the short-term MA is above the long-term MA it is a buy signal and if below a sell signal
    # short_MA indicates the short-term MA in the golden/dead cross strategy. Needs to be lower than the MA variable
    # rebalance_freq is how often rebalancing should occur when using strategy 1
    # rebalance_cost is the cost every time the portfolio is changing from long to neutral/short or the other way around - normally assuming 25bps
    # long_short variable indicates whether the position should be short (-1) or just out of the market (0) when not long

    prices = pd.read_csv(r'C:/Users/nikla/.spyder-py3/STOXX 600 Historical Data, Daily.csv')
    prices = prices[['Date', 'Price']]
    # Reverts the dataframe so day 0 is first row
    prices = prices.iloc[::-1].reset_index(drop=True)
    prices['Return'] = (prices['Price'] / prices['Price'].shift()) - 1
    #Calculates the Moving Average price using Moving Average frequency input
    prices['MA_price'] = prices['Price'].rolling(MA).mean()

    # slices the dataframe to only include rows where a MA price exist + reset index
    prices = prices.iloc[MA::].reset_index(drop=True)
   
    #Creating a 'Day' column used to asses rebalancing dates, equal to the index
    prices['Day'] = range(0, len(prices))

    # fixed rebalancing frequency strategy
    if strategy == 1:
        # Using list comprehension to rebalance on rebalancing dates, by changing Position column, other rows
        # is set to None.  Cannot set position to position.shift in latest else
        # as list comprehension result cannot look at previous rows
        prices['Position'] = [1 if z % rebalance_freq == 0 and x >= y
                              else long_short if z % rebalance_freq == 0 and x < y
                              else None
                              for z, x, y in zip(prices['Day'], prices['MA_price'].shift(), prices['Price'].shift())]
    
        # Sets the initial position until first rebalancing
        # .loc and .iloc returns a Pandas series or DataFrame, why np.values needs to be used
        # as it converts it to values (float) that can be used in if statements
        if prices['MA_price'].values[0] >= prices['Price'].values[0]:
            prices['Position'].values[1] = 1
        else:
            prices['Position'].values[1] = long_short
            prices['Position'].values[2:rebalance_freq] = prices['Position'].values[1]

        # Populates remainig position values in between rebalancing dates as equal to latest position set
        # at previous rebalancing. Omits index 0 as that position is always nan.
        for i in range(1, len(prices)):
            if pd.isna(prices.iloc[i, 5]):
                prices['Position'].values[i] = prices['Position'].values[i-1]
                
    # crossover strategy
    elif strategy == 2:
        prices['Position'] = None
        for i in range(1, len(prices)):
            if prices['Price'].values[i-1] > prices['MA_price'].values[i-1]:
                prices['Position'].values[i] = 1
            else:
                prices['Position'].values[i] = long_short
                
    # golden/dead cross strategy
    # strategy is long (1) until first short_MA_price exist
    elif strategy == 3:
        prices['Position'] = None
        prices['short_MA_price'] = prices['Price'].rolling(short_MA).mean()
        for i in range(1, len(prices)):
            if prices['short_MA_price'].values[i-1] > prices['MA_price'].values[i-1]:
                prices['Position'].values[i] = 1
            elif pd.isna(prices.iloc[i, 6]):
                prices['Position'].values[i] = 1
            else:
                prices['Position'].values[i] = long_short
        # manualy sets the last day before short_MA price appears to 1 as well, as position is calculated with a row lag (i-1)
        prices['Position'].values[short_MA-1] = 1       

    prices['MA_return'] = prices['Return'] * prices['Position']
    
    # Makes sure the long index is calculated from day 1, so time series have same length as the MA strategy
    # MA strategy starts at day 1 and not 0, as day 0 is used to evaluate whether to take a long or short/neutral position on the first day
    prices['Long_index'] = None
    for i in range(1, len(prices)):
        prices['Long_index'].values[i] = 1 + prices['Return'].values[i] 
    
    prices['MA_index'] = 1 + prices['MA_return']
    prices['Long_wealth_index'] = prices['Long_index'].cumprod()
    prices['MA_wealth_index'] = prices['MA_index'].cumprod()

    # Setting trade cost column to 0 as default but needs to be float
    prices['Trade_cost'] = 0.0

    for i in range(1, len(prices)):
        if prices.iloc[i, 5] != prices.iloc[i-1, 5]:
            prices['Trade_cost'].values[i] = prices['MA_wealth_index'].values[i] * rebalance_cost

    prices['MA_wealth_index_after_cost'] = 1.0
    for i in range(1, len(prices)):
        prices['MA_wealth_index_after_cost'].values[i] = prices['MA_index'].values[i] * prices['MA_wealth_index_after_cost'].values[i-1] - prices['Trade_cost'].values[i]

    trading_costs = prices['Trade_cost'].sum()
    Long_return = prices['Long_wealth_index'].values[len(prices)-1]
    MA_return = prices['MA_wealth_index_after_cost'].values[len(prices)-1]
    Performance = MA_return - Long_return

    return(Performance)

# Looping over function to generate results
# STRATEGY 1
MA_input = range(98, 103)
rebalance_freq_input = range(20, 25)

import seaborn as sns
import matplotlib.pyplot as plt

results = pd.DataFrame() #creates empty df
result_list = [] #creates empty list
for i in MA_input:
    for j in rebalance_freq_input:
        result = MA_strategy(i, 1, 25, j, 0.002, 0)
        result_list.append(result) # adds each result to the list
    result_series = pd.Series(result_list, name = i) #transforms the list into a series to be used in .concat
    results = pd.concat([results, result_series], axis = 1) #concat is faster than .insert
    # empties the list so it is ready for next inner loop
    result_list.clear()
rebalance_freq_input_list = list(rebalance_freq_input) #converts range to a list of integer values
rebalance_freq_input_series = pd.Series(rebalance_freq_input_list) #converts the list to a pd.series
results.set_index(rebalance_freq_input_series, inplace = True) #uses the pd.series to set the index, #inplace needed to modify existing df
print(results)
heatmap = sns.heatmap(results, cmap = "PiYG", center = 0) #cmap is coloring and center makes 0 neutral colour
plt.title("Performance heatmap of Moving Average strategies")
plt.xlabel("Moving Average")
plt.ylabel("Rebalancing frequency")
plt.show()

performance_mean = results.values.mean() # mean for entire df
performance_std = results.values.std(ddof=1) # std. for entire df
print(performance_mean, performance_std)

minvalue_series = results.min()
maxvalue_series = results.max()

# Rolling avg to choose top scenarios
Sum_of_rows = results.sum(axis = 1)
Sum_of_rows = Sum_of_rows.to_frame('Sum_rows') #converts the pandas series to pandas dtaframe
Sum_of_rows['Top 10'] = Sum_of_rows['Sum_rows'].rolling(10).mean()

Sum_of_columns = results.sum(axis = 0)
Sum_of_columns = Sum_of_columns.to_frame('Sum_columns') #converts the pandas series to pandas dtaframe
Sum_of_columns['Top 20'] = Sum_of_columns['Sum_columns'].rolling(10).mean()

# Gets best performing row and column
Sum_of_rows[['Sum_rows']].idxmax() #returns the index of the maximum value, the rebalancing frequency that performed the best
Sum_of_columns[['Sum_columns']].idxmax() #returns the index of the maximum value, the moving average frequency that performed best

# Looping over function to generate results
# STRATEGY 2
MA_input = range(1, 200)

results = pd.DataFrame() #creates empty df
result_list = [] #creates empty list
for i in MA_input:
    result = MA_strategy(i, 2, 25, 1, 0.00, -1)
    result_list.append(result) # adds each result to the list
result_series = pd.Series(result_list) #transforms the list into a series to be used in .concat
results = pd.concat([results, result_series], axis = 0)
print(results)
heatmap = sns.heatmap(results, cmap = "PiYG", center = 0) #cmap is coloring and center makes 0 neutral colour
plt.title("Performance heatmap of Moving Average strategies")
plt.show()
performance_mean = results.values.mean() # mean for entire df
performance_std = results.values.std(ddof=1) # std. for entire df
print(performance_mean, performance_std)


# Looping over function to generate results
# STRATEGY 3
MA_input = range(10, 200)
short_MA_input = range(5, 195)

import seaborn as sns
import matplotlib.pyplot as plt

results = pd.DataFrame() #creates empty df
result_list = [] #creates empty list
for i in MA_input:
    for j in short_MA_input:
        result = MA_strategy(i, 3, j, 10, 0.002, 0)
        result_list.append(result) # adds each result to the list
    result_series = pd.Series(result_list, name = i) #transforms the list into a series to be used in .concat
    results = pd.concat([results, result_series], axis = 1) #concat is faster than .insert
    # empties the list so it is ready for next inner loop
    result_list.clear()
short_MA_input_list = list(short_MA_input) #converts range to a list of integer values
short_MA_input_series = pd.Series(short_MA_input_list) #converts the list to a pd.series
results.set_index(short_MA_input_series, inplace = True) #uses the pd.series to set the index, #inplace needed to modify existing df
print(results)
heatmap = sns.heatmap(results, cmap = "PiYG", center = 0) #cmap is coloring and center makes 0 neutral colour
plt.title("Performance heatmap of Moving Average strategies")
plt.xlabel("Moving Average")
plt.ylabel("Shorter Period Moving Average")
plt.show()
performance_mean = results.values.mean() # mean for entire df
performance_std = results.values.std(ddof=1) # std. for entire df
print(performance_mean, performance_std)


# Looping over function to generate results
# STRATEGY 3 - LOWER / UPPER DIAGONAL
MA_input = range(10, 200)
short_MA_input = range(5, 195)

import statistics

result_list = [] #creates empty list
result_final_list = []
for i in MA_input:
    for j in short_MA_input:
        if i < j: # skips iterations where the MA_input is greater than the short_MA input. Change < and > in order to get upper/lower diagonal
            continue
        result = MA_strategy(i, 3, j, 10, 0.002, 0)
        result_list.append(result) # adds each result to the list
    result_final_list.extend(result_list) #adds the floating values of the list to the final list instead of making a list within list which .append would have done
    result_list.clear() # empties the list so it is ready for next inner loop
avg_performance = sum(result_final_list) / len(result_final_list)
stddv_performance = statistics.pstdev(result_final_list)
print(avg_performance, stddv_performance)


