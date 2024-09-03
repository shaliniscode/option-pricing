import math
import numpy as np

def std_dev(price_data, window=20,trading_periods = 252):
    log_return = (price_data['Close']/price_data['Close'].shift(1)).apply(np.log)
    result = log_return.rolling(window=window, center = False).std()*math.sqrt(trading_periods)
    return result.dropna()

    
def parkison_vol(price_data, window=20,trading_periods = 252):
    """Parkison volatility uses the 
        stock price high and low price of the day"""
    rs = (1.0 / (4.0 * math.log(2.0))) * (( price_data['High'] /
                                          price_data['Low']).apply(np.log))** 2.0
    
    def f(v):
        return (trading_periods * v.mean()) ** 0.5
    
    result = rs.rolling(window = window, center = False).apply(func = f)
    
    return result.dropna()
  
    
def garman_klass(price_data, window=20,trading_periods = 252):
    """ Garman_klass uses opeing and closing price along with high and low price """
    log_hl = (price_data['High'] / price_data['Low']).apply(np.log)
    log_co = (price_data["Close"] / price_data['Open']).apply(np.log)

    rs = 0.5 *log_hl**2 - (2* math.log(2)-1) *log_co**2

    def f(v):
        return (trading_periods * v.mean()) ** 0.5
    result = rs.rolling(window = window, center = False).apply(func = f)
    
    return result.dropna()

def hodges_tompkins(price_data, window=20,trading_periods = 252):
    """bias correction for estimation"""
    log_return = (price_data['Close']/price_data['Close'].shift(1)).apply(np.log)

    vol = log_return.rolling(window = window, center = False).std() *math.sqrt(trading_periods)

    h = window
    n = (log_return.count() -h) +1

    adj_factor = 1.0 / (1.0 -(h/n) + ((h **2 -1) / (3 * n**2)))

    result = vol * adj_factor

    return result

def rogers_satchell(price_data, window = 20, trading_periods = 252):
    """ incorporates drift term"""
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low']/ price_data['Open']).apply(np.log)
    log_co = (price_data["Close"] / price_data['Open']).apply(np.log)

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    def f(v):
        return (trading_periods * v.mean()) ** 0.5
    return rs.rolling(window = window, center = False).apply(func = f)

def yang_zhang(price_data, window=20,trading_periods=252):
    """ combination of the overnight (close-to-open volatility)"""

    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low']/ price_data['Open']).apply(np.log)
    log_co = (price_data["Close"] / price_data['Open']).apply(np.log) 

    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc ** 2

    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc ** 2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(window = window, center = False).sum() * (1.0 / (window-1.0))

    open_vol = log_oc_sq.rolling(window = window, center = False).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window = window, center = False).sum() * (1.0 / (window - 1.0))
    k = 0.34 / (1.34 + (window + 1)/(window - 1))
    result = (open_vol + k * close_vol + (1- k)* window_rs).apply(np.sqrt)
    return result

    

