import numpy as np
from scipy.stats import norm


# d1 is an intermediate term in the Black-Scholes formula
def d1(S, K, T, r, sigma):
    """
    S: spot price
    K: strike price
    T: time to maturity
    r: risk-free rate
    sigma: volatility
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


# d2 is derived from d1
def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    """
    Computes the price of a European call option
    using the Black-Scholes formula
    """
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)

    # norm.cdf = standard normal cumulative distribution function
    return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def put_price(S, K, T, r, sigma):
    """
    Computes the price of a European put option
    """
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)

    return K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)