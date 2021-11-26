# Dexter Dysthe
# Dr. Johannes
# B9337
# 22 November 2021

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import norm

plt.style.use('seaborn')

np.random.seed(37)

# ----------------------------------------------------- Q2 ----------------------------------------------------- #

# Using the closing price of the S&P 500 rounded to a whole number for SPX_0 and
# the yield on 1-month treasury bills for r, from Friday afternoon November 19th, 2021
SP_0 = 4697
rf_rate = 0.00110
expiry = 1/12

spx_options_data = pd.read_csv("spx_quotedata.csv")
market_prices = np.array(spx_options_data['Bid/Ask Avg'].values)
strikes = np.array(spx_options_data['Strike'].values)
IVs = np.array(spx_options_data['IV'].values)


# ----------------------------- Helper Functions ----------------------------- #

def d1_BS(S_t, K, sigma, r, time_till_mat):
    """
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the parameter d1 (sometimes denoted as d+) used in the Black-Scholes formula
    """

    d_1_BS = (np.log(S_t/K) + (r + 0.5 * (sigma**2)) * time_till_mat) / (sigma * np.sqrt(time_till_mat))

    return d_1_BS


def d2_BS(S_t, K, sigma, r, time_till_mat):
    """
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the parameter d2 (sometimes denoted as d-) used in the Black-Scholes formula
    """

    d_2_BS = d1_BS(S_t, K, sigma, r, time_till_mat) - sigma * np.sqrt(time_till_mat)

    return d_2_BS


def d1_RM(S_t, K, sigma, r, time_till_mat, mu_Q, lambda_Q, mu_z_Q, sigma_z_Q, k):
    """
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option
    :param mu_Q:          Mean of jump distribution
    :param lambda_Q:      Rate of jump arrival
    :param mu_z_Q:        Mean of jump distribution
    :param sigma_z_Q:     Standard deviation of jump distribution
    :param k:             Number of jumps witnessed

    :return: Returns the parameter d1_RM used in the option price formula in Merton's jump diffusion model
    """

    B = np.sqrt(k * (sigma_z_Q ** 2) + (sigma ** 2) * time_till_mat)
    d_1RM = d2_RM(S_t, K, sigma, r, time_till_mat, mu_Q, lambda_Q, mu_z_Q, sigma_z_Q, k) + B

    return d_1RM


def d2_RM(S_t, K, sigma, r, time_till_mat, mu_Q, lambda_Q, mu_z_Q, sigma_z_Q, k):
    """
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option
    :param mu_Q:          Mean of jump distribution
    :param lambda_Q:      Rate of jump arrival
    :param mu_z_Q:        Mean of jump distribution
    :param sigma_z_Q:     Standard deviation of jump distribution
    :param k:             Number of jumps witnessed

    :return: Returns the parameter d2_RM used in the option price formula in Merton's jump diffusion model
    """

    B = np.sqrt(k * (sigma_z_Q ** 2) + (sigma ** 2) * time_till_mat)
    d_2RM = (np.log(S_t / K) + (r - mu_Q * lambda_Q - 0.5 * (sigma ** 2) +
                                k * mu_z_Q / time_till_mat) * time_till_mat) / B

    return d_2RM


# ----------------------------- Pricing Functions ----------------------------- #

def euro_put_BS(S_t, K, sigma, r, time_till_mat):
    """
    European put option pricing function (without dividends)
    """

    d_1_BS = 1 / (sigma * np.sqrt(time_till_mat)) * (np.log(S_t / K) + (r + (sigma ** 2) / 2) * time_till_mat)
    d_2_BS = d_1_BS - sigma * np.sqrt(time_till_mat)

    euro_put_price_BS = K * np.exp(-r * time_till_mat) * ndtr(-d_2_BS) - S_t * ndtr(-d_1_BS)

    return euro_put_price_BS


def euro_put_RM(S_t, K, sigma, r, time_till_mat, lambda_Q, mu_z_Q, sigma_z_Q):
    """
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option
    :param lambda_Q:      Rate of jump arrival
    :param mu_z_Q:        Mean of jump distribution
    :param sigma_z_Q:     Standard deviation of jump distribution


    :return: Returns the price of a european put option under Merton's jump diffusion model.
             We truncate the infinite series at 7 terms.
    """

    mu_Q = np.exp(mu_z_Q + 0.5 * (sigma_z_Q ** 2)) - 1
    put_price_RM = euro_put_BS(S_t, K, sigma, r - mu_Q * lambda_Q, time_till_mat) / np.exp(mu_Q * lambda_Q * time_till_mat)

    for k in range(1, 6):
        B = np.sqrt(k * (sigma_z_Q ** 2) + (sigma ** 2) * time_till_mat)
        E = (-mu_Q * lambda_Q - (sigma ** 2) / 2 + (k * mu_z_Q) / time_till_mat) * time_till_mat
        poi_prob = ((lambda_Q * time_till_mat) ** k) / np.math.factorial(k)
        N_d2_RM = ndtr(-d2_RM(S_t, K, sigma, r, time_till_mat, mu_Q, lambda_Q, mu_z_Q, sigma_z_Q, k))
        N_d1_RM = ndtr(-d1_RM(S_t, K, sigma, r, time_till_mat, mu_Q, lambda_Q, mu_z_Q, sigma_z_Q, k))
        put_price_k_jumps = K * np.exp(-r * time_till_mat) * N_d2_RM - S_t * np.exp(0.5 * (B ** 2) + E) * N_d1_RM

        put_price_RM += (poi_prob * put_price_k_jumps)

    put_price_RM *= np.exp(-lambda_Q * time_till_mat)

    return put_price_RM


# ----------------------------- Greek Functions ----------------------------- #

def delta_BS(option_type, S_t, K, sigma, r, time_till_mat):
    """
    :param option_type:   C for Call or P for Put
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the corresponding delta for an option whose parameters match the input
             parameters to this function.
    """

    if option_type == 'C':
        return ndtr(d1_BS(S_t, K, sigma, r, time_till_mat))
    elif option_type == 'P':
        return ndtr(d1_BS(S_t, K, sigma, r, time_till_mat)) - 1
    else:
        print("Invalid type")


def gamma_BS(option_type, S_t, K, sigma, r, time_till_mat):
    """
    :param option_type:   C for Call or P for Put
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the corresponding gamma for an option whose parameters match the input
             parameters to this function.
    """

    if option_type == 'C' or option_type == 'P':
        return norm._pdf(d1_BS(S_t, K, sigma, r, time_till_mat)) * (1 / (sigma * S_t * np.sqrt(time_till_mat)))
    else:
        print("Invalid type")


def delta_RM(option_type, S_t, K, sigma, r, time_till_mat):
    """
    :param option_type:   C for Call or P for Put
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Time till maturity in years

    :return: Returns the corresponding Merton jump diffusion delta for an option whose
             parameter values match the input parameters to this function.
    """
    lambda_Q, mu_z_Q, sigma_z_Q = 1, -0.1, 0.1
    mu_Q = np.exp(mu_z_Q + 0.5 * (sigma_z_Q ** 2)) - 1
    delta_merton = delta_BS(option_type, S_t, K, sigma, r - mu_Q * lambda_Q, time_till_mat) / np.exp(mu_Q * lambda_Q * time_till_mat)

    for k in range(1, 101):
        B = np.sqrt(k * (sigma_z_Q ** 2) + (sigma ** 2) * time_till_mat)
        E = (-mu_Q * lambda_Q - (sigma ** 2) / 2 + (k * mu_z_Q) / time_till_mat) * time_till_mat
        poi_prob = ((lambda_Q * time_till_mat) ** k) / np.math.factorial(k)
        call_delta_k_jumps = np.exp(0.5 * (B ** 2) + E) * ndtr(d1_RM(S_t, K, sigma, r, time_till_mat, mu_Q, lambda_Q,
                                                                     mu_z_Q, sigma_z_Q, k))
        delta_merton += poi_prob * call_delta_k_jumps

    delta_merton *= np.exp(-lambda_Q * time_till_mat)
    if option_type == 'C':
        return delta_merton
    elif option_type == 'P':
        return delta_merton - 1
    else:
        print("Invalid type")


def gamma_RM(option_type, S_t, K, sigma, r, time_till_mat):
    """
    :param option_type:   C for Call or P for Put
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Time till maturity in years

    :return: Returns the corresponding Merton jump diffusion gamma for an option whose
             parameter values match the input parameters to this function.
    """
    lambda_Q, mu_z_Q, sigma_z_Q = 1, -0.1, 0.1
    mu_Q = np.exp(mu_z_Q + 0.5 * (sigma_z_Q ** 2)) - 1
    gamma_merton = gamma_BS(option_type, S_t, K, sigma, r - mu_Q * lambda_Q, time_till_mat) / np.exp(mu_Q * lambda_Q * time_till_mat)

    for k in range(1, 101):
        B = np.sqrt(k * (sigma_z_Q ** 2) + (sigma ** 2) * time_till_mat)
        E = (-mu_Q * lambda_Q - (sigma ** 2) / 2 + (k * mu_z_Q) / time_till_mat) * time_till_mat
        poi_prob = ((lambda_Q * time_till_mat) ** k) / np.math.factorial(k)
        gamma_k_jumps = np.exp(0.5 * (B ** 2) + E) * norm._pdf(d1_RM(S_t, K, sigma, r, time_till_mat, mu_Q, lambda_Q,
                                                                     mu_z_Q, sigma_z_Q, k)) / (B * S_t)
        gamma_merton += poi_prob * gamma_k_jumps

    gamma_merton *= np.exp(-lambda_Q * time_till_mat)
    if option_type == 'C' or option_type == 'P':
        return gamma_merton
    else:
        print("Invalid type")


# ------------------------ Calibration Functions ------------------------ #

def calibrate_put_RM(params):
    """

    :param params: A 4-tuple corresponding to the Merton jump diffusion parameters of interest

    :return:       Returns the sum, over market option prices, of the squared difference between
                   the market price and the Merton model price. That is, returns the square of the
                   Euclidean distance between the vector of Merton model prices and the vector of
                   observed market prices.
    """

    sigma, lambda_Q, mu_z_Q, sigma_z_Q = params
    sum_of_squared_differences = 0

    for index, price in enumerate(market_prices):
        model_price = euro_put_RM(SP_0, strikes[index], sigma, rf_rate, expiry, lambda_Q, mu_z_Q, sigma_z_Q)
        sum_of_squared_differences += (model_price - price) ** 2

    return sum_of_squared_differences


# ----------- Print Merton Jump Diffusion calibrated values ------------ #
params_0 = np.array([0.20, 1, -0.1, 0.1])
calibrated_params = minimize(calibrate_put_RM, params_0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

sigma_Q_marketdata = calibrated_params.x[0]
lambda_Q_marketdata = calibrated_params.x[1]
mu_z_Q_marketdata = calibrated_params.x[2]
sigma_z_Q_marketdata = np.abs(calibrated_params.x[3])

np.set_printoptions(suppress=True)
print("Calibrated SP500 volatility: ", sigma_Q_marketdata)
print("Calibrated jump frequency: ", lambda_Q_marketdata)
print("Calibrated mean jump size: ", mu_z_Q_marketdata)
print("Calibrated jump size volatility: ", sigma_z_Q_marketdata)

# Storing merton model prices to be used for calculating Black-Scholes implied vols
merton_prices = np.vectorize(euro_put_RM)(SP_0, strikes, sigma_Q_marketdata, rf_rate, expiry, lambda_Q_marketdata,
                                          mu_z_Q_marketdata, sigma_z_Q_marketdata)


def implied_vol(merton_price, S_t, K, r, time_till_mat):
    """
    Bisection method for finding the implied volatility of SPX put
    options. Newton-Raphson produces highly unstable results.

    :param merton_price:  Put option Merton jump diffusion prices using calibrated parameters
    :param S_t:           Current spot price
    :param K:             Strike
    :param r:             Interest rate
    :param time_till_mat: Time till maturity in years

    :return: Returns the corresponding implied volatility
    """

    max_iterations = 200
    right_endpoint = 1 + calibrated_params.x[0]
    left_endpoint = 0

    for i in range(max_iterations):
        sigma_0 = (left_endpoint + right_endpoint) / 2
        put_price = euro_put_BS(S_t, K, sigma_0, r, time_till_mat)
        error = merton_price - put_price

        if abs(error) < 0.00001:
            return sigma_0
        # If error < 0, then BS put price is too large and thus need to decrease
        # sigma. Set right endpoint to sigma_0 for next iteration.
        elif error < 0:
            right_endpoint = sigma_0
        # If error > 0, then BS put price is too small and thus need to increase
        # sigma. Set left endpoint to sigma_0 for next iteration.
        else:
            left_endpoint = sigma_0

    # The returned value sigma_0 is equal to np.sqrt(sigma_0 ** 2 + (k / time_till_mat) * (sigma_z_Q_marketdata ** 2))
    # where k denotes the number of terms kept from the infinite series.
    return sigma_0


# Using the calibrated Merton model option prices, we calculate the Black-Scholes implied volatilities for a
# range of strikes.
implied_vol_vec = np.abs(np.vectorize(implied_vol)(merton_prices, SP_0, strikes, rf_rate, expiry))

plt.plot(strikes, implied_vol_vec, label='Black-Scholes using calibrated Merton model prices')
plt.plot(strikes, IVs, label='Market')
plt.title("Implied Volatility Smile for 12/17/2021 Delivery SPX Put Options")
plt.xlabel("Strike Price (S&P 500_0 = 4697)")
plt.ylabel("Implied Volatility")
plt.legend()
plt.show()


# ------------------------------------- Plotting Functions ------------------------------------- #

def plot_greek(greek_func, option_type, sigma):
    """

    :param greek_func:  One of the greek functions defined above, e.g. delta_BS, delta_RM, gamma_BS, gamma_RM
    :param option_type: C for Call or P for Put
    :param sigma:       Volatility, included since our HW asks us to consider 2 values for sigma

    :return: Does not return anything, purpose of function is to create plots
    """

    initial_spot = np.linspace(40, 160, 240)

    for K in [80, 90, 100, 110, 120]:
        greek_vec = np.vectorize(greek_func)(option_type, initial_spot, K, sigma, 0.02, 30 / 365)
        plt.plot(initial_spot, greek_vec, label="K = {}".format(K))

    if option_type == 'C':
        plt.title("Call Option {} (sigma = {})".format(greek_func.__name__, sigma))
    elif option_type == 'P':
        plt.title("Put Option {} (sigma = {})".format(greek_func.__name__, sigma))

    plt.xlabel("Initial Stock Price S_t")
    plt.ylabel("{}".format(greek_func.__name__))
    plt.legend(loc='upper left')
    plt.show()


def plot_greeks(sigma):
    """

    :param sigma: volatility

    :return: Does not return anything, purpose of function is to create plots
    """

    # For each of the greek functions defined above, e.g. delta, gamma, vega, rho, and theta, we
    # apply the plot_greek function -- implemented immediately above this function -- which
    # will plot each of the greeks as functions of the current spot price S_t and for 5 different
    # strike prices.
    for greek_func in [delta_BS, delta_RM, gamma_BS, gamma_RM]:
        plot_greek(greek_func, 'C', sigma)
        plot_greek(greek_func, 'P', sigma)


plot_greeks(0.15)
plot_greeks(0.25)


# ----------------------------------------------------- Q3 ----------------------------------------------------- #

# Test values for comparing the exact black scholes formula with the Gil-Pelaez method
S_Q3 = 100
K_Q3 = 100
r_Q3 = 0.02
sigma_Q3 = 0.20
T_Q3 = 1/12


def composite_simpson(func, a, b, num_points):
    """
    This function applies the composite simpson's rule to the function func over the
    interval [a, b] with num_points-many points of evaluation.
    """
    h = (b - a) / num_points
    v = np.array([a + index * h for index in range(num_points + 1)])

    approx_integral = 0

    for i in range(1, num_points + 1):
        approx_integral += (func(v[i-1]) + 4 * func(np.mean([v[i], v[i-1]])) + func(v[i]))

    approx_integral *= (h / 6)

    return approx_integral


def phi_t(v):
    return np.exp(1j * v * (np.log(S_Q3) + (r_Q3 - 0.5 * (sigma_Q3 ** 2)) * T_Q3) - 0.50 * ((sigma_Q3 * v) ** 2) * T_Q3)


def pi_1_integrand(v):
    phi_T_v_minus_i = phi_t(v - 1j)
    phi_t_minus_i = phi_t(-1j)
    numerator = phi_T_v_minus_i * np.exp(-1j * v * np.log(K_Q3))
    denominator = phi_t_minus_i * (1j * v)

    return np.real(numerator / denominator)


def pi_2_integrand(v):
    phi_t_v = phi_t(v)

    return np.real(phi_t_v * np.exp(-1j * v * np.log(K_Q3)) / (1j * v))


def gil_pelaez_BS(a, b, num_of_points):
    pi_1 = 0.50 + composite_simpson(pi_1_integrand, a, b, num_of_points) / np.pi
    pi_2 = 0.50 + composite_simpson(pi_2_integrand, a, b, num_of_points) / np.pi

    call_price = S_Q3 * pi_1 - K_Q3 * np.exp(-r_Q3 * T_Q3) * pi_2

    return call_price


d_1 = 1 / (sigma_Q3 * np.sqrt(T_Q3)) * (np.log(S_Q3 / K_Q3) + (r_Q3 + (sigma_Q3 ** 2) / 2) * T_Q3)
d_2 = d_1 - sigma_Q3 * np.sqrt(T_Q3)

euro_call_price = S_Q3 * ndtr(d_1) - K_Q3 * np.exp(-r_Q3 * T_Q3) * ndtr(d_2)
print("Exact BS Price: {}".format(euro_call_price))
print("Gil Peleaz Price: {}".format(gil_pelaez_BS(0.000001, 100, 1250)))


# ---------------------------- Unused Functions (may be useful in the future) ---------------------------- #

# def vega_BS(option_type, S_t, K, sigma, r, time_till_mat):
#    """
#    :param option_type:   C for Call or P for Put
#    :param S_t:           Current spot price
#    :param K:             Strike price
#    :param sigma:         Volatility
#    :param r:             Current short rate
#    :param time_till_mat: Time till maturity in years
#
#    :return: Returns the corresponding vega for an option whose parameter values
#             match the input parameters to this function.
#    """
#
#    d_1 = 1 / (sigma * np.sqrt(time_till_mat)) * (np.log(S_t / K) + (r + (sigma ** 2) / 2) * time_till_mat)
#
#    if option_type == 'C' or option_type == 'P':
#        return norm._pdf(d_1) * np.sqrt(time_till_mat) * S_t
#    else:
#        print("Invalid type")
#


# def vega_RM(option_type, S_t, K, sigma, r, time_till_mat, lambda_Q, mu_z_Q, sigma_z_Q):
#    """
#    :param option_type:   C for Call or P for Put
#    :param S_t:           Current spot price
#    :param K:             Strike price
#    :param sigma:         Volatility
#    :param r:             Current short rate
#    :param time_till_mat: Time till maturity in years
#    :param lambda_Q:      Rate of jump arrival
#    :param mu_z_Q:        Mean of jump distribution
#    :param sigma_z_Q:     Standard deviation of jump distribution
#
#    :return: Returns the corresponding Merton jump diffusion vega for an option whose
#             parameter values match the input parameters to this function.
#    """
#    mu_Q = np.exp(mu_z_Q + 0.5 * (sigma_z_Q ** 2)) - 1
#    put_vega_RM = vega_BS(option_type, S_t, K, sigma, r, time_till_mat)
#
#    for k in range(1, 6):
#        poi_prob = ((lambda_Q * time_till_mat) ** k) / np.math.factorial(k)
#        d2_merton = -d2_RM(S_t, K, sigma, r, time_till_mat, mu_Q, lambda_Q, mu_z_Q, sigma_z_Q, k)
#        put_vega_k_jumps = (K * np.exp(-r * time_till_mat) * norm.pdf(d2_merton) * sigma * time_till_mat) / (np.sqrt(k * (sigma_z_Q ** 2) + (sigma ** 2) * time_till_mat))
#        put_vega_RM += (poi_prob * put_vega_k_jumps)
#
#    put_vega_RM *= np.exp(-lambda_Q * time_till_mat)
#    if option_type == 'C' or option_type == 'P':
#        return put_vega_RM
#    else:
#        print("Invalid type")
#