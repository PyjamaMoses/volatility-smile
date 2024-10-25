import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
from typing import List, Union
from scipy.optimize import minimize

class BouchardSornettOptionPricing:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize Bouchard-Sornett Option Pricing model.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing 'close' prices.
        """
        self.prices = df["close"].astype(float)
        self.n = len(self.prices)
        self.epsilon = 1e-10  # Small constant to prevent division by zero
        self.returnrate = None
        self.detrended = None
        self.pdf_series = None
        self.sigma = None
        self.step = None

    def calculate_returnrate(self, interval: int = 1) -> pd.Series:
        """
        Calculate return rate over a specified interval.
        
        Parameters:
            interval (int): Time interval for calculating return rate (default: 1).
        
        Returns:
            pd.Series: Return rate series.
        """
        self.returnrate = (self.prices.shift(-interval) - self.prices) / (
            self.prices + self.epsilon
        )
        return self.returnrate

    def detrended_returnrate(self, interval: int = 1) -> pd.Series:
        """
        Calculate detrended return rate series.
        
        Parameters:
            interval (int): Time interval for calculating return rate (default: 1).
        
        Returns:
            pd.Series: Detrended return rate series.
        """
        returnrate = self.calculate_returnrate(interval)
        self.detrended = returnrate - returnrate.mean()
        return self.detrended

    def calculate_pdf(self, ifplot: bool = False, nbin: int = None, interval: int = 1) -> pd.Series:
        """
        Calculate PDF based on detrended return rates.

        Parameters:
            ifplot (bool): If True, plot the PDF (default: False).
            nbin (int): Number of bins for PDF (default: None, calculated as sqrt(n)).
            interval (int): Interval for detrended return calculation (default: 1).

        Returns:
            pd.Series: PDF series.
        """
        # Calculate detrended return rates
        self.detrended_returnrate(interval)
        
        # Define bin parameters
        Rmax, Rmin = self.detrended.max(), self.detrended.min()
        nbin = nbin or int(round(np.sqrt(self.n)))
        self.step = (Rmax - Rmin) / nbin
        
        # Calculate histogram with `np.histogram` using bin edges for consistency
        counts, bin_edges = np.histogram(self.detrended, bins=nbin, range=(Rmin, Rmax))
        
        # Calculate bin centers and create normalized PDF series
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        count_series = pd.Series(counts, index=bin_centers)
        self.pdf_series = count_series / self.n  # Normalize by the total number of data points

        # Plot the PDF if requested
        if ifplot:
            plt.figure(figsize=(10, 6))
            plt.bar(count_series.index, self.pdf_series.values, width=self.step, align="center", alpha=0.6, color="b")
            plt.title("Probability Density Function (PDF)")
            plt.xlabel("Detrended Return Rate")
            plt.ylabel("Probability")
            plt.grid(True)
            plt.show()

        return self.pdf_series

    def price_option(self, S0: float, X_range: List[float], T: int = 21, r: float = 0.00, nbin: int = None) -> List[float]:
        """
        Calculate option prices using the Bouchard-Sornett model.
        
        Parameters:
            S0 (float): Current asset price.
            X_range (List[float]): Range of strike prices.
            T (int): Time to maturity in trading days (default: 21).
            r (float): Risk-free rate (default: 0.00).
            nbin (int): Number of bins for PDF (default: None).
        
        Returns:
            List[float]: List of option prices.
        """
        if self.pdf_series is None:
            self.calculate_pdf(nbin=nbin, interval=T)

        option_prices = []
        for X in X_range:
            payoff_sum = sum(max(S0 * (1 + k) - X, 0) * prob for k, prob in self.pdf_series.items())
            V_0 = np.exp(-r * T / 252) * payoff_sum  # Discounted expected payoff
            option_prices.append(V_0)

        return option_prices

    def black_scholes(self, S: float, X: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """
        Black-Scholes option pricing formula.
        
        Parameters:
            S (float): Current stock price.
            X (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate.
            sigma (float): Volatility.
            option_type (str): Option type ('call' or 'put').
        
        Returns:
            float: Option price.
        """
        d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * si.norm.cdf(d1) - X * np.exp(-r * T) * si.norm.cdf(d2)
        elif option_type == "put":
            return X * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

    def implied_volatility(
        self, S0: float, X_range: Union[float, List[float]], T: int, r: float, option_type: str = "call", tol: float = 1e-3
    ) -> List[float]:
        """
        Calculate implied volatility by minimizing error between observed and model prices.
        
        Parameters:
            S0 (float): Current stock price.
            X_range (Union[float, List[float]]): Single strike price or list of strike prices.
            T (int): Time to maturity.
            r (float): Risk-free rate.
            option_type (str): Option type ('call' or 'put').
            tol (float): Tolerance for optimization (default: 1e-3).
        
        Returns:
            List[float]: List of implied volatilities.
        """
        X_range = [X_range] if isinstance(X_range, float) else X_range
        option_prices = self.price_option(S0, X_range, T, r)
        implied_vols = []

        for observed_price, strike in zip(option_prices, X_range):
            objective = lambda sigma: (self.black_scholes(S0, strike, T, r, sigma[0], option_type) - observed_price) ** 2
            result = minimize(objective, [0.2], bounds=[(1e-10, 3.0)], tol=tol)
            implied_vols.append(result.x[0])

        return implied_vols

    def Delta(self, S: float, X: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """
        Calculate Delta for an option.
        
        Parameters:
            S (float): Current stock price.
            X (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free rate.
            sigma (float): Volatility.
            option_type (str): Option type ('call' or 'put').
        
        Returns:
            float: Delta value.
        """
        d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return si.norm.cdf(d1) if option_type == "call" else si.norm.cdf(d1) - 1

    def Sigma(self, T: int) -> float:
        """
        Calculate the standard deviation (volatility) of the detrended return rate.
        
        Parameters:
            T (int): Time interval for calculating volatility.
        
        Returns:
            float: Volatility.
        """
        if self.detrended is None:
            self.detrended_returnrate(interval=T)
        self.sigma = self.detrended.std()
        return self.sigma

    def hedging_strategy(self, xt_range: List[float], X: float, T: int, t: int, if_surrogate: bool = False) -> List[float]:
        """
        Calculate hedging strategy positions.
        
        Parameters:
            xt_range (List[float]): Range of current positions.
            X (float): Strike price.
            T (int): Time to maturity.
            t (int): Current time.
            if_surrogate (bool): Whether to apply a surrogate model (default: False).
        
        Returns:
            List[float]: List of hedging strategy positions.
        """
        if self.sigma is None:
            self.Sigma(T)
        if self.pdf_series is None:
            self.calculate_pdf()
        if if_surrogate==False:
            self.positions = []
            for x_t in xt_range:
                position = sum(
                    (x_T**2 * self.step**2 / 3 - x_t * self.step / 2 * (x_t + X - 2 * x_T) + (x_T - X) * (x_T - x_t)) * prob
                    for x_T, prob in self.pdf_series.items()
                ) / (self.sigma**2 * (T - t))
                self.positions.append(position)
        else: 
            self.positions = []
        
        return self.positions
    
            


