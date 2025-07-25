import numpy as np
from scipy.optimize import curve_fit

def fit_parabola(x, y):
    coeffs = np.polyfit(x, y, 2)
    y_fit = np.polyval(coeffs, x)

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return coeffs, r2, y_fit

def fit_parabola_curvfit(x: np.ndarray, y: np.ndarray):
    def parabola(x, a, b, c):
        return a * x**2 + b * x + c
    popt, _ = curve_fit(parabola, x, y)
    y_fit = parabola(x, *popt)
    r2 = 1 - np.sum((y - y_fit)**2) / np.sum((y - np.mean(y))**2)
    return popt, r2, y_fit