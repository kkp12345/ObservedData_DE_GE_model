import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ============================
# Generalized Analytical Model
# ============================

def calculate_voltage(t_th, R, C, Vcc):
    """
    Calculate Vin using the analytical formula.
    :param t_th: Time in seconds (array or scalar)
    :param R: Resistance in ohms
    :param C: Capacitance in farads
    :param Vcc: Supply voltage
    :return: Calculated Vin
    """
    return (0.5 * Vcc) / (1 - np.exp(-t_th / (R * C)))

# ============================
# Double Exponential Decay Fit
# ============================

# Define the double exponential decay model
def double_exponential_decay(x, a1, b1, a2, b2, c):
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x) + c

def fit_exponential_decay(time_us, voltage):
    """
    Fit the double exponential decay model and return fitted parameters.
    :param time_us: Observed time in microseconds
    :param voltage: Observed battery voltage
    :return: Fitted parameters and time for plotting the curve
    """
    # Scale down time to seconds
    time_us_scaled = time_us / 1e6

    # Initial guesses and bounds for curve fitting
    initial_guesses = [5, -1, 1, -0.1, 2]
    lower_bounds = [0, -50, 0, -50, 0]
    upper_bounds = [10, 0, 10, 0, 5]

    # Fit the model
    params, _ = curve_fit(double_exponential_decay, time_us_scaled, voltage,
                          p0=initial_guesses, bounds=(lower_bounds, upper_bounds))
    return params, time_us_scaled

# ============================
# Main Plot for Comparison
# ============================

# Data for fitting the exponential model
# voltage = np.array([4.2,4.00, 3.75, 3.50, 3.25, 3.00, 2.75, 2.50, 2.25, 2.00])
# time_us = np.array([51531,55227,60412,67549,75446,86913,101172, 129071,164059,250736])
# voltage = np.array([4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 
#                     3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3,
#                     2.2, 2.1, 2.0,])


# time_us = np.array([51531, 53233, 55227, 57352, 59247, 61769, 64713, 67549, 71088, 73784,
#                     77192, 82520, 86913, 95022, 98122, 105056,114508, 129071,136475,159050, 
#                     175686, 209195, 250736])
voltage = np.array([4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 
                    3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3,
                    2.2, 2.1, 2.0,])


time_us = np.array([54921, 56891, 58954, 61145, 63623, 66093, 68810, 72029, 75334, 79326,
                    83708, 89706, 95890, 102632, 108205, 116660, 125017, 137652, 158169, 178971,
                    212281, 264004,295948]) 
# voltage = np.array([4.2, 4.15, 4.1, 4.05, 4, 3.95, 3.9, 3.85, 3.8, 
#                     3.75, 3.7, 3.65,3.6, 3.55, 3.5, 3.45, 3.4, 3.35, 
#                     3.3, 3.25, 3.2, 3.15, 3.1, 3.05, 3, 2.95, 2.9,
#                     2.85, 2.8, 2.75, 2.7, 2.65, 2.6, 2.55, 2.5, 2.45, 
#                     2.4, 2.35, 2.3, 2.25, 2.2, 2.15, 2.1, 2.05, 2
                   
# ])
# time_us = np.array([51531, 52339, 53233, 54204, 55227, 56174, 57352, 58342, 59247,
#                     60412, 61769, 63139, 64713, 66460, 67549, 69935, 71088, 71980, 
#                     73784, 75446, 77192, 80213, 82520, 84627, 86913, 90281, 95022, 
#                     96781, 98122, 101172, 105056, 109737, 114508, 118149, 129071, 
#                     129415, 136475, 143675, 159050, 164059, 175686, 189318, 209195,
#                     226988, 250736])

# Parameters for the generalized model
parameters = [
        # (85000, 0.000001, 4.00)  # Example set 1: R=100kΩ, C=1µF, Vcc=3.3V,0.56%[1]0.59%
        (84000, 0.000001, 4.03)# Example set 1: R=100kΩ, C=1µF, Vcc=3.3V,0.56%[1]
        # (91000, 0.000001, 3.67)  # Example set 1: R=100kΩ, C=1µF, Vcc=3.3V,0.56%[1]
    #   (92400, 0.000001, 3.75),  # Best so far 1.33% error[2]
    #   (91000, 0.000001, 3.67),  # Best so far 1.14% error new dataset [3]
]

# Time range for the analytical model
# t_th_range = (0.05, 0.25)
t_th_range = (0.25, 0.05)

# Create a single plot for comparison
plt.figure(figsize=(12, 8))
params, time_us_scaled = fit_exponential_decay(time_us, voltage)
time_fit_scaled = np.linspace(min(time_us_scaled), max(time_us_scaled), 1000)
voltage_fit = double_exponential_decay(time_fit_scaled, *params)
# Generalized Analytical Model
t_th = np.linspace(t_th_range[0], t_th_range[1], 500)  # Generate time values
for R, C, Vcc in parameters:
    V_in = calculate_voltage(t_th, R, C, Vcc)  # Calculate voltage
    plt.scatter(time_us_scaled, voltage, color='blue', label="Observed Data(Interpolation)")
    plt.plot(time_fit_scaled, voltage_fit, color='red', label="Double exponential Decay Fit")
    plt.plot(t_th, V_in, linestyle='--', label=f'Generalized: R={R}Ω, C={C}F, Vcc={Vcc}V')
    
   
# Double Exponential Decay Fit


# Plot observed data and fitted curve
# plt.scatter(time_us_scaled, voltage, color='blue', label="Observed Data(Interpolation)")
# plt.plot(time_fit_scaled, voltage_fit, color='red', label="Double exponential Decay Fit")

# Calculate MAPE and R^2 for the double exponential decay model
voltage_predicted_decay = double_exponential_decay(time_us_scaled, *params)
mape_decay = np.mean(np.abs((voltage - voltage_predicted_decay) / voltage)) * 100
ss_total_decay = np.sum((voltage - np.mean(voltage))**2)
ss_residual_decay = np.sum((voltage - voltage_predicted_decay)**2)
r_squared_decay = 1 - (ss_residual_decay / ss_total_decay)

# Calculate MAPE and R^2 for the generalized analytical model
V_in_generalized = calculate_voltage(time_us_scaled, *parameters[0])
mape_generalized = np.mean(np.abs((voltage - V_in_generalized) / voltage)) * 100
ss_total_generalized = np.sum((voltage - np.mean(voltage))**2)
ss_residual_generalized = np.sum((voltage - V_in_generalized)**2)
r_squared_generalized = 1 - (ss_residual_generalized / ss_total_generalized)



# Plot settings
plt.title(f"Comparison of Models\nDouble Exp.: MAPE={mape_decay:.2f}%, R²={r_squared_decay:.4f} | Generalized: MAPE={mape_generalized:.2f}%, R²={r_squared_generalized:.4f}")
plt.xlabel("Time (s)",fontsize=16)
plt.ylabel("Voltage (V)",fontsize=16)
plt.grid(True)
plt.legend()
plt.show()

# Print the fitted parameters for the double exponential decay
a1, b1, a2, b2, c = params
print(f"Fitted Parameters:\na1 = {a1:.4f}\nb1 = {b1:.4f}\na2 = {a2:.4f}\nb2 = {b2:.4f}\nc = {c:.4f}")

# Print MAPE and R^2 values
print(f"Double Exponential Decay Model:\nMAPE = {mape_decay:.2f}%\nR² = {r_squared_decay:.4f}")
print(f"Generalized Analytical Model:\nMAPE = {mape_generalized:.2f}%\nR² = {r_squared_generalized:.4f}")
