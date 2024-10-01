import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm




# BINOMIAL MODEL

def binomial_model(S0, K, T, r, u, d, N, option_type="call", tree="yes"):

    '''
    Parameters:
    S0          : initial stock price at time t=0
    K           : strike price of the option
    T           : time to maturity (in years)
    r           : risk-free interest rate (annualized)
    u           : upward movement factor (stock price increase multiplier)
    d           : downward movement factor (stock price decrease multiplier)
    N           : number of time steps in the binomial tree
    option_type : "call" or "put", default is "call"
    tree        : "yes" or "no", default is "yes"
    '''
    
    # dt = Time increment per step in binomial tree
    dt = T / N  

    # R = Risk-free rate per step
    R = (1 + r * dt)  

    # p = Risk-neutral probability of an up move
    p = (R - d) / (u - d)
   
    # q = Risk-neutral probability of a down move
    q = 1 - p   

    # Initialising the stock prices at maturity in a binomial tree
    stock_tree = [[S0]]
    for i in range(1, N + 1):
        level_prices = [S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        stock_tree.append(level_prices)

    # Initialising the option values at maturity for either call or put
    if option_type == "call":
        option_values = [max(0, stock_price - K) for stock_price in stock_tree[-1]]
    elif option_type == "put":
        option_values = [max(0, K - stock_price) for stock_price in stock_tree[-1]]
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    # Calculating the option price at earlier nodes
    for i in range(N - 1, -1, -1):
        option_values = [
            (p * option_values[j + 1] + q * option_values[j]) / R
            for j in range(i + 1)
        ]

    # If tree is set to "yes", visualise the binomial tree
    if tree == "yes":
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Binomial Tree for Stock Prices')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')

        ax.set_yticks([])

        # Plotting the nodes and lines connecting them
        for i in range(len(stock_tree)):
            for j in range(len(stock_tree[i])):
                # Add text slightly above each node to avoid overlapping with lines
                ax.text(i, stock_tree[i][j] * 1.015, f'{stock_tree[i][j]:.0f}', ha='center', va='bottom')
                if i < len(stock_tree) - 1:
                    # Plotting the lines in black connecting nodes
                    ax.plot([i, i + 1], [stock_tree[i][j], stock_tree[i + 1][j]], 'k-')
                    ax.plot([i, i + 1], [stock_tree[i][j], stock_tree[i + 1][j + 1]], 'k-')

        # Setting the x axis to Integers only
        ax.set_xticks(range(N + 1))
        plt.show()

    # Returning the formatted option price
    if option_type == "call":
        print(f"Binomial Call Option Price: {option_values[0]:.2f}")
    elif option_type == "put":
        print(f"Binomial Put Option Price: {option_values[0]:.2f}")





# MONTE CARLO MODEL

def monte_carlo_model(Np, Nt, T, r, sigma, K, S0, option_type="call", graph="no"):
    """    
    Parameters:
    Np         : number of paths (simulations) for the Monte Carlo simulation
    Nt         : number of time steps in each path
    T          : time to maturity (in years)
    r          : risk-free interest rate (annualized)
    sigma      : volatility of the stock 
    K          : strike price of the option
    S0         : initial stock price
    option_type: "call" or "put", default is "call"
    graph      : "yes" or "no", default is "yes"
    """
   
    # dt = Time step
    dt = T / Nt  # time step

    # Generating the random numbers for the price paths
    Z = np.random.normal(0, 1, (Np, Nt))
    dB = np.sqrt(dt) * Z
    
    # Initialising the stock price matrix
    S = np.zeros((Np, Nt+1))
    S[:, 0] = S0  # initial stock price
    
    # Loop to simulate the stock price paths
    for i in range(1, Nt+1):
        S[:, i] = S[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dB[:, i-1])

    # Stock price at maturity
    ST = S[:, -1]  

    # Call option price calculation
    if option_type == "call":
        payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.exp(-r * T) * np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    # Calcualting mean option price
    option_price = np.mean(payoffs)

    # Standard deviation (error calculation)
    error = np.std(payoffs) / np.sqrt(Np)
    
    # Plotting all the paths if graph is enabled
    if graph == "yes":
        plt.figure(figsize=(10,6))
         # For loop to plot all paths
        for i in range(Np):  
            plt.plot(np.linspace(0, T, Nt+1), S[i, :], lw=0.75, alpha=0.75)

        # Adding labels and title
        plt.title(f'Monte Carlo Simulation of {Np} Stock Price Paths')
        plt.xlabel('Time (Years)')
        plt.ylabel('Stock Price')
        plt.xticks(np.arange(0, T+1, 1))  # Set the x-axis to show only integer values

        # Show the plot
        plt.show()

    # Print results
    print(f"Monte Carlo {option_type.capitalize()} Option Price:", round(option_price, 2), "with error:", round(error, 2))





# BLACK SCHOLES MODEL

def black_scholes(S0, K, T, r, sigma, option_type="call", graph="no"):
    """
    S0: Current stock price
    K: Strike price of the option
    T: Time to maturity (in years)
    r: Risk-free interest rate (annualized)
    sigma: Volatility of the underlying asset (annualized)
    option_type: "call" for call option, "put" for put option
    graph: "yes" to generate a graph of the option prices vs. stock prices
    """
    
    # Calculating d1, the probability that the option will expire 'in the money'
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
   
    # Calculating d2, the discounted factor for the strike price, K
    d2 = d1 - sigma * np.sqrt(T)
    
    # If graph is set to "yes", generate a graph of option prices for a range of stock prices
    if graph == "yes":
        stock_prices = np.linspace(1e-3, 2*S0, 100)  # Start from a small positive value (e.g., 0.001) to avoid log(0)
        option_prices = []

        for S_val in stock_prices:
            d1 = (np.log(S_val / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                price = (S_val * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
            elif option_type == "put":
                price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S_val * norm.cdf(-d1))
            option_prices.append(price)

        plt.plot(stock_prices, option_prices)
        plt.title(f"{option_type.capitalize()} option with strike ${K}")
        plt.xlabel("Underlying price")
        plt.ylabel("Option price")
        plt.show()

    # Computing the value of the option whether it is a 'call' or 'put' option
    if option_type == "call":
        option_price = (S0 * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        print(f"Black Scholes Call Option Price: {option_price:.2f}")
    elif option_type == "put":
        # Black-Scholes formula for a put option
        option_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S0 * norm.cdf(-d1))
        print(f"Black Scholes Put Option Price: {option_price:.2f}")
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'")




# HESTON MODEL

def heston_option_price(S0, V0, r, k, theta, sigma, rho, T, K, n_steps=1000, n_paths=10000):
    """
    Calculates the price of a European call option using the Heston model.

    Parameters:
    S0 : float - Initial stock price
    V0 : float - Initial variance (volatility squared)
    r : float - Risk-free rate
    k : float - Speed of mean reversion of variance
    theta : float - Long-term variance
    sigma : float - Volatility of volatility
    rho : float - Correlation between the asset price and variance
    T : float - Time to maturity
    K : float - Strike price
    n_steps : int - Number of time steps (default: 1000)
    n_paths : int - Number of simulated paths (default: 10000)

    Returns:
    call_price : float - Estimated European call option price
    """
    dt = T / n_steps  # Time step size
    
    # Random numbers for the two Wiener processes
    dW1 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    dW2 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

    # Correlate dW1 and dW2 using the correlation coefficient rho
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2

    # Arrays to hold simulated asset prices and variances
    S = np.zeros((n_paths, n_steps + 1))
    V = np.zeros((n_paths, n_steps + 1))

    # Initial values
    S[:, 0] = S0
    V[:, 0] = V0

    # Simulating the paths using the Heston model
    for t in range(1, n_steps + 1):
        V[:, t] = np.maximum(V[:, t - 1] + k * (theta - V[:, t - 1]) * dt + sigma * np.sqrt(V[:, t - 1]) * dW2[:, t - 1], 0)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * V[:, t - 1]) * dt + np.sqrt(V[:, t - 1]) * dW1[:, t - 1])

    # Payoff for a European Call Option (S_T - K)+
    call_payoff = np.maximum(S[:, -1] - K, 0)

    # Discounting the payoff to present value
    call_price = np.exp(-r * T) * np.mean(call_payoff)
    print(f"Heston Call Option Price: {call_price:.2f}")