import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import logging
import datetime
import numpy as np
from scipy.stats import norm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import matplotlib.dates as mdates
from PIL import Image, ImageTk
import requests
from io import BytesIO
import webbrowser

# Configure logging
logging.basicConfig(filename='wheel_strategy.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize meter_frame as None
meter_frame = None

class DraggableResizableButton(tk.Button):
    """
    A Tkinter Button that can be dragged and resized by the user.
    Left-click and drag to move.
    Right-click and drag to resize.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master

        # Initialize dragging variables
        self.dragging = False
        self.resizing = False
        self.start_x = 0
        self.start_y = 0
        self.start_width = 0
        self.start_height = 0

        # Bind mouse events
        self.bind("<ButtonPress-1>", self.on_left_press)
        self.bind("<B1-Motion>", self.on_left_drag)
        self.bind("<ButtonRelease-1>", self.on_left_release)

        self.bind("<ButtonPress-3>", self.on_right_press)
        self.bind("<B3-Motion>", self.on_right_drag)
        self.bind("<ButtonRelease-3>", self.on_right_release)

    def on_left_press(self, event):
        """Start dragging on left mouse button press."""
        self.dragging = True
        self.start_x = event.x
        self.start_y = event.y

    def on_left_drag(self, event):
        """Handle dragging motion."""
        if self.dragging:
            x = self.winfo_x() + event.x - self.start_x
            y = self.winfo_y() + event.y - self.start_y
            # Ensure the button stays within the window bounds
            x = max(0, min(x, self.master.winfo_width() - self.winfo_width()))
            y = max(0, min(y, self.master.winfo_height() - self.winfo_height()))
            self.place(x=x, y=y)

    def on_left_release(self, event):
        """End dragging."""
        self.dragging = False

    def on_right_press(self, event):
        """Start resizing on right mouse button press."""
        self.resizing = True
        self.start_x = event.x
        self.start_y = event.y
        self.start_width = self.winfo_width()
        self.start_height = self.winfo_height()

    def on_right_drag(self, event):
        """Handle resizing motion."""
        if self.resizing:
            # Calculate new size
            new_width = self.start_width + event.x - self.start_x
            new_height = self.start_height + event.y - self.start_y

            # Set minimum size
            min_width = 80
            min_height = 30
            new_width = max(min_width, new_width)
            new_height = max(min_height, new_height)

            # Set maximum size based on window size
            max_width = self.master.winfo_width() - self.winfo_x()
            max_height = self.master.winfo_height() - self.winfo_y()
            new_width = min(new_width, max_width)
            new_height = min(new_height, max_height)

            self.place(width=new_width, height=new_height)

    def on_right_release(self, event):
        """End resizing."""
        self.resizing = False

def calculate_black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes Delta and Theta for European options.
    """
    if T <= 0 or sigma <= 0:
        return 'N/A', 'N/A'

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return delta, theta

def calculate_rsi(df, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame.
    """
    try:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        return pd.Series()

def fetch_stock_data(ticker):
    """
    Fetch stock data including current price, sector, implied volatility, delta, theta, option premium, and RSI.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Fetch option chain for the nearest expiration date
        if not stock.options:
            logging.error(f"No options available for {ticker}")
            atm_call = None
        else:
            try:
                nearest_expiration = stock.options[0]
                options = stock.option_chain(nearest_expiration)
                if not options.calls.empty:
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                    # Find ATM call option
                    atm_call_index = (options.calls['strike'] - current_price).abs().idxmin()
                    atm_call = options.calls.loc[atm_call_index]
                else:
                    atm_call = None
            except Exception as e:
                logging.error(f"Error fetching options for {ticker}: {e}")
                atm_call = None

        if atm_call is not None and 'impliedVolatility' in atm_call:
            # Calculate time to expiration
            expiration_date = datetime.datetime.strptime(nearest_expiration, '%Y-%m-%d')
            today = datetime.datetime.today()
            T = (expiration_date - today).days / 365.25  # Time in years

            # Assume a risk-free rate, e.g., 1%
            r = 0.01

            # Fetch required parameters for Black-Scholes
            S = current_price
            K = atm_call['strike']
            sigma = atm_call['impliedVolatility']

            # Calculate Delta and Theta
            delta, theta = calculate_black_scholes_greeks(S, K, T, r, sigma, option_type='call')
        else:
            delta, theta = 'N/A', 'N/A'

        # Prepare the data dictionary with validation
        data = {
            "Current Price": current_price if 'current_price' in locals() else 'N/A',
            "Sector": info.get('sector', 'N/A'),
            "IV": f"{atm_call['impliedVolatility'] * 100:.2f}%" if atm_call is not None and 'impliedVolatility' in atm_call else 'N/A',
            "Delta": f"{delta:.2f}" if isinstance(delta, float) else 'N/A',
            "Theta": f"{theta:.2f}" if isinstance(theta, float) else 'N/A',
            "Option Premium": f"${atm_call['lastPrice']:.2f}" if atm_call is not None and 'lastPrice' in atm_call else 'N/A',
            "RSI": 'N/A'  # Placeholder, to be updated later
        }

        # Debugging: Print the prepared data
        print(f"Data for {ticker}: {data}\n")

        return data

    except KeyError as ke:
        logging.error(f"KeyError for {ticker}: {ke}")
        return {
            "Current Price": 'N/A',
            "Sector": 'N/A',
            "IV": 'N/A',
            "Delta": 'N/A',
            "Theta": 'N/A',
            "Option Premium": 'N/A',
            "RSI": 'N/A'
        }
    except Exception as e:
        logging.error(f"Unexpected error for {ticker}: {e}")
        return {
            "Current Price": 'N/A',
            "Sector": 'N/A',
            "IV": 'N/A',
            "Delta": 'N/A',
            "Theta": 'N/A',
            "Option Premium": 'N/A',
            "RSI": 'N/A'
        }

def calculate_support_resistance(df):
    """
    Calculate support and resistance levels using the highest high and lowest low over the past 20 days.
    """
    try:
        # Ensure that the DataFrame has the necessary columns
        if not {'High', 'Low'}.issubset(df.columns):
            return None, None
        
        resistance = df['High'].max()
        support = df['Low'].min()
        return resistance, support
    except Exception as e:
        logging.error(f"Error calculating support/resistance: {e}")
        return None, None

def plot_stock_graph(ticker):
    """
    Fetch historical data, calculate support and resistance, calculate RSI, and plot the stock graph.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo", interval="1d")  # Fetch 6 months of daily data

        if hist.empty:
            messagebox.showerror("Data Error", f"No historical data available for ticker '{ticker}'.")
            return

        # Calculate support and resistance
        resistance, support = calculate_support_resistance(hist)

        # Calculate Moving Averages
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        hist['MA200'] = hist['Close'].rolling(window=200).mean()
        hist['EMA20'] = hist['Close'].ewm(span=20, adjust=False).mean()

        # Calculate RSI
        hist['RSI'] = calculate_rsi(hist, window=14)

        # Calculate Trend Line (Linear Regression)
        hist['Date_Ordinal'] = mdates.date2num(hist.index)
        # Use the last 100 days or the available number of days
        trend_data = hist[-100:] if len(hist) >= 100 else hist
        slope, intercept = np.polyfit(trend_data['Date_Ordinal'], trend_data['Close'], 1)
        trend_line = slope * hist['Date_Ordinal'] + intercept

        # Plotting
        fig = plt.Figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111)

        # Plot candlestick chart using mplfinance
        mpf.plot(hist, type='candle', ax=ax1, style='charles', volume=False, show_nontrading=True)

        # Plot Moving Averages
        ax1.plot(hist.index, hist['MA50'], label='50-day MA', color='blue')
        ax1.plot(hist.index, hist['MA200'], label='200-day MA', color='orange')
        ax1.plot(hist.index, hist['EMA20'], label='20-day EMA', color='purple')

        # Plot Trend Line
        ax1.plot(hist.index, trend_line, label='Trend Line', color='cyan', linestyle='--')

        # Plot support and resistance
        if resistance:
            ax1.axhline(resistance, color='red', linestyle='--', label=f'Resistance: {resistance:.2f}')
        if support:
            ax1.axhline(support, color='green', linestyle='--', label=f'Support: {support:.2f}')

        # Formatting dates on x-axis
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        ax1.set_title(f"{ticker} Stock Price with Support, Resistance, Moving Averages, and Trend Line")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price ($)")
        ax1.legend()

        fig.tight_layout()

        # Embed the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=stock_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        logging.error(f"Error plotting stock graph for {ticker}: {e}")
        messagebox.showerror("Plot Error", f"An error occurred while plotting the stock graph for '{ticker}'.")

def fetch_and_display_news(ticker):
    """
    Fetch the most recent news headlines for the given ticker and display them in the news frame.
    """
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news

        # Clear existing news
        for widget in news_canvas_frame.winfo_children():
            widget.destroy()

        if not news_items:
            no_news_label = tk.Label(news_canvas_frame, text="No recent news available.", wraplength=350, justify="left")
            no_news_label.pack(pady=10)
            return

        # Limit to the 5 most recent news items
        for news in news_items[:5]:
            frame = ttk.Frame(news_canvas_frame, relief=tk.RIDGE, borderwidth=1)
            frame.pack(pady=5, padx=5, fill='x')

            # Fetch image if available
            image_url = news.get('thumbnail', None)
            if image_url:
                try:
                    response = requests.get(image_url)
                    img_data = response.content
                    img = Image.open(BytesIO(img_data))
                    img = img.resize((50, 50), Image.ANTIALIAS)
                    photo = ImageTk.PhotoImage(img)
                except Exception as e:
                    logging.error(f"Error fetching image for news: {e}")
                    # Use a placeholder image
                    photo = ImageTk.PhotoImage(Image.new('RGB', (50, 50), color='gray'))
            else:
                # Use a placeholder image
                photo = ImageTk.PhotoImage(Image.new('RGB', (50, 50), color='gray'))

            # Keep a reference to the image to prevent garbage collection
            frame.image = photo

            img_label = tk.Label(frame, image=photo)
            img_label.pack(side='left', padx=5, pady=5)

            # News details
            news_text_frame = ttk.Frame(frame)
            news_text_frame.pack(side='left', fill='both', expand=True)

            title = news.get('title', 'No Title')
            publisher = news.get('publisher', 'Unknown Publisher')
            link = news.get('link', '#')

            # Adjusted text color and font size for better visibility
            title_label = tk.Label(news_text_frame, text=title, font=('Arial', 10, 'bold'), wraplength=280, justify="left", fg="black", cursor="hand2")
            title_label.pack(anchor='w')
            title_label.bind("<Button-1>", lambda e, url=link: open_url(url))

            publisher_label = tk.Label(news_text_frame, text=publisher, font=('Arial', 8), fg="gray")
            publisher_label.pack(anchor='w')

    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        messagebox.showerror("News Error", f"An error occurred while fetching news for '{ticker}'.")

def open_url(url):
    """
    Open the given URL in the default web browser.
    """
    try:
        webbrowser.open_new_tab(url)
    except Exception as e:
        logging.error(f"Error opening URL {url}: {e}")
        messagebox.showerror("URL Error", f"Failed to open the link: {url}")

def plot_news_scrollbar(news_frame):
    """
    Configure scrollbar for the news canvas.
    """
    global news_canvas_frame  # Declare as global if used elsewhere
    
    news_canvas = tk.Canvas(news_frame, width=400)
    news_canvas.pack(side='left', fill='both', expand=True)

    scrollbar = ttk.Scrollbar(news_frame, orient="vertical", command=news_canvas.yview)
    scrollbar.pack(side='right', fill='y')

    news_canvas.configure(yscrollcommand=scrollbar.set)
    news_canvas.bind('<Configure>', lambda e: news_canvas.configure(scrollregion=news_canvas.bbox("all")))

    news_canvas_frame = ttk.Frame(news_canvas)
    news_canvas.create_window((0, 0), window=news_canvas_frame, anchor='nw')

def reset_stock_chart():
    """
    Reset the stock chart by clearing the frame and updating the status.
    """
    # Clear Stock Chart Frame
    for widget in stock_chart_frame.winfo_children():
        widget.destroy()
    
    # Update status
    status_label.config(text="Stock chart has been reset.")

def reset_news():
    """
    Reset the news frame by clearing existing news.
    """
    for widget in news_canvas_frame.winfo_children():
        widget.destroy()
    status_label.config(text="News has been reset.")

def reset_data_and_graphs():
    """
    Reset both the Treeview and the stock chart and news.
    """
    # Clear Treeview
    for item in tree.get_children():
        tree.delete(item)
    
    # Clear Stock Chart Frame
    reset_stock_chart()
    
    # Clear News Frame
    reset_news()
    
    # Reset Bullish/Bearish Meter
    reset_meter()
    
    # Update status
    status_label.config(text="Data and graphs have been reset.")

def search_stock():
    """
    Handle the search operation: fetch data, update the Treeview, plot the graph, and display news.
    """
    # Get the ticker symbol from the entry widget
    ticker = ticker_entry.get().upper().strip()
    
    if not ticker:
        messagebox.showwarning("Input Error", "Please enter a stock ticker symbol.")
        return
    
    # Clear existing data in the treeview
    for item in tree.get_children():
        tree.delete(item)
    
    # Clear existing graph and news
    reset_stock_chart()
    reset_news()
    reset_meter()
    
    try:
        data = fetch_stock_data(ticker)
        
        # Fetch historical data to calculate RSI
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo", interval="1d")
        if not hist.empty:
            hist['RSI'] = calculate_rsi(hist, window=14)
            latest_rsi = hist['RSI'].iloc[-1]
            if not np.isnan(latest_rsi):
                data["RSI"] = f"{latest_rsi:.2f}"
            else:
                data["RSI"] = 'N/A'
        else:
            data["RSI"] = 'N/A'
        
        # Check if data is all 'N/A' indicating a possible invalid ticker
        if all(value == 'N/A' for key, value in data.items() if key != "RSI"):
            messagebox.showerror("Data Error", f"No data available for ticker '{ticker}'. Please check the symbol and try again.")
            return
        
        # Define the order of values explicitly
        values = (
            ticker,
            data["Current Price"],
            data["Sector"],
            data["IV"],
            data["Delta"],
            data["Theta"],
            data["Option Premium"],
            data["RSI"]
        )
        tree.insert("", "end", values=values)
        status_label.config(text=f"Data for '{ticker}' updated successfully.")
        
        # Plot the stock graph with support, resistance, moving averages, and trend line
        plot_stock_graph(ticker)
        
        # Fetch and display news
        fetch_and_display_news(ticker)
        
        # Plot the bullish/bearish meter
        plot_meter(hist)
        
        # Schedule the next update
        root.after(update_interval, lambda: update_data(ticker))
        
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        messagebox.showerror("Error", f"An error occurred while fetching data for '{ticker}'. Please try again later.")
        status_label.config(text="Failed to update data.")

def update_data(ticker):
    """
    Periodically fetch and update the data, plot for the given ticker, and refresh news and meter.
    """
    try:
        # Fetch updated data
        data = fetch_stock_data(ticker)
        
        # Fetch historical data to calculate RSI
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo", interval="1d")
        if not hist.empty:
            hist['RSI'] = calculate_rsi(hist, window=14)
            latest_rsi = hist['RSI'].iloc[-1]
            if not np.isnan(latest_rsi):
                data["RSI"] = f"{latest_rsi:.2f}"
            else:
                data["RSI"] = 'N/A'
        else:
            data["RSI"] = 'N/A'
        
        # Update Treeview
        # Clear existing data
        for item in tree.get_children():
            tree.delete(item)
        
        # Insert updated data
        values = (
            ticker,
            data["Current Price"],
            data["Sector"],
            data["IV"],
            data["Delta"],
            data["Theta"],
            data["Option Premium"],
            data["RSI"]
        )
        tree.insert("", "end", values=values)
        status_label.config(text=f"Data for '{ticker}' updated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Update Stock Graph
        reset_stock_chart()
        plot_stock_graph(ticker)
        
        # Update News
        reset_news()
        fetch_and_display_news(ticker)
        
        # Update Bullish/Bearish Meter
        reset_meter()
        plot_meter(hist)
        
        # Schedule the next update
        root.after(update_interval, lambda: update_data(ticker))
        
    except Exception as e:
        logging.error(f"Error updating data for {ticker}: {e}")
        messagebox.showerror("Update Error", f"An error occurred while updating data for '{ticker}'.")
        status_label.config(text="Failed to update data.")

def export_to_csv():
    """
    Export the current data to a CSV file.
    """
    try:
        # Fetch the current ticker
        ticker = ticker_entry.get().upper().strip()
        if not ticker:
            messagebox.showwarning("Input Error", "Please enter a stock ticker symbol.")
            return
        
        # Get the data from the Treeview
        items = tree.get_children()
        if not items:
            messagebox.showwarning("No Data", "No data available to export.")
            return
        
        # Collect data
        data = []
        for item in items:
            data.append(tree.item(item, 'values'))
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=tree["columns"])
        
        # Save to CSV
        df.to_csv(f"{ticker}_data.csv", index=False)
        messagebox.showinfo("Export Successful", f"Data exported to {ticker}_data.csv")
    except Exception as e:
        logging.error(f"Error exporting data for {ticker}: {e}")
        messagebox.showerror("Export Error", f"An error occurred while exporting data for '{ticker}'.")

def save_graph():
    """
    Save the current stock graph as a PNG image.
    """
    try:
        # Fetch the current ticker
        ticker = ticker_entry.get().upper().strip()
        if not ticker:
            messagebox.showwarning("Input Error", "Please enter a stock ticker symbol.")
            return
        
        # Save stock chart
        for widget in stock_chart_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.figure.savefig(f"{ticker}_stock_chart.png")
        
        # Save Bullish/Bearish Meter
        if hasattr(meter_frame, 'meter_fig'):
            meter_frame.meter_fig.savefig(f"{ticker}_sentiment_meter.png")
        
        messagebox.showinfo("Save Successful", f"Graphs saved as {ticker}_stock_chart.png and {ticker}_sentiment_meter.png")
    except Exception as e:
        logging.error(f"Error saving graph for {ticker}: {e}")
        messagebox.showerror("Save Error", f"An error occurred while saving the graph for '{ticker}'.")

def plot_meter(hist):
    """
    Plot a bullish/bearish sentiment meter based on RSI.
    """
    global meter_frame
    try:
        latest_rsi = hist['RSI'].iloc[-1]
        if pd.isna(latest_rsi):
            latest_rsi = 50  # Neutral if RSI is not available

        # Determine sentiment based on RSI
        if latest_rsi < 30:
            sentiment = 'Bullish'
            rsi_value = latest_rsi
        elif latest_rsi > 70:
            sentiment = 'Bearish'
            rsi_value = latest_rsi
        else:
            sentiment = 'Neutral'
            rsi_value = latest_rsi

        # Create a frame for the meter
        meter_frame = ttk.Frame(right_frame)
        meter_frame.pack(pady=20, padx=10, fill='x')

        # Create matplotlib figure for the meter
        fig, ax = plt.subplots(figsize=(4, 2), subplot_kw={'projection': 'polar'})

        # Hide the polar coordinates
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_ylim(0, 100)
        ax.axis('off')

        # Define the meter
        # Background arc
        ax.barh(0, 100, height=1, color='lightgray', alpha=0.3)

        # Define colors based on sentiment
        if sentiment == 'Bullish':
            color = 'green'
        elif sentiment == 'Bearish':
            color = 'red'
        else:
            color = 'yellow'

        # Plot the sentiment
        ax.barh(0, rsi_value, height=1, color=color)

        # Add text
        ax.text(0, 50, f"RSI: {latest_rsi:.1f}\nSentiment: {sentiment}", 
                horizontalalignment='center', verticalalignment='center', fontsize=10, fontweight='bold')

        # Embed the meter in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=meter_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Save the figure for saving functionality
        meter_frame.meter_fig = fig

    except Exception as e:
        logging.error(f"Error plotting sentiment meter: {e}")
        messagebox.showerror("Meter Error", f"An error occurred while plotting the sentiment meter.")

def reset_meter():
    """
    Reset the sentiment meter by clearing the frame.
    """
    global meter_frame
    if meter_frame and hasattr(meter_frame, 'winfo_children'):
        for widget in meter_frame.winfo_children():
            widget.destroy()
        status_label.config(text="Sentiment meter has been reset.")

# Create the main window
root = tk.Tk()
root.title("Wheel Strategy Stocks")
root.geometry("1800x800")  # Increased width to accommodate the news section and meter

# Create a main frame to hold left and right sections
main_frame = ttk.Frame(root)
main_frame.pack(fill='both', expand=True)

# Left Frame: Search Bar, Treeview, Stock Chart
left_frame = ttk.Frame(main_frame)
left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

# Frame for the search bar within the left frame
search_frame = tk.Frame(left_frame)
search_frame.pack(pady=10, anchor='w')

# Label for ticker entry
ticker_label = tk.Label(search_frame, text="Enter Ticker Symbol:")
ticker_label.pack(side='left', padx=(0, 10))

# Entry widget for ticker symbol
ticker_entry = tk.Entry(search_frame, width=20)
ticker_entry.pack(side='left', padx=(0, 10))

# Create and pack the search button
search_button = tk.Button(search_frame, text="Search", command=search_stock)
search_button.pack(side='left')

# Create Export and Save buttons
export_button = tk.Button(search_frame, text="Export CSV", command=export_to_csv)
export_button.pack(side='left', padx=(10, 0))

save_button = tk.Button(search_frame, text="Save Graphs", command=save_graph)
save_button.pack(side='left', padx=(10, 0))

# Create and pack the status label within the left frame
status_label = tk.Label(left_frame, text="")
status_label.pack(pady=5, anchor='w')

# Create the treeview within the left frame
columns = ("Ticker", "Current Price", "Sector", "IV", "Delta", "Theta", "Option Premium", "RSI")
tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=5)

# Set column headings
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=140, anchor="center")  # Adjusted width for better visibility

# Add a vertical scrollbar to the treeview
vsb = ttk.Scrollbar(left_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=vsb.set)
vsb.pack(side='right', fill='y')

# Pack the treeview within the left frame
tree.pack(pady=10, fill='x')

# Frame for the stock chart within the left frame
stock_chart_frame = ttk.Frame(left_frame)
stock_chart_frame.pack(pady=10, fill='both', expand=True)

# Create the Draggable and Resizable Reset Button for Stock Chart
reset_stock_button = DraggableResizableButton(
    master=stock_chart_frame,
    text="Reset Chart",
    bg="lightgreen",
    relief=tk.RAISED,
    cursor="arrow"
)

# Define button dimensions and padding
button_width = 100
button_height = 30
padding_x = 10
padding_y = 10

# Place the button at the bottom-right of the stock chart frame using relative positioning
reset_stock_button.place(
    relx=1.0,
    rely=1.0,
    x=-padding_x - button_width,
    y=-padding_y - button_height,
    width=button_width,
    height=button_height,
    anchor='se'
)

# Assign the action to the button directly without defining a separate function
reset_stock_button.config(command=reset_stock_chart)

# Function to handle frame resizing for stock chart button
def on_stock_frame_resize(event):
    try:
        # Only reposition if the button hasn't been moved by the user
        if not reset_stock_button.dragging and not reset_stock_button.resizing:
            reset_stock_button.place(
                relx=1.0,
                rely=1.0,
                x=-padding_x - reset_stock_button.winfo_width(),
                y=-padding_y - reset_stock_button.winfo_height(),
                width=reset_stock_button.winfo_width(),
                height=reset_stock_button.winfo_height(),
                anchor='se'
            )
    except Exception as e:
        logging.error(f"Error repositioning stock reset button on resize: {e}")

# Bind resize event to the stock chart frame
stock_chart_frame.bind("<Configure>", on_stock_frame_resize)

# Right Frame: News Headlines and Sentiment Meter
right_frame = ttk.Frame(main_frame, width=400)
right_frame.pack(side='right', fill='y', padx=10, pady=10)

# Define news_frame within right_frame
news_frame = ttk.Frame(right_frame)
news_frame.pack(fill='both', expand=True)

# News Section Configuration
plot_news_scrollbar(news_frame)  # Initialize the scrollable news frame with the defined news_frame

# Real-time update interval in milliseconds (e.g., 60000 ms = 60 seconds)
update_interval = 60000

# Start the GUI event loop
root.mainloop()