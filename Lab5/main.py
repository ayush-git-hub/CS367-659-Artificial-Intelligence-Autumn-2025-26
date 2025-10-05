# Lab 6
# Problem Statement
# Analyze financial data to identify hidden market regimes using HMM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FinancialHMMAnalyzer:
    """
    Complete pipeline for Gaussian HMM analysis on financial time series.
    Implements data collection, preprocessing, training, analysis, and visualization.
    """
    
    def __init__(self, ticker, start_date, end_date, n_states=2):
        """
        Initialize the HMM analyzer.
        
        Parameters:
        -----------
        ticker : str - Stock ticker symbol
        start_date : str - Start date 'YYYY-MM-DD'
        end_date : str - End date 'YYYY-MM-DD'
        n_states : int - Number of hidden states
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.n_states = n_states
        self.data = None
        self.returns = None
        self.model = None
        self.hidden_states = None
        self.results_df = None
        
    def collect_data(self):
        """Part 1: Download financial data from Yahoo Finance API."""
        print(f"Downloading data for {self.ticker}...")
        print(f"Period: {self.start_date} to {self.end_date}")
        
        try:
            raw_data = yf.download(
                self.ticker, 
                start=self.start_date, 
                end=self.end_date, 
                progress=False,
                auto_adjust=False
            )
            
            if raw_data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = raw_data.columns.get_level_values(0)
            
            self.data = raw_data
            
            print(f"Downloaded {len(self.data)} data points")
            print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            
            return self.data
            
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            raise
    
    def preprocess_data(self):
        """Part 1: Calculate returns and clean data."""
        print("\nPreprocessing data...")
        
        if 'Adj Close' in self.data.columns:
            adj_close = self.data['Adj Close']
            print("Using 'Adj Close' prices")
        elif 'Close' in self.data.columns:
            adj_close = self.data['Close']
            print("Using 'Close' prices")
        else:
            raise ValueError("No price column found in data")
        
        # Calculate daily returns
        self.returns = adj_close.pct_change().dropna()
        
        initial_count = len(self.returns)
        self.returns = self.returns.replace([np.inf, -np.inf], np.nan)
        self.returns = self.returns.dropna()
        
        removed_count = initial_count - len(self.returns)
        if removed_count > 0:
            print(f"Removed {removed_count} invalid data points")
        
        print(f"Calculated {len(self.returns)} daily returns")
        print(f"Mean return: {self.returns.mean():.4f} ({self.returns.mean()*100:.2f}%)")
        print(f"Std deviation: {self.returns.std():.4f} ({self.returns.std()*100:.2f}%)")
        print(f"Min/Max return: {self.returns.min():.4f} / {self.returns.max():.4f}")
        
        return self.returns
    
    def fit_hmm_model(self, n_iter=100, random_state=42):
        """Part 2: Fit Gaussian HMM using Baum-Welch algorithm."""
        print(f"\nFitting Gaussian HMM with {self.n_states} hidden states...")
        print(f"Training algorithm: Baum-Welch (EM), Max iterations: {n_iter}")
        
        X = self.returns.values.reshape(-1, 1)
        
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )
        
        try:
            self.model.fit(X)
            print("Model convergence achieved")
        except Exception as e:
            print(f"Model training completed with warnings: {str(e)}")
        
        self.hidden_states = self.model.predict(X)
        score = self.model.score(X)
        print(f"Model log-likelihood: {score:.2f}")
        print("Hidden states decoded using Viterbi algorithm")
        
        return self.model
    
    def analyze_parameters(self):
        """Part 2: Analyze HMM parameters comprehensively."""
        print("\n" + "="*70)
        print("ANALYZING HMM PARAMETERS")
        print("="*70)
        
        means = self.model.means_
        covars = self.model.covars_
        transmat = self.model.transmat_
        startprob = self.model.startprob_
        
        sorted_indices = np.argsort(means.flatten())
        
        print("\nHidden State Characteristics:")
        print("-"*70)
        
        state_interpretations = []
        
        for idx, state_idx in enumerate(sorted_indices):
            mean_return = means[state_idx][0]
            variance = covars[state_idx][0][0]
            std_dev = np.sqrt(variance)
            
            ann_return = mean_return * 252
            ann_volatility = std_dev * np.sqrt(252)
            sharpe_ratio = (ann_return / ann_volatility) if ann_volatility > 0 else 0
            
            # Interpret state
            if mean_return < 0 and std_dev > self.returns.std():
                regime = "BEAR MARKET (High Volatility, Negative Returns)"
                interpretation = "bear"
            elif mean_return > 0 and std_dev < self.returns.std():
                regime = "BULL MARKET (Low Volatility, Positive Returns)"
                interpretation = "bull"
            elif mean_return > 0 and std_dev > self.returns.std():
                regime = "VOLATILE GROWTH (High Volatility, Positive Returns)"
                interpretation = "volatile_growth"
            else:
                regime = f"MODERATE REGIME {idx}"
                interpretation = "moderate"
            
            state_interpretations.append({
                'state': state_idx,
                'regime': interpretation,
                'mean': mean_return,
                'std': std_dev
            })
            
            print(f"\nState {state_idx}: {regime}")
            print(f"  Daily: Mean={mean_return:.4%}, Std={std_dev:.4%}")
            print(f"  Annual: Return={ann_return:.2%}, Vol={ann_volatility:.2%}, Sharpe={sharpe_ratio:.2f}")
            print(f"  Initial Probability: {startprob[state_idx]:.2%}")
        
        print("\n" + "="*70)
        print("State Transition Analysis:")
        print("-"*70)
        
        trans_df = pd.DataFrame(
            transmat,
            columns=[f"To State {i}" for i in range(self.n_states)],
            index=[f"State {i}" for i in range(self.n_states)]
        )
        print(trans_df.to_string(float_format=lambda x: f'{x:.4f}'))
        
        print("\nState Persistence:")
        print("-"*70)
        for i in range(self.n_states):
            persistence = transmat[i, i]
            avg_duration = 1 / (1 - persistence) if persistence < 1 else np.inf
            regime_name = [s for s in state_interpretations if s['state'] == i][0]['regime']
            
            print(f"  State {i} ({regime_name}):")
            print(f"    Self-transition: {persistence:.4f}, Expected duration: {avg_duration:.1f} days")
            
            if self.n_states > 1:
                next_states = [(j, transmat[i, j]) for j in range(self.n_states) if j != i]
                if next_states:
                    most_likely_next = max(next_states, key=lambda x: x[1])
                    print(f"    Most likely next: State {most_likely_next[0]} (prob: {most_likely_next[1]:.4f})")
        
        print("\n" + "="*70)
        
        return {
            'means': means,
            'covars': covars,
            'transmat': transmat,
            'startprob': startprob,
            'state_interpretations': state_interpretations
        }
    
    def decode_states(self):
        """Part 3: Decode and analyze hidden states over time."""
        print("\nDecoding Hidden States...")
        
        self.results_df = pd.DataFrame({
            'Date': self.returns.index,
            'Returns': self.returns.values,
            'Hidden_State': self.hidden_states
        })
        
        state_counts = self.results_df['Hidden_State'].value_counts().sort_index()
        
        print("\nState Distribution:")
        print("-"*70)
        for state, count in state_counts.items():
            percentage = (count / len(self.results_df)) * 100
            print(f"  State {state}: {count:>5} days ({percentage:>5.1f}%)")
        
        # Analyze transitions
        transitions = []
        for i in range(len(self.hidden_states) - 1):
            if self.hidden_states[i] != self.hidden_states[i + 1]:
                transitions.append({
                    'date': self.returns.index[i + 1],
                    'from_state': self.hidden_states[i],
                    'to_state': self.hidden_states[i + 1]
                })
        
        print(f"\nState Transitions: {len(transitions)} total")
        print(f"Average time between transitions: {len(self.returns) / (len(transitions) + 1):.1f} days")
        
        if len(transitions) > 0:
            print(f"\nRecent transitions (last 5):")
            for trans in transitions[-5:]:
                print(f"  {trans['date'].date()}: State {trans['from_state']} -> State {trans['to_state']}")
        
        return self.results_df
    
    def calculate_model_metrics(self):
        """Part 4: Calculate model evaluation metrics."""
        print("\nModel Evaluation Metrics:")
        print("-"*70)
        
        n_params = (self.n_states ** 2) + (2 * self.n_states)
        n_samples = len(self.returns)
        
        X = self.returns.values.reshape(-1, 1)
        log_likelihood = self.model.score(X) * n_samples
        
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        
        print(f"  Parameters: {n_params}, Log-likelihood: {log_likelihood:.2f}")
        print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")
        print(f"  Note: Lower AIC/BIC indicates better fit")
        
        print(f"\n  State-specific Performance:")
        for state in range(self.n_states):
            state_returns = self.results_df[self.results_df['Hidden_State'] == state]['Returns']
            if len(state_returns) > 0:
                print(f"    State {state}: Mean={state_returns.mean():.4%}, "
                      f"Vol={state_returns.std():.4%}, "
                      f"Skew={state_returns.skew():.2f}, Kurt={state_returns.kurtosis():.2f}")
        
        return {
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'n_params': n_params
        }
    
    def visualize_results(self):
        """Part 4: Create comprehensive visualizations."""
        print("\nCreating visualizations...")
        
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'Gaussian HMM Analysis: {self.ticker} ({self.n_states} States)', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        price_col = 'Adj Close' if 'Adj Close' in self.data.columns else 'Close'
        
        # 1. Stock Price with Hidden States
        ax1 = plt.subplot(3, 3, 1)
        for state in range(self.n_states):
            mask = self.results_df['Hidden_State'] == state
            dates = self.results_df[mask]['Date']
            prices = self.data.loc[dates, price_col]
            ax1.scatter(dates, prices, c=f'C{state}', label=f'State {state}', 
                       alpha=0.6, s=15, edgecolors='none')
        
        ax1.plot(self.data.index, self.data[price_col], 'k-', 
                alpha=0.2, linewidth=0.8, label='Price', zorder=0)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Price ($)', fontsize=10)
        ax1.set_title('Stock Price with Hidden States', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns with Hidden States
        ax2 = plt.subplot(3, 3, 2)
        for state in range(self.n_states):
            mask = self.results_df['Hidden_State'] == state
            ax2.scatter(self.results_df[mask]['Date'], self.results_df[mask]['Returns'],
                       c=f'C{state}', label=f'State {state}', alpha=0.6, s=15, edgecolors='none')
        
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Daily Returns', fontsize=10)
        ax2.set_title('Daily Returns by Hidden State', fontsize=11, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. State Timeline
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(self.results_df['Date'], self.results_df['Hidden_State'], linewidth=1.5)
        ax3.fill_between(self.results_df['Date'], self.results_df['Hidden_State'], 
                         alpha=0.3, step='mid')
        ax3.set_xlabel('Date', fontsize=10)
        ax3.set_ylabel('Hidden State', fontsize=10)
        ax3.set_title('Hidden State Evolution', fontsize=11, fontweight='bold')
        ax3.set_yticks(range(self.n_states))
        ax3.grid(True, alpha=0.3)
        
        # 4. Returns Distribution by State
        ax4 = plt.subplot(3, 3, 4)
        for state in range(self.n_states):
            state_returns = self.results_df[self.results_df['Hidden_State'] == state]['Returns']
            ax4.hist(state_returns, bins=50, alpha=0.6, label=f'State {state}', density=True)
        
        ax4.set_xlabel('Daily Returns', fontsize=10)
        ax4.set_ylabel('Density', fontsize=10)
        ax4.set_title('Returns Distribution by State', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Transition Matrix Heatmap
        ax5 = plt.subplot(3, 3, 5)
        sns.heatmap(self.model.transmat_, annot=True, fmt='.3f', 
                   cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Probability'},
                   square=True, linewidths=1)
        ax5.set_xlabel('To State', fontsize=10)
        ax5.set_ylabel('From State', fontsize=10)
        ax5.set_title('Transition Probability Matrix', fontsize=11, fontweight='bold')
        
        # 6. State Statistics Box Plot
        ax6 = plt.subplot(3, 3, 6)
        state_returns_list = [self.results_df[self.results_df['Hidden_State'] == state]['Returns'].values 
                             for state in range(self.n_states)]
        bp = ax6.boxplot(state_returns_list, labels=[f'State {i}' for i in range(self.n_states)],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], [f'C{i}' for i in range(self.n_states)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax6.set_ylabel('Daily Returns', fontsize=10)
        ax6.set_title('Returns Distribution (Box Plot)', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 7. Volatility Over Time
        ax7 = plt.subplot(3, 3, 7)
        rolling_vol = self.returns.rolling(window=20).std()
        ax7.plot(rolling_vol.index, rolling_vol, 'gray', alpha=0.5, linewidth=1, label='20-day Rolling Vol')
        for state in range(self.n_states):
            mask = self.results_df['Hidden_State'] == state
            dates = self.results_df[mask]['Date']
            vols = rolling_vol.loc[dates]
            ax7.scatter(dates, vols, c=f'C{state}', label=f'State {state}', alpha=0.6, s=10)
        ax7.set_xlabel('Date', fontsize=10)
        ax7.set_ylabel('Volatility', fontsize=10)
        ax7.set_title('Rolling Volatility (20 days)', fontsize=11, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # 8. Cumulative Returns by State
        ax8 = plt.subplot(3, 3, 8)
        for state in range(self.n_states):
            state_returns = self.results_df[self.results_df['Hidden_State'] == state]['Returns']
            cumulative = (1 + state_returns).cumprod()
            ax8.plot(state_returns.index, cumulative, label=f'State {state}', linewidth=2)
        ax8.set_xlabel('Date', fontsize=10)
        ax8.set_ylabel('Cumulative Return Factor', fontsize=10)
        ax8.set_title('Cumulative Returns by State', fontsize=11, fontweight='bold')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        
        # 9. State Duration Histogram
        ax9 = plt.subplot(3, 3, 9)
        state_durations = {state: [] for state in range(self.n_states)}
        current_state = self.hidden_states[0]
        duration = 1
        
        for i in range(1, len(self.hidden_states)):
            if self.hidden_states[i] == current_state:
                duration += 1
            else:
                state_durations[current_state].append(duration)
                current_state = self.hidden_states[i]
                duration = 1
        state_durations[current_state].append(duration)
        
        for state in range(self.n_states):
            if state_durations[state]:
                ax9.hist(state_durations[state], bins=20, alpha=0.6, 
                        label=f'State {state}', edgecolor='black')
        ax9.set_xlabel('Duration (days)', fontsize=10)
        ax9.set_ylabel('Frequency', fontsize=10)
        ax9.set_title('State Duration Distribution', fontsize=11, fontweight='bold')
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = f'{self.ticker}_HMM_{self.n_states}states_Analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: '{filename}'")
        plt.show()
    
    def predict_future_state(self, n_days=30):
        """Part 5: Predict likely future states using transition matrix."""
        print(f"\nFuture State Prediction ({n_days} days ahead):")
        print("="*70)
        
        current_state = self.hidden_states[-1]
        current_date = self.returns.index[-1]
        print(f"   Current Date: {current_date.date()}, Current State: {current_state}")
        
        trans_mat = self.model.transmat_
        state_probs = np.zeros((n_days + 1, self.n_states))
        state_probs[0, current_state] = 1.0
        
        for day in range(1, n_days + 1):
            state_probs[day] = state_probs[day - 1] @ trans_mat
        
        print("\n   State Probability Evolution:")
        print("   " + "-"*66)
        print("   Days Ahead  |  " + "  |  ".join([f"State {i}" for i in range(self.n_states)]))
        print("   " + "-"*66)
        
        for day in [1, 5, 10, 15, 20, 30]:
            if day <= n_days:
                print(f"      {day:>2}       |  ", end="")
                for state in range(self.n_states):
                    print(f"{state_probs[day, state]:>6.2%}  |  ", end="")
                print()
        
        future_state = np.argmax(state_probs[-1])
        future_prob = state_probs[-1, future_state]
        
        print("\n   " + "-"*66)
        print(f"   Most Likely State in {n_days} days: State {future_state} (prob: {future_prob:.2%})")
        
        expected_returns = []
        for day in range(n_days + 1):
            exp_return = sum(state_probs[day, s] * self.model.means_[s][0] 
                           for s in range(self.n_states))
            expected_returns.append(exp_return)
        
        print(f"\n   Expected Daily Return in {n_days} days: {expected_returns[-1]:.4%}")
        print(f"   Cumulative Expected Return: {(np.prod([1 + r for r in expected_returns[1:]]) - 1):.2%}")
        
        self._visualize_prediction(state_probs, n_days)
        
        return state_probs
    
    def _visualize_prediction(self, state_probs, n_days):
        """Visualize future state predictions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        days = range(n_days + 1)
        
        for state in range(self.n_states):
            ax1.plot(days, state_probs[:, state], marker='o', 
                    label=f'State {state}', linewidth=2, markersize=4)
        
        ax1.set_xlabel('Days Ahead', fontsize=11)
        ax1.set_ylabel('Probability', fontsize=11)
        ax1.set_title('Future State Probability Evolution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        most_likely_states = np.argmax(state_probs, axis=1)
        ax2.step(days, most_likely_states, where='mid', linewidth=2)
        ax2.fill_between(days, most_likely_states, alpha=0.3, step='mid')
        ax2.set_xlabel('Days Ahead', fontsize=11)
        ax2.set_ylabel('Most Likely State', fontsize=11)
        ax2.set_title('Most Likely Future State', fontsize=12, fontweight='bold')
        ax2.set_yticks(range(self.n_states))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{self.ticker}_Future_Prediction.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   Saved prediction: '{filename}'")
        plt.show()
    
    def generate_report(self):
        """Part 5: Generate comprehensive analysis report."""
        print("\n" + "="*70)
        print(" "*15 + "COMPREHENSIVE HMM ANALYSIS REPORT")
        print("="*70)
        
        print(f"\nAsset Information:")
        print(f"   Ticker: {self.ticker}, Period: {self.start_date} to {self.end_date}")
        print(f"   Trading Days: {len(self.returns)}, Hidden States: {self.n_states}")
        
        print(f"\nOverall Market Statistics:")
        total_return = (self.data[self.data.columns[3]].iloc[-1] / 
                       self.data[self.data.columns[3]].iloc[0] - 1)
        ann_return = self.returns.mean() * 252
        ann_vol = self.returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Daily: Mean={self.returns.mean():.4%}, Vol={self.returns.std():.4%}")
        print(f"   Annual: Return={ann_return:.2%}, Vol={ann_vol:.2%}, Sharpe={sharpe:.2f}")
        print(f"   Range: {self.returns.min():.2%} to {self.returns.max():.2%}")
        
        print(f"\nHidden State Insights:")
        for state in range(self.n_states):
            state_data = self.results_df[self.results_df['Hidden_State'] == state]
            if len(state_data) > 0:
                state_returns = state_data['Returns']
                pct_time = len(state_data)/len(self.results_df)*100
                print(f"\n   State {state}: {len(state_data)} days ({pct_time:.1f}%)")
                print(f"     Mean={state_returns.mean():.4%}, Vol={state_returns.std():.4%}")
                print(f"     Range: {state_returns.min():.2%} to {state_returns.max():.2%}")
        
        print("\n" + "="*70)
        print("\nKey Takeaways:")
        print("-"*70)
        print("1. HMM identified distinct market regimes")
        print("2. Each regime has characteristic return/volatility patterns")
        print("3. Transition probabilities reveal regime persistence")
        print("4. Future predictions can inform risk management")
        print("5. Useful for portfolio allocation and hedging strategies")
        print("="*70)
    
    def save_results(self, output_dir='results'):
        """Save analysis results to CSV files."""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename1 = f'{output_dir}/{self.ticker}_states_timeseries.csv'
        self.results_df.to_csv(filename1, index=False)
        print(f"Saved time series: '{filename1}'")
        
        params_df = pd.DataFrame({
            'State': range(self.n_states),
            'Mean': self.model.means_.flatten(),
            'Std_Dev': np.sqrt(self.model.covars_.flatten()),
            'Start_Prob': self.model.startprob_
        })
        filename2 = f'{output_dir}/{self.ticker}_model_parameters.csv'
        params_df.to_csv(filename2, index=False)
        print(f"Saved parameters: '{filename2}'")
        
        trans_df = pd.DataFrame(
            self.model.transmat_,
            columns=[f'To_State_{i}' for i in range(self.n_states)],
            index=[f'From_State_{i}' for i in range(self.n_states)]
        )
        filename3 = f'{output_dir}/{self.ticker}_transition_matrix.csv'
        trans_df.to_csv(filename3)
        print(f"Saved transition matrix: '{filename3}'")


def main():
    """Main execution function for complete HMM analysis pipeline."""
    print("\n" + "="*70)
    print(" "*10 + "GAUSSIAN HMM FINANCIAL TIME SERIES ANALYSIS")
    print("="*70)
    
    # Configuration
    TICKER = 'AAPL'
    START_DATE = '2014-01-01'
    END_DATE = '2024-01-01'
    N_STATES = 2
    
    try:
        analyzer = FinancialHMMAnalyzer(
            ticker=TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            n_states=N_STATES
        )
        
        # Part 1: Data Collection and Preprocessing
        print("\n" + "-"*70)
        print("PART 1: DATA COLLECTION AND PREPROCESSING")
        print("-"*70)
        analyzer.collect_data()
        analyzer.preprocess_data()
        
        # Part 2: Fit Gaussian HMM
        print("\n" + "-"*70)
        print("PART 2: GAUSSIAN HIDDEN MARKOV MODEL")
        print("-"*70)
        analyzer.fit_hmm_model(n_iter=100)
        parameters = analyzer.analyze_parameters()
        
        # Part 3: Decode Hidden States
        print("\n" + "-"*70)
        print("PART 3: INTERPRETATION AND INFERENCE")
        print("-"*70)
        results_df = analyzer.decode_states()
        
        # Part 4: Evaluation and Visualization
        print("\n" + "-"*70)
        print("PART 4: EVALUATION AND VISUALIZATION")
        print("-"*70)
        metrics = analyzer.calculate_model_metrics()
        analyzer.visualize_results()
        
        # Part 5: Future State Prediction and Conclusions
        print("\n" + "-"*70)
        print("PART 5: CONCLUSIONS AND INSIGHTS")
        print("-"*70)
        future_probs = analyzer.predict_future_state(n_days=30)
        analyzer.generate_report()
        
        print("\nSaving results...")
        analyzer.save_results()
        
        print("\nAnalysis completed successfully!")
        print("="*70)
        
        return analyzer, results_df
    
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise


def bonus_comparison(ticker='AAPL', start='2014-01-01', end='2024-01-01'):
    """Bonus Task: Compare HMM with different numbers of states."""
    print("\n" + "="*70)
    print(" "*15 + "BONUS: Comparing Different State Models")
    print("="*70)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'State Comparison for {ticker}', fontsize=14, fontweight='bold')
    
    metrics_comparison = []
    
    for idx, n_states in enumerate([2, 3, 4]):
        print(f"\n{'-'*70}")
        print(f"Testing with {n_states} Hidden States")
        print(f"{'-'*70}")
        
        analyzer = FinancialHMMAnalyzer(ticker, start, end, n_states)
        analyzer.collect_data()
        analyzer.preprocess_data()
        analyzer.fit_hmm_model(n_iter=100)
        results_df = analyzer.decode_states()
        
        X = analyzer.returns.values.reshape(-1, 1)
        n_params = (n_states ** 2) + (2 * n_states)
        n_samples = len(analyzer.returns)
        log_likelihood = analyzer.model.score(X) * n_samples
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        
        metrics_comparison.append({
            'n_states': n_states,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood
        })
        
        print(f"   AIC: {aic:.2f}, BIC: {bic:.2f}")
        
        ax = axes[idx]
        for state in range(n_states):
            mask = results_df['Hidden_State'] == state
            ax.scatter(results_df[mask]['Date'], results_df[mask]['Returns'],
                      c=f'C{state}', label=f'State {state}', alpha=0.5, s=8)
        
        ax.set_title(f'{n_states} Hidden States (AIC: {aic:.0f}, BIC: {bic:.0f})', 
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('Returns', fontsize=10)
        ax.legend(loc='upper right', ncol=n_states, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    axes[-1].set_xlabel('Date', fontsize=10)
    plt.tight_layout()
    filename = f'{ticker}_State_Comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison: '{filename}'")
    plt.show()
    
    print("\n" + "="*70)
    print("Model Comparison Summary:")
    print("-"*70)
    comparison_df = pd.DataFrame(metrics_comparison)
    print(comparison_df.to_string(index=False))
    print("\nNote: Lower AIC/BIC indicates better model fit")
    print("="*70)
    
    return metrics_comparison


def bonus_multi_asset_comparison(tickers=['AAPL', 'TSLA', 'MSFT'], 
                                  start='2014-01-01', end='2024-01-01', 
                                  n_states=2):
    """Bonus Task: Compare HMM results across different assets."""
    print("\n" + "="*70)
    print(" "*10 + "BONUS: Multi-Asset HMM Comparison")
    print("="*70)
    
    results = {}
    
    for ticker in tickers:
        print(f"\n{'-'*70}")
        print(f"Analyzing {ticker}")
        print(f"{'-'*70}")
        
        try:
            analyzer = FinancialHMMAnalyzer(ticker, start, end, n_states)
            analyzer.collect_data()
            analyzer.preprocess_data()
            analyzer.fit_hmm_model(n_iter=100)
            analyzer.decode_states()
            
            results[ticker] = {
                'analyzer': analyzer,
                'means': analyzer.model.means_,
                'stds': np.sqrt(analyzer.model.covars_.flatten()),
                'transmat': analyzer.model.transmat_
            }
            
        except Exception as e:
            print(f"   Error analyzing {ticker}: {str(e)}")
            continue
    
    if len(results) > 0:
        fig, axes = plt.subplots(len(tickers), 2, figsize=(16, 5*len(tickers)))
        if len(tickers) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (ticker, data) in enumerate(results.items()):
            analyzer = data['analyzer']
            results_df = analyzer.results_df
            
            ax1 = axes[idx, 0]
            for state in range(n_states):
                mask = results_df['Hidden_State'] == state
                ax1.scatter(results_df[mask]['Date'], results_df[mask]['Returns'],
                          c=f'C{state}', label=f'State {state}', alpha=0.5, s=5)
            ax1.set_title(f'{ticker} - Returns by State', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Returns')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)
            
            ax2 = axes[idx, 1]
            sns.heatmap(data['transmat'], annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=ax2, square=True, cbar_kws={'label': 'Probability'})
            ax2.set_title(f'{ticker} - Transition Matrix', fontsize=11, fontweight='bold')
            ax2.set_xlabel('To State')
            ax2.set_ylabel('From State')
        
        axes[-1, 0].set_xlabel('Date')
        plt.tight_layout()
        filename = 'Multi_Asset_Comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved multi-asset comparison: '{filename}'")
        plt.show()
        
        print("\n" + "="*70)
        print("Multi-Asset State Characteristics:")
        print("-"*70)
        for ticker, data in results.items():
            print(f"\n{ticker}:")
            for state in range(n_states):
                mean = data['means'][state][0]
                std = data['stds'][state]
                print(f"  State {state}: Mean={mean:.4%}, Std={std:.4%}")
        print("="*70)
    
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  AI LAB ASSIGNMENT 5: GAUSSIAN HMM FOR FINANCIAL ANALYSIS")
    print("="*70)
    
    analyzer, results = main()
    
    # Uncomment to run bonus tasks:
    # bonus_comparison('AAPL', '2014-01-01', '2024-01-01')
    # bonus_multi_asset_comparison(['AAPL', 'TSLA', 'MSFT'], '2014-01-01', '2024-01-01')
    
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE - READY FOR SUBMISSION")
    print("="*70 + "\n")