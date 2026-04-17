import pandas as pd
import numpy as np
import sklearn.mixture as mix
import matplotlib.pyplot as plt
import yfinance as yf
import argparse

# Machine Learning
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# Create the class
class StrategyManager():

    # Initialize the class
    def __init__(self, symbol, start_date, end_date):
        self.df = self._extract_data(symbol, start_date, end_date)
        self.sharpe = 0

    # Extract data
    def _extract_data(self, symbol, start_date, end_date):
        import yfinance as yf
        data = yf.download(symbol, start=start_date, end=end_date)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data = self._structure_df(data)
        return data

    def extract_data_for_hmm(self, symbol, start_date, end_date):
        data = yf.download(symbol, start=start_date, end=end_date)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data = self._structure_df(data)
        return data
    
    def prepare_hmm_features(self, df):
        # Not used
        df["Returns"] = df["Close"].pct_change()
        df["Range"] = df["High"] / df["Low"] - 1
        log_close = np.log(df["Close"])
        r5 = log_close.diff(5)  # r5 = log(P_t / P_{t-5})
        r20 = log_close.diff(20)  # r20 = log(P_t / P_{t-20})
        df["r5"] = r5
        df["r20"] = r20

        df.dropna(inplace=True)
        return df

    # Calculates general period returns and volatility
    def _structure_df(self, df):
        df["Returns"] = df["Close"].pct_change()
        df["Range"] = df["High"] / df["Low"] - 1
        df["Bench_C_Rets"], sharpe = self._calculate_returns(df, True)

        log_close = np.log(df["Close"])
        r5 = log_close.diff(5)  # r5 = log(P_t / P_{t-5})
        r20 = log_close.diff(20)  # r20 = log(P_t / P_{t-20})
        df["r5"] = r5
        df["r20"] = r20
        self.sharpe = sharpe
        df.dropna(inplace=True)
        return df

    # Adjusts the signal to represent our strategy
    def _set_multiplier(self, direction):
        if direction == "long":
            pos_multiplier = 1
            neg_multiplier = 0
        elif direction == "long_short":
            pos_multiplier = 1
            neg_multiplier = -1
        else:
            pos_multiplier = 0
            neg_multiplier = -1
        return pos_multiplier, neg_multiplier

    # Calculates returns for equity curve
    def _calculate_returns(self, df, is_benchmark):
        
        # Calculate multiplier
        if not is_benchmark:
            multiplier_1 = df["Signal"].shift(1)
            multiplier_2 = 1 if "PSignal" not in df.columns else df["PSignal"].shift(1)
            
            # Assume open price on following day to avoid lookahead bias for close calculation
            log_rets = np.log(df["Open"].shift(-1) / df["Open"]) * multiplier_1 * multiplier_2
        else:            
            log_rets = np.log(df["Close"] / df["Close"].shift(1))
        
        # Calculate Sharpe Ratio
        sharpe_ratio = self.sharpe_ratio(log_rets)
        
        # Calculate Cumulative Returns
        c_log_rets = log_rets.cumsum()
        c_log_rets_exp = np.exp(c_log_rets) - 1
        
        # Return result and Sharpe ratio
        return c_log_rets_exp, sharpe_ratio
    
    def sharpe_ratio(self, return_series):
        N = 255 # Trading days in the year (change to 365 for crypto)
        rf = 0.005 # Half a percent risk free rare
        mean = return_series.mean() * N -rf
        sigma = return_series.std() * np.sqrt(N)
        sharpe = round(mean / sigma, 3)
        return sharpe

    # Plot price coloured by HMM hidden state labels
    def plot_hmm_labels(self, hidden_states, start_date=None, end_date=None, n_components=None):
        df = self.df["Close"]
        if start_date or end_date:
            df = df.loc[start_date:end_date]

        prices = df.to_numpy(dtype=float).reshape(-1)
        hs = np.asarray(hidden_states).reshape(-1).astype(int)

        if len(prices) != len(hs):
            raise ValueError(f"Length mismatch: prices={len(prices)}, hidden_states={len(hs)}")

        n_states = n_components if n_components is not None else len(np.unique(hs))
        x = np.arange(len(prices))

        plt.figure(figsize=(18, 10))
        plt.plot(x, prices, color="lightgray", linewidth=1, label="Price")

        for state in range(n_states):
            mask = hs == state
            plt.scatter(x[mask], prices[mask], s=10, label=f"State {state}")

        plt.legend()
        plt.title("Price by Hidden State")
        plt.tight_layout()
        plt.show()

    # Determine favourable states ranked by a combined score of mean, volatility and up_ratio
    def get_favourable_states(self, hidden_states, returns=None, n_favourable=2,
                              mean_weight=1.0, vol_weight=1.0, up_ratio_weight=1.0,
                              min_mean=None, max_volatility=None, min_up_ratio=None,
                              verbose=True):
        hs = np.asarray(hidden_states).reshape(-1).astype(int)
        rets = (
            np.asarray(returns, dtype=float).reshape(-1)
            if returns is not None
            else self.df["Returns"].values[-len(hs):].astype(float)
        )

        if len(hs) != len(rets):
            raise ValueError(f"Length mismatch: hidden_states={len(hs)}, returns={len(rets)}")

        unique_states = np.unique(hs)
        stats = []
        for state in unique_states:
            mask = hs == state
            state_rets = rets[mask]
            mean_ret   = state_rets.mean()
            volatility = state_rets.std()
            up_days    = int((state_rets > 0).sum())
            down_days  = int((state_rets < 0).sum())
            up_ratio   = up_days / len(state_rets) if len(state_rets) > 0 else 0.0
            stats.append({
                "state":      state,
                "mean":       mean_ret,
                "volatility": volatility,
                "up_days":    up_days,
                "down_days":  down_days,
                "up_ratio":   up_ratio,
                "count":      int(mask.sum()),
            })

        # Apply hard filters before scoring
        filtered = [
            s for s in stats
            if (min_mean       is None or s["mean"]       >= min_mean)
            and (max_volatility is None or s["volatility"] <= max_volatility)
            and (min_up_ratio   is None or s["up_ratio"]   >= min_up_ratio)
        ]

        if not filtered:
            if verbose:
                print("No states passed the hard filters. Returning empty list.")
            return []

        # Normalise each metric across the filtered set to [0, 1]
        means  = np.array([s["mean"]       for s in filtered])
        vols   = np.array([s["volatility"] for s in filtered])
        ratios = np.array([s["up_ratio"]   for s in filtered])

        def _norm(arr):
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / rng if rng > 0 else np.ones_like(arr) * 0.5

        mean_scores     = _norm(means)               # higher mean  → higher score
        vol_scores      = 1.0 - _norm(vols)          # lower vol    → higher score
        up_ratio_scores = _norm(ratios)              # higher ratio → higher score

        combined = (
            mean_weight     * mean_scores
            + vol_weight    * vol_scores
            + up_ratio_weight * up_ratio_scores
        )

        for i, s in enumerate(filtered):
            s["score"] = round(float(combined[i]), 4)
            s["mean"]       = round(s["mean"],       6)
            s["volatility"] = round(s["volatility"], 6)
            s["up_ratio"]   = round(s["up_ratio"],   3)

        # Sort by combined score descending
        filtered.sort(key=lambda s: s["score"], reverse=True)

        if verbose:
            print(f"{'State':>6} {'Mean':>10} {'Volatility':>12} {'Up Days':>9} {'Down Days':>10} {'Up Ratio':>10} {'Score':>8} {'Count':>7}")
            print("-" * 80)
            for s in filtered:
                print(f"{s['state']:>6} {s['mean']:>10.6f} {s['volatility']:>12.6f} "
                      f"{s['up_days']:>9} {s['down_days']:>10} {s['up_ratio']:>10.3f} "
                      f"{s['score']:>8.4f} {s['count']:>7}")

        top_n = n_favourable if n_favourable <= len(filtered) else len(filtered)
        favourable = [s["state"] for s in filtered[:top_n]]

        if verbose:
            print(f"\nTop {top_n} favourable states: {favourable}")

        return favourable

    # Replace Dataframe
    def change_df(self, new_df, drop_cols=[]):
        new_df = new_df.drop(columns=drop_cols)
        self.df = new_df


    def find_best_random_state(
        self,
        X_train_scaled: np.ndarray,
        n_random_states: int = 10,
        n_iter: int = 10,
        covariance_type: str = "diag",
        tol: float = 1e-4,
        n_components: int = 4
    ) -> int:
        """
        Find best random state by trying multiple initializations.
        
        Trains models with different random states and returns the one
        with highest log-likelihood score.
        
        Args:
            X_train_scaled: Scaled training features
            n_random_states: Number of random states to try
            n_iter: Number of EM iterations per trial
            
        Returns:
            Best random state (integer)
        """
        best_score = -np.inf
        best_rs = 1
        
        for rs in range(n_random_states):
            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=rs,
                verbose=False
            )
            
            try:
                model.fit(X_train_scaled)
                score = model.score(X_train_scaled)
                
                if score > best_score:
                    best_score = score
                    best_rs = rs
            except Exception:
                # Skip failed fits
                continue
        
        return best_rs
    

    # Moving average crossover strategy
    def backtest_ma_crossover(self, period_1, period_2, direction, drop_cols=[], calculate_benchmark=True):
        
        # Set df
        df = self.df
        
        # Get multipliers
        pos_multiplier, neg_multiplier = self._set_multiplier(direction)
            
        # Calculate Moving Averages
        if f"MA_{period_1}" or f"MA_{period_2}" not in df.columns:
            df[f"MA_{period_1}"] = df["Close"].rolling(window=period_1).mean()
            df[f"MA_{period_2}"] = df["Close"].rolling(window=period_2).mean()
            df.dropna(inplace=True)
        
        # Calculate Benchmark Returns
        if calculate_benchmark:
            df["Bench_C_Rets"], sharpe_ratio_bench = self._calculate_returns(df, True)
        else:
            sharpe_ratio_bench = None
        
        # Calculate Signal
        df.loc[df[f"MA_{period_1}"] > df[f"MA_{period_2}"], "Signal"] = pos_multiplier
        df.loc[df[f"MA_{period_1}"] <= df[f"MA_{period_2}"], "Signal"] = neg_multiplier
        
        # Calculate Strategy Returns
        out, sharpe_ratio_strat = self._calculate_returns(df, False)
        df["Strat_C_Rets"] = out
        
        # Get values for output
        if calculate_benchmark:
            bench_rets = df["Bench_C_Rets"].values.astype(float)
        else:
            bench_rets = None
        strat_rets = df["Strat_C_Rets"].values.astype(float)

        print("Sense check: ", round(df["Close"].values[-1] / df["Close"].values[0] - 1, 3), round(bench_rets[-1], 3) if bench_rets is not None else None)
        
        # Remove irrelevant features
        if len(drop_cols) > 0:
            df = df.drop(columns=drop_cols)
        
        # Ensure Latest DF matches
        df = df.dropna()
        self.df = df
        
        # Return df
        return df, sharpe_ratio_bench, sharpe_ratio_strat

    # Exponential moving average crossover strategy
    def backtest_ema(self, period_1, period_2, direction, drop_cols=[], calculate_benchmark=True):

        # Set df
        df = self.df

        # Get multipliers
        pos_multiplier, neg_multiplier = self._set_multiplier(direction)

        # Calculate Exponential Moving Averages
        if f"EMA_{period_1}" not in df.columns or f"EMA_{period_2}" not in df.columns:
            df[f"EMA_{period_1}"] = df["Close"].ewm(span=period_1, adjust=False).mean()
            df[f"EMA_{period_2}"] = df["Close"].ewm(span=period_2, adjust=False).mean()
            df.dropna(inplace=True)

        # Calculate Benchmark Returns
        if calculate_benchmark:
            df["Bench_C_Rets"], sharpe_ratio_bench = self._calculate_returns(df, True)
        else:
            sharpe_ratio_bench = None

        # Calculate Signal
        df.loc[df[f"EMA_{period_1}"] > df[f"EMA_{period_2}"], "Signal"] = pos_multiplier
        df.loc[df[f"EMA_{period_1}"] <= df[f"EMA_{period_2}"], "Signal"] = neg_multiplier

        # Calculate Strategy Returns
        out, sharpe_ratio_strat = self._calculate_returns(df, False)
        df["Strat_C_Rets"] = out

        # Get values for output
        if calculate_benchmark:
            bench_rets = df["Bench_C_Rets"].values.astype(float)
        else:
            bench_rets = None
        strat_rets = df["Strat_C_Rets"].values.astype(float)

        print("Sense check: ", round(df["Close"].values[-1] / df["Close"].values[0] - 1, 3), round(bench_rets[-1], 3) if bench_rets is not None else None)

        # Remove irrelevant features
        if len(drop_cols) > 0:
            df = df.drop(columns=drop_cols)

        # Ensure Latest DF matches
        df = df.dropna()
        self.df = df

        # Return df
        return df, sharpe_ratio_bench, sharpe_ratio_strat

    def macd_strategy(self, short_period=12, long_period=26, signal_period=9, direction="long", drop_cols=[], calculate_benchmark=True):

        # Set df
        df = self.df

        # Get multipliers
        pos_multiplier, neg_multiplier = self._set_multiplier(direction)

        # Calculate MACD and Signal Line
        # Calculate Exponential Moving Averages
        if f"EMA_{short_period}" not in df.columns or f"EMA_{long_period}" not in df.columns:
            df[f"EMA_{short_period}"] = df["Close"].ewm(span=short_period, adjust=False).mean()
            df[f"EMA_{long_period}"] = df["Close"].ewm(span=long_period, adjust=False).mean()
            df.dropna(inplace=True)

        df["MACD"] = df[f"EMA_{short_period}"] - df[f"EMA_{long_period}"]
        df["Signal_Line"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
        df.dropna(inplace=True)

        # Calculate Benchmark Returns
        if calculate_benchmark:
            df["Bench_C_Rets"], sharpe_ratio_bench = self._calculate_returns(df, True)
        else:
            sharpe_ratio_bench = None

        # Calculate Signal
        df.loc[df["MACD"] > df["Signal_Line"], "Signal"] = pos_multiplier
        df.loc[df["MACD"] <= df["Signal_Line"], "Signal"] = neg_multiplier

        # Calculate Strategy Returns
        out, sharpe_ratio_strat = self._calculate_returns(df, False)
        df["Strat_C_Rets"] = out

        # Get values for output
        if calculate_benchmark:
            bench_rets = df["Bench_C_Rets"].values.astype(float)
        else:
            bench_rets = None
        strat_rets = df["Strat_C_Rets"].values.astype(float)

        print("Sense check: ", round(df["Close"].values[-1] / df["Close"].values[0] - 1, 3), round(bench_rets[-1], 3) if bench_rets is not None else None)

        # Remove irrelevant features
        if len(drop_cols) > 0:
            df = df.drop(columns=drop_cols)

        # Ensure Latest DF matches
        df = df.dropna()
        self.df = df

        # Return df
        return df, sharpe_ratio_bench, sharpe_ratio_strat

    # Moving average crossover strategy
    def backtest_ma_crossover(self, period_1, period_2, direction, drop_cols=[], calculate_benchmark=True):
        
        # Set df
        df = self.df
        
        # Get multipliers
        pos_multiplier, neg_multiplier = self._set_multiplier(direction)
            
        # Calculate Moving Averages
        if f"MA_{period_1}" or f"MA_{period_2}" not in df.columns:
            df[f"MA_{period_1}"] = df["Close"].rolling(window=period_1).mean()
            df[f"MA_{period_2}"] = df["Close"].rolling(window=period_2).mean()
            df.dropna(inplace=True)
        
        # Calculate Benchmark Returns
        if calculate_benchmark:
            df["Bench_C_Rets"], sharpe_ratio_bench = self._calculate_returns(df, True)
        else:
            sharpe_ratio_bench = None
        
        # Calculate Signal
        df.loc[df[f"MA_{period_1}"] > df[f"MA_{period_2}"], "Signal"] = pos_multiplier
        df.loc[df[f"MA_{period_1}"] <= df[f"MA_{period_2}"], "Signal"] = neg_multiplier
        
        # Calculate Strategy Returns
        out, sharpe_ratio_strat = self._calculate_returns(df, False)
        df["Strat_C_Rets"] = out
        
        # Get values for output
        if calculate_benchmark:
            bench_rets = df["Bench_C_Rets"].values.astype(float)
        else:
            bench_rets = None
        strat_rets = df["Strat_C_Rets"].values.astype(float)

        print("Sense check: ", round(df["Close"].values[-1] / df["Close"].values[0] - 1, 3), round(bench_rets[-1], 3) if bench_rets is not None else None)
        
        # Remove irrelevant features
        if len(drop_cols) > 0:
            df = df.drop(columns=drop_cols)
        
        # Ensure Latest DF matches
        df = df.dropna()
        self.df = df
        
        # Return df
        return df, sharpe_ratio_bench, sharpe_ratio_strat
    
    def stateless_backtest_ma_crossover(self, period_1, period_2, direction, drop_cols=[], calculate_benchmark=True):
        
        # Set df
        df = self.df.copy()
        
        # Get multipliers
        pos_multiplier, neg_multiplier = self._set_multiplier(direction)
            
        # Calculate Moving Averages
        if f"MA_{period_1}" or f"MA_{period_2}" not in df.columns:
            df[f"MA_{period_1}"] = df["Close"].rolling(window=period_1).mean()
            df[f"MA_{period_2}"] = df["Close"].rolling(window=period_2).mean()
            df.dropna(inplace=True)
        
        # Calculate Benchmark Returns
        if calculate_benchmark:
            df["Bench_C_Rets"], sharpe_ratio_bench = self._calculate_returns(df, True)
        else:
            sharpe_ratio_bench = None
        
        # Calculate Signal
        df.loc[df[f"MA_{period_1}"] > df[f"MA_{period_2}"], "Signal"] = pos_multiplier
        df.loc[df[f"MA_{period_1}"] <= df[f"MA_{period_2}"], "Signal"] = neg_multiplier
        
        # Calculate Strategy Returns
        out, sharpe_ratio_strat = self._calculate_returns(df, False)
        df["Strat_C_Rets"] = out
        
        # Get values for output
        if calculate_benchmark:
            bench_rets = df["Bench_C_Rets"].values.astype(float)
        else:
            bench_rets = None
        strat_rets = df["Strat_C_Rets"].values.astype(float)

        # Not needed here
        #print("Sense check: ", round(df["Close"].values[-1] / df["Close"].values[0] - 1, 3), round(bench_rets[-1], 3) if bench_rets is not None else None)
        
        # Remove irrelevant features
        if len(drop_cols) > 0:
            df = df.drop(columns=drop_cols)
        
        # Ensure Latest DF matches
        df = df.dropna()
        
        # Return df
        return df, sharpe_ratio_bench, sharpe_ratio_strat

def compute_bic(hmm_model: GaussianHMM, X: np.ndarray) -> tuple:
        """
        Compute Bayesian Information Criterion (BIC) for HMM.
        
        BIC = -2*logL + k*log(T)
        where k is the number of parameters and T is the number of observations.
        
        Lower BIC indicates better model (balances fit vs complexity).
        
        Args:
            hmm_model: Fitted GaussianHMM model
            X: Feature data (T, d)
            
        Returns:
            Tuple of (bic, logL) where:
            - bic: Bayesian Information Criterion
            - logL: Log-likelihood
        """
        K = hmm_model.n_components
        T, d = X.shape
        logL = hmm_model.score(X)
        
        # Count parameters:
        # - Start probabilities: K-1 (sum to 1 constraint)
        # - Transition matrix: K*(K-1) (each row sums to 1)
        # - Emission parameters per state:
        #   * Mean vector: d parameters
        #   * Diagonal covariance: d parameters (for 'diag' type)
        k_params = (K - 1) + (K * (K - 1)) + K * (2 * d)
        
        bic = -2.0 * logL + k_params * np.log(T)
        
        return bic, logL

def find_best_n_components(
        X_train_scaled: np.ndarray,
        state_range: range = range(2, 10),
        n_random_seeds: int = 5,
        covariance_type: str = "diag",
        n_iter: int = 1000,
        tol: float = 1e-3,
        random_state_base: int = 42,
        verbose: bool = True
    ) -> dict:
        """
        Find optimal number of HMM states using BIC model selection.
        """
        if X_train_scaled.ndim != 2:
            raise ValueError(f"X_train_scaled must be 2D, got shape {X_train_scaled.shape}")
        
        if X_train_scaled.size == 0:
            raise ValueError("X_train_scaled cannot be empty")
        
        results = []
        best = {"bic": np.inf, "model": None, "K": None, "logL": None}
        
        if verbose:
            print(f"\nSearching for optimal number of states (K) using BIC...")
            print(f"  State range: {list(state_range)}")
            print(f"  Random seeds per K: {n_random_seeds}")
            print(f"  Covariance type: {covariance_type}")
        
        for K in state_range:
            best_for_K = {"bic": np.inf, "model": None, "logL": None}
            
            for seed_idx in range(n_random_seeds):
                seed = random_state_base + seed_idx
                
                hmm = GaussianHMM(
                    n_components=K,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    tol=tol,
                    random_state=seed,
                    verbose=False
                )
                
                try:
                    hmm.fit(X_train_scaled)
                    bic, logL = compute_bic(hmm, X_train_scaled)
                    
                    if bic < best_for_K["bic"]:
                        best_for_K = {"bic": bic, "model": hmm, "logL": logL}
                
                except Exception as e:
                    if verbose:
                        print(f"    WARNING: K={K}, seed={seed} failed: {str(e)}")
                    continue
            
            # Record best result for this K
            if best_for_K["model"] is not None:
                results.append((K, best_for_K["bic"], best_for_K["logL"]))
                
                if best_for_K["bic"] < best["bic"]:
                    best = {
                        "bic": best_for_K["bic"],
                        "model": best_for_K["model"],
                        "K": K,
                        "logL": best_for_K["logL"]
                    }
                
                if verbose:
                    print(f"  K={K}: BIC={best_for_K['bic']:,.1f}  logL={best_for_K['logL']:,.1f}")
        
        if best["model"] is None:
            raise RuntimeError("Failed to fit any models. Check data quality.")
        
        if verbose:
            print(f"\n✓ Selected K={best['K']} with BIC={best['bic']:,.1f}")
        
        return {
            'best_n_states': best['K'],
            'best_model': best['model'],
            'best_bic': best['bic'],
            'best_logL': best['logL'],
            'all_results': results
        }

def count_trades(signals) -> int:
    """
    Count the number of round-trip trades in a signal series.
    A trade is one open + one close, i.e. each 0→1 transition.

    Examples:
        [0,1,0]     -> 1 trade
        [0,1,1,0]   -> 1 trade
        [0,1,0,1,0] -> 2 trades
    """
    s = np.asarray(signals, dtype=int).reshape(-1)
    # A new trade starts wherever the signal flips from 0 to 1
    return int(np.sum((s[1:] == 1) & (s[:-1] == 0)))


def regime_gate(p: np.ndarray, gate_type: str = "threshold",
               tau: float = 0.5, k: float = 10.0) -> np.ndarray:
    """
    Map posterior probability p ∈ [0,1] → binary integer 0 or 1.

    Uses the HMM's posterior confidence that the market is in a favourable
    regime to decide whether to allow the strategy signal through:

        PSignal_t = gate(p_t)   ∈ {0, 1}

    State is considered favourable (1) when the gate criterion is met,
    unfavourable (0) otherwise.

    Args:
        p        : Array of posterior probabilities, shape (T,)
        gate_type: Decision rule used before binarisation:
                   'threshold' – 1 if p ≥ τ, else 0                [default]
                   'linear'    – 1 if p ≥ τ, else 0  (same as threshold;
                                 kept for API compatibility)
                   'logistic'  – 1 if σ(k·(p−τ)) ≥ 0.5, else 0
                                 (equivalent to p ≥ τ, smooth boundary)
        tau      : Threshold τ — minimum confidence to mark state as favourable
        k        : Steepness of logistic curve (used only for 'logistic')

    Returns:
        Integer array of 0s and 1s, same shape as p.
    """
    p = np.asarray(p, dtype=float)
    if gate_type in ("linear", "threshold"):
        if not (0.0 <= tau < 1.0):
            raise ValueError(f"tau must be in [0, 1), got {tau}")
        return (p >= tau).astype(int)
    elif gate_type == "logistic":
        # σ(k·(p−τ)) ≥ 0.5  ↔  p ≥ τ
        g = 1.0 / (1.0 + np.exp(-k * (p - tau)))
        return (g >= 0.5).astype(int)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type!r}. "
                         "Choose from 'linear', 'threshold', 'logistic'.")


def perform_brute_force_short_long_test(symbol, start_date, end_date, period_1_range, period_2_range):
    results = []
    strat_mgr = StrategyManager(symbol, start_date, end_date)

    best_returns = -np.inf
    short_mv = 0
    long_mv = 0

    for p1 in period_1_range:
        for p2 in period_2_range:
            if p1 >= p2:
                continue  # Skip invalid combinations
            
            strat_df, sharpe_bench, sharpe_strat = strat_mgr.stateless_backtest_ma_crossover(p1, p2, "long", calculate_benchmark=False)
            

            if(strat_df["Strat_C_Rets"].values[-1] > best_returns):
                best_returns = strat_df["Strat_C_Rets"].values[-1]
                short_mv = p1
                long_mv = p2
                print(f"New best returns: {round(best_returns * 100, 2)}% with periods {p1} and {p2}")
                print(f"Number of trades: {count_trades(strat_df['Signal'].values)}")

    return best_returns, short_mv, long_mv


def run_hmm_strategy_with_posterior_prob(
        X_train,
        X_test,
        hmm_df,
        strat_df,
        strat_mgr,
        n_components=4,
        covariance_type="diag",
        n_favourable=2,
        random_state=42,
        threshold=0.5,
        gate_type="threshold"):
    df_strat_mgr_test = strat_df.copy()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    hmm_model = GaussianHMM(
        n_components=n_components, 
        covariance_type=covariance_type, 
        n_iter=1000, 
        random_state=random_state).fit(X_train_scaled)
    #hmm_model = mix.GaussianMixture(n_components=n_components, covariance_type=covariance_type, n_init=1000, random_state=best_random_state).fit(X_train_scaled)
    print(f"Model Score: {hmm_model.score(X_train_scaled)}")

    hidden_states_preds = hmm_model.predict(X_train_scaled)
    len(hidden_states_preds)
    test_start_date = X_test.index[0]
    #strat_mgr.plot_hmm_labels(hidden_states_preds, start_date=test_start_date, end_date=end_date, n_components=n_components)

    # Set Favourable States - derived from per-state return statistics
    favourable_states = strat_mgr.get_favourable_states(hidden_states_preds, returns=hmm_df['Returns'], n_favourable=n_favourable)

    # --- Soft regime gate via posterior probabilities ---
    # p_t = P(z_t ∈ F | data) = sum of posterior proba over favourable states
    state_proba = hmm_model.predict_proba(X_test_scaled)
    p_favourable = state_proba[:, favourable_states].sum(axis=1)  # shape (T,)

    # g(p_t): map confidence → exposure scalar ∈ [0, 1]
    state_signals = regime_gate(p_favourable, gate_type=gate_type, tau=threshold)

    print(f"Gate type: {gate_type},  threshold τ={threshold}")
    print("Favourable-state probabilities (first 10):", np.round(p_favourable[:10], 3))
    print("Gate outputs / PSignal        (first 10):", state_signals[:10].tolist())
    print("Length of States: ", len(state_signals))

    # Update strategy manager with new PSignal column for backtesting
    df_strat_mgr_test = df_strat_mgr_test.tail(len(X_test))
    df_strat_mgr_test["PSignal"] = np.round(state_signals)
    strat_mgr.change_df(df_strat_mgr_test)
    strat_mgr.df.head()

    return strat_mgr

def run_strategy(symbol, 
                start_hmm_date,
                end_hmm_date,
                test_start_date,
                test_end_date,
                short_mv=12,
                long_mv=26,
                n_components=4,
                covariance_type="diag",
                n_favourable=2,
                random_state=42,
                threshold=0.91,
                gate_type="threshold",
                show_plots=False,
                calc_BIC=False,
                find_best_random_state=True):
    strat_mgr = StrategyManager(symbol, test_start_date, test_end_date)

    # Check MACD Strategy performance
    signal_period = 35
    strat_df, sharpe_b, sharpe_base_s = strat_mgr.macd_strategy(short_mv, long_mv, signal_period=signal_period, direction="long", drop_cols=["High", "Low", "Volume"])

    # Review equity curve and metrics
    print("Sharpe Ratio Base Strategy Benchmark: ", sharpe_b)
    print("Sharpe Ratio Base Strategy: ", sharpe_base_s)
    total_returns_strategy_base = strat_df["Strat_C_Rets"].values[-1]
    print("Total returns base strategy: ", total_returns_strategy_base)
    print(f"Number of trades (base MACD strategy): {count_trades(strat_df['Signal'].values)}")

    if show_plots:
        fig = plt.figure(figsize = (18,10))
        plt.plot(strat_df["Bench_C_Rets"])
        plt.plot(strat_df["Strat_C_Rets"])
        plt.show()

    hmm_df = strat_mgr.extract_data_for_hmm(symbol=symbol, start_date=start_hmm_date, end_date=end_hmm_date)
    columns = ["Returns", "Range", "r5", "r20"]
    X_train= hmm_df[columns] # Train
    X_test = strat_df[columns] # Test

    # Standardize features - fit scaler on train, transform both train and test
    scaler = StandardScaler()

    # Fit Model
    X_train_scaled = scaler.fit_transform(X_train.values)

    print(f"Fitting HMM model... {len(X_train)} samples")

    best_random_state = random_state
    if find_best_random_state:
        print("Finding best random state for HMM initialization...")
        best_random_state = strat_mgr.find_best_random_state(
            X_train_scaled, 
            n_random_states=50, 
            n_iter=100, 
            covariance_type=covariance_type, 
            n_components=n_components)

        # Find best number of states for HMM model
    if calc_BIC:
        result = find_best_n_components(X_train_scaled, state_range=range(2, 14), n_random_seeds=20, covariance_type=covariance_type, n_iter=100, tol=1e-3, random_state_base=best_random_state)
        n_components = result['best_n_states']
        print(f"Best number of states: {n_components}")
        print(f"Best BIC: {result['best_bic']}")

    run_hmm_strategy_with_posterior_prob(
        X_train,
        X_test,
        hmm_df,
        strat_df,
        strat_mgr,
        n_components=n_components,
        covariance_type=covariance_type,
        n_favourable=n_favourable,
        random_state=best_random_state,
        threshold=threshold,
        gate_type=gate_type)

    # Replace Strategy Dataframe
    strat_df_2, sharpe_b_2, sharpe_s_2 = strat_mgr.macd_strategy(short_mv, long_mv, signal_period=signal_period, direction="long")
    strat_df_2

    returns_combined_str = strat_df_2["Strat_C_Rets"].values[-1]
    return_improvement = returns_combined_str - total_returns_strategy_base
    sharpe_improvement = sharpe_s_2 - sharpe_base_s

    # Review equity curve
    print("Sharpe Ratio Benchmark: ", sharpe_b_2)
    print("Sharpe Ratio Regime Strategy with MACD: ", sharpe_s_2)
    print("--- ---")
    print(f"Returns Benchmark: {round(strat_df_2['Bench_C_Rets'].values[-1] * 100, 2)}%")
    print(f"Returns Regime Strategy with MACD: {round(strat_df_2['Strat_C_Rets'].values[-1] * 100, 2)}%")
    print(f"Returns Baseline Strategy MACD: {round(total_returns_strategy_base * 100, 2)}%")
    print(f"Returns improvement over baseline: {round(return_improvement * 100, 2)}%")
    print(f"Sharpe improvement over baseline: {round(sharpe_improvement * 100, 2)}%")
    # Effective signal: MACD signal gated by binary regime filter (0 or 1)
    effective_signal = strat_df_2["Signal"].values * strat_df_2["PSignal"].values
    print(f"Number of trades (regime + MACD strategy): {count_trades(effective_signal)}")

    if show_plots:
        fig = plt.figure(figsize = (18, 10))
        plt.plot(strat_df_2["Bench_C_Rets"])
        plt.plot(strat_df_2["Strat_C_Rets"])
        plt.show()

    return {
        "return_improvement": return_improvement,
        "sharpe_improvement": sharpe_improvement,
        "returns_ma_with_regime": returns_combined_str,
        "sharpe_ma_with_regime": sharpe_s_2,
        "returns_ma": total_returns_strategy_base,
        "sharpe_ma": sharpe_base_s,
        "benchmark_buy_and_hold": strat_df_2["Bench_C_Rets"].values[-1]
    }

# Step 1: Set up parameters and periods
# Configure validation, HMM training, testing periods
# Parse command-line arguments
parser = argparse.ArgumentParser(description="HMM Regime Strategy")

# HMM parameters
parser.add_argument("--n_components", type=int, default=8, help="Number of HMM hidden states (default: 8)")
parser.add_argument("--n_favourable", type=int, default=6, help="Number of favourable states to use as buy signal (default: 6)")
parser.add_argument("--gate_type", type=str, default="threshold",
                    choices=["linear", "threshold", "logistic"],
                    help="Soft gate function: linear | threshold | logistic (default: threshold)")
parser.add_argument("--threshold", type=float, default=0.91,
                    help="Probability threshold τ for regime gate (default: 0.91)")

# Backtest parameters
parser.add_argument("--show_plots", type=bool, default=False, help="Whether to show plots (default: True)")
parser.add_argument("--test_start_date", type=str, default="2024-01-01", help="Start date for backtesting (default: 2024-01-01)")
parser.add_argument("--test_end_date", type=str, default="2026-02-25", help="End date for backtesting (default: 2026-02-25)")
parser.add_argument("--hmm_training_years", type=int, default=5, help="Number of years of data to use for HMM training (default: 5)")

# optimization parameters
parser.add_argument("--optimize", type=bool, default=False, help="Whether to perform hyperparameter optimization (default: False)")
parser.add_argument("--max_components", type=int, default=16, help="Number of random states to try when finding best random state (default: 16)")

args = parser.parse_args()

backtest_start_date = pd.Timestamp(args.test_start_date)
backtest_end_date = pd.Timestamp(args.test_end_date)
hmm_model_training_in_years = args.hmm_training_years

test_length_years = backtest_end_date.year - backtest_start_date.year
validation_start_date = backtest_end_date - pd.DateOffset(years=test_length_years) # Validation
validation_end_date = backtest_end_date - pd.DateOffset(days=30) # Validation

start_hmm_date = validation_end_date - pd.DateOffset(years=hmm_model_training_in_years) # HMM Training (5 years of data for training)
end_hmm_date = validation_end_date

symbol = "SPY"
short_mv = 9
long_mv = 30

print(f"Backtest period (TESTING): {backtest_start_date} to {backtest_end_date}")
print(f"Validation period (VALIDATION): {validation_start_date} to {validation_end_date}")
print(f"HMM training period (HMM TRAINING): {start_hmm_date} to {end_hmm_date}")
print(f"Initial MA periods: short={short_mv}, long={long_mv}")
print(f"Test length (years): {test_length_years}")
print(f"HMM training length (years): {hmm_model_training_in_years}")
print(f"Symbol: {symbol}")

# Step 2: Run strategy

print("\n--- Running strategy on VALIDATION period ---")
# Parameter search
n_components = args.n_components
n_favourable = args.n_favourable
threshold = args.threshold
best_delta_returns = -np.inf
best_delta_sharpe = -np.inf
if args.optimize:
    all_results = []   # ← collect here
    for components in range(1, args.max_components + 1):
        for n_fav in range(1, components):
            for threshold in [0.8, 0.85, 0.9, 0.91, 0.92, 0.95, 0.96, 0.97, 0.98, 0.99]:
                print(f"\nTesting with n_components={components}, n_favourable={n_fav}, threshold={threshold}")
                rr = run_strategy(
                    symbol=symbol,
                    start_hmm_date=start_hmm_date,
                    end_hmm_date=end_hmm_date,
                test_start_date=validation_start_date,
                test_end_date=validation_end_date,
                short_mv=short_mv,
                long_mv=long_mv,
                n_components=components,
                n_favourable=n_fav,
                random_state=42,
                show_plots=args.show_plots,
                calc_BIC=False,
                find_best_random_state=True,
                threshold=threshold,
                gate_type=args.gate_type
            )
                
            delta_returns = rr["return_improvement"]
            delta_sharpe = rr["sharpe_improvement"]
            ret = rr["returns_ma_with_regime"]
            sharpe = rr["sharpe_ma_with_regime"]

            if delta_returns > best_delta_returns:
                best_delta_returns = delta_returns
                best_delta_sharpe = delta_sharpe
                n_components = components
                n_favourable = n_fav
                print(f"------> Found improvement: {round(delta_returns * 100, 2)}%, Sharpe improvement: {round(delta_sharpe * 100, 2)}%")

                all_results.append({
                    "n_components": components,
                    "n_favourable": n_fav,
                    "threshold": threshold,
                    "returns_pct": round(ret * 100, 2),
                    "sharpe": sharpe,
                    "delta_returns_pct": round(delta_returns * 100, 2),
                    "delta_sharpe_pct": round(delta_sharpe * 100, 2),
                    "type": "validation",
                    "buy_and_hold": round(rr["benchmark_buy_and_hold"] * 100, 2),
                    "base_strategy_returns": round(rr["returns_ma"] * 100, 2),
                    "base_strategy_sharpe": round(rr["sharpe_ma"], 2)
                })
    
    pd.DataFrame(all_results).to_csv("./optimize_validation_results.csv", index=False)
    print("Optimization results saved.")
else:
    rr = run_strategy(
        symbol=symbol,
        start_hmm_date=start_hmm_date,
        end_hmm_date=end_hmm_date,
        test_start_date=validation_start_date,
        test_end_date=validation_end_date,
        short_mv=short_mv,
        long_mv=long_mv,
        n_components=args.n_components,
        n_favourable=args.n_favourable,
        random_state=42,
        show_plots=args.show_plots,
        calc_BIC=False,
        find_best_random_state=True,
        threshold=args.threshold,
        gate_type=args.gate_type
    )

    delta_returns = rr["return_improvement"]
    delta_sharpe = rr["sharpe_improvement"]
    ret = rr["returns_ma_with_regime"]
    sharpe = rr["sharpe_ma_with_regime"]

    print(f"Validation returns improvement over baseline: {round(delta_returns * 100, 2)}%")
    print(f"Validation Sharpe improvement over baseline: {round(delta_sharpe * 100, 2)}%")
    print(f"Validation returns with regime strategy: {round(ret * 100, 2)}%")
    print(f"Validation Sharpe with regime strategy: {round(sharpe, 2)}")

print("\n--- Running strategy on BACKTEST period ---")
print(f"Using parameters: n_components={n_components}, n_favourable={n_favourable}, threshold={threshold}")
test_results = []
rr = run_strategy(
    symbol=symbol,
    start_hmm_date=start_hmm_date,
    end_hmm_date=end_hmm_date,
    test_start_date=backtest_start_date,
    test_end_date=backtest_end_date,
    short_mv=short_mv,
    long_mv=long_mv,
    n_components=n_components,
    n_favourable=n_favourable,
    random_state=42,
    show_plots=args.show_plots,
    calc_BIC=False,
    find_best_random_state=True,
    threshold=threshold,
    gate_type=args.gate_type
)
test_results.append({
    "n_components": n_components,
    "n_favourable": n_favourable,
    "threshold": threshold,
    "returns_pct": round(rr["returns_ma_with_regime"] * 100, 2),
    "sharpe": round(rr["sharpe_ma_with_regime"], 2),
    "delta_returns_pct": round(rr["return_improvement"] * 100, 2),
    "delta_sharpe_pct": round(rr["sharpe_improvement"] * 100, 2),
    "type": "backtest",
    "buy_and_hold": round(rr["benchmark_buy_and_hold"] * 100, 2),
    "base_strategy_returns": round(rr["returns_ma"] * 100, 2),
    "base_strategy_sharpe": round(rr["sharpe_ma"], 2)
})
print("=======================================")
print(f"Final returns improvement over baseline: {round(rr['return_improvement'] * 100, 2)}%")
print(f"Final Sharpe improvement over baseline: {round(rr['sharpe_improvement'] * 100, 2)}%")
print("=======================================")
pd.DataFrame(test_results).to_csv("./optimize_backtest_results.csv", index=False)
print("Backtest results saved.")