import os
import csv
import numpy as np

# Define global constants for CSV structure
# These refer to the 4th and 5th columns (0-indexed) in the source files
TIME_COL = 3
SIG_COL = 4

def load_csv(path):

    """
    Parses oscilloscope CSV files to extract time-series data and sampling rates.
    
    Args:
        path (str): Path to the CSV file.
    Returns:
        tuple: (time array, signal array, sampling frequency) or None if invalid.
    """

    sample_interval = None
    data_start = None
    
    with open(path, "r") as f:
        rows = list(csv.reader(f))
        
    # Scan the file header for metadata and the start of the raw numeric data
    for i, row in enumerate(rows):

        # Extract 'Sample Interval' to determine the hardware sampling rate
        if len(row) >= 2 and row[0].strip().lower() == "sample interval":
            val = row[1].split(":")[-1].strip()
            sample_interval = float(val)
            
        # Detect where the actual numeric data starts by attempting to cast columns to float
        if data_start is None and len(row) > max(TIME_COL, SIG_COL):
            try:
                float(row[TIME_COL]); float(row[SIG_COL])
                data_start = i
            except (ValueError, TypeError):
                pass
    
    if data_start is None:
        return None

    # Load numeric data using NumPy, skipping the metadata header rows
    data = np.genfromtxt(path, delimiter=",", skip_header=data_start, 
                         usecols=(TIME_COL, SIG_COL), invalid_raise=False)
    
    # Handle single-row files to ensure consistent array dimensions
    if data.ndim == 1: 
        data = np.array([data])
        
    t, x = data[:, 0], data[:, 1]
    
    # Data Cleaning: Remove any non-finite values (NaNs or Inf)
    mask = np.isfinite(t) & np.isfinite(x)
    t, x = t[mask], x[mask]
    
    # Calculate Sampling Frequency (fs)
    # Prefer the header's interval; fallback to calculating the median delta between timestamps
    fs = 1.0 / sample_interval if sample_interval else 1.0 / np.median(np.diff(t))
    
    return t, x, fs

def analyze_current_dir():
    
    """
    Processes A.CSV, B.CSV, and C.CSV in the current directory and prints 
    spectral analysis metrics for the top 10 frequency peaks of each.
    """
    filenames = ["A.CSV", "B.CSV", "C.CSV"]
    current_dir = os.path.basename(os.getcwd())
    
    # Print formatted table header
    header = f"{'File':<10} | {'Freq Low (kHz)':<15} | {'Freq High (kHz)':<15} | {'Mag Min':<10} | {'Mag Max':<10} | {'CV'}"
    print(f"\n--- Analysis for {current_dir} ---")
    print(header)
    print("-" * len(header))

    for fname in filenames:
        if not os.path.exists(fname):
            continue
            
        result = load_csv(fname)
        if not result: 
            continue
            
        t, x, fs = result
        
        # --- Signal Processing (FFT) ---
        # 1. Detrend: Subtract mean to remove the DC offset (0 Hz component)
        x_detrend = x - np.mean(x)
        N = len(x_detrend)
        
        # 2. Compute FFT and the corresponding frequency bins
        X = np.fft.fft(x_detrend)
        freqs = np.fft.fftfreq(N, d=1.0/fs)
        
        # 3. Use only positive frequencies (the first half of the FFT result)
        pos_mask = freqs >= 0
        # Normalize magnitude: Multiply by 2/N to account for energy split in FFT
        X_mag = np.abs(X[pos_mask]) * 2.0 / N
        freqs_pos = freqs[pos_mask]
        
        # --- Feature Extraction ---
        # Identify the indices of the top 10 highest magnitude peaks
        idx = np.argsort(X_mag)[-10:]
        peak_freqs = freqs_pos[idx] / 1000.0  # Convert Hz to kHz
        peak_mags = X_mag[idx]
        
        # Compute summary statistics for the top peaks
        f_min, f_max = np.min(peak_freqs), np.max(peak_freqs)
        m_min, m_max = np.min(peak_mags), np.max(peak_mags)
        
        # Coefficient of Variation (CV): Measures relative variability of peak magnitudes
        cv = np.std(peak_mags) / np.mean(peak_mags)
        
        # Print results for the current file
        print(f"{fname:<10} | {f_min:14.2f} | {f_max:14.2f} | {m_min:9.4f} | {m_max:9.4f} | {cv:.4f}")

if __name__ == "__main__":
    analyze_current_dir()