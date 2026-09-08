import os
import csv
import numpy as np

# Indices for the expected data columns in the oscilloscope CSV
TIME_COL = 3
SIG_COL = 4

def load_csv(path):

    """
    Extracts raw signal data and hardware sampling rate from an oscilloscope export.
    """
    sample_interval = None
    data_start = None
    with open(path, "r") as f:
        rows = list(csv.reader(f))
        
    # Search metadata for 'sample interval' and locate the start of numeric data
    for i, row in enumerate(rows):
        if len(row) >= 2 and row[0].strip().lower() == "sample interval":
            val = row[1].split(":")[-1].strip()
            sample_interval = float(val)
        if data_start is None and len(row) > max(TIME_COL, SIG_COL):
            try:
                float(row[TIME_COL]); float(row[SIG_COL])
                data_start = i
            except (ValueError, TypeError):
                pass
    
    if data_start is None: return None
    
    # Load the raw voltage/time pairs, ignoring non-numeric header rows
    data = np.genfromtxt(path, delimiter=",", skip_header=data_start, 
                         usecols=(TIME_COL, SIG_COL), invalid_raise=False)
    
    if data.ndim == 1: data = np.array([data])
    t, x = data[:, 0], data[:, 1]
    
    # Clean data: Remove any invalid or infinite entries
    mask = np.isfinite(t) & np.isfinite(x)
    t, x = t[mask], x[mask]
    
    # Define fs (Sampling Frequency). Uses median delta if interval is missing.
    fs = 1.0 / sample_interval if sample_interval else 1.0 / np.median(np.diff(t))
    return t, x, fs

if __name__ == "__main__":

    # Path configuration
    BASELINE_PATH = "."
    groups_files = ["A.CSV", "B.CSV", "C.CSV"]
    
    # Lists to aggregate features across all three baseline files
    all_freqs = []
    all_mags = []
    file_complexities = []

    for file in groups_files:
        path = os.path.join(BASELINE_PATH, file)
        if not os.path.exists(path): continue
        
        result = load_csv(path)
        if result is None: continue
        t, x, fs = result
        
        # --- Pre-Processing 
        # Detrending (removing DC offset) to focus on AC signal components
        x = x - np.mean(x)
        N = len(x)
        
        # Perform Fast Fourier Transform (FFT)
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(N, d=1.0 / fs)
        
        # Keep only the positive half of the frequency spectrum
        pos_mask = freqs >= 0
        X_mag = np.abs(X[pos_mask]) * 2.0 / N
        freqs_pos = freqs[pos_mask]
        
        # ---Feature Exteaction (Local to file) ---
        # Identify the indices of the top 10 most dominant signal peaks
        idx = np.argsort(X_mag)[-10:]
        top_freqs = freqs_pos[idx] / 1000.0  # Convert to kHz for readability
        top_mags = X_mag[idx]
        
        # Add to global lists for aggregate cross-file analysis
        all_freqs.extend(top_freqs)
        all_mags.extend(top_mags)
        
        # Calculate CV (Coefficient of Variation) to measure signal 'noisiness'
        cv = np.std(top_mags) / np.mean(top_mags)
        file_complexities.append(cv)

    # --- Aggregate Baseline Metrics 
    # We use Median and MAD instead of Mean/StdDev because they are more robust 
    # against outliers in the oscilloscope data.
    if all_freqs:

        # The central frequency of the legitimate signal
        med_freq = np.median(all_freqs)

        # The typical power magnitude of legitimate peaks
        med_mag  = np.median(all_mags)
        
        # The typical complexity/noise level of the device
        med_cv   = np.median(file_complexities)
        
        # Median Absolute Deviation (MAD): A robust measure of frequency spread.
        # This helps in defining the 'Allowable Range' (e.g., med_freq +/- 3*mad_freq)
        mad_freq = np.median(np.abs(np.array(all_freqs) - med_freq))

        print(f"--- Baseline Fingerprint Results ---")
        print(f"Median Frequency:  {med_freq:.4f} kHz")
        print(f"Median Magnitude:  {med_mag:.4f}")
        print(f"Median Complexity: {med_cv:.4f} (CV)")
        print(f"Freq MAD:          {mad_freq:.4f} kHz")
        print("-" * 35)
        print("\nUse these values in your testing script configuration.")
    else:
        print("No baseline data found.")