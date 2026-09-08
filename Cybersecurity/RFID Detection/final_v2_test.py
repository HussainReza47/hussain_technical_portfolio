import os
import csv
import numpy as np

# 1. Configuration: Authentication Thresholds
# These values define the Gold Standard for a legitimate device.
# Any signal falling outside these bounds is flagged as an anomaly.

FREQ_LOW   = 740.00     # Minimum acceptable median frequency in kHz
FREQ_HIGH  = 2300.00    # Maximum acceptable median frequency in kHz
MAG_LIMIT  = 0.2500     # Maximum power magnitude (prevents "Active Power" attacks or high-power clones)
CV_LIMIT   = 1.8200     # Max Coefficient of Variation (ensures the signal isn't too noisy/complex)

def load_csv(path):

    """
    Standardizes data ingestion from oscilloscope CSV exports.
    Identifies the hardware sampling rate and extracts raw voltage/time columns.
    """
    sample_interval, data_start = None, None
    with open(path, "r") as f:
        rows = list(csv.reader(f))
    
    # Header Parsing: Locate the sampling rate and the start of the numeric data block
    for i, row in enumerate(rows):
        if len(row) >= 2 and row[0].strip().lower() == "sample interval":
            val = row[1].split(":")[-1].strip()
            sample_interval = float(val)
        
        # Check column 3 (Time) and column 4 (Signal) for numeric data
        if data_start is None and len(row) > 4:
            try:
                float(row[3]); float(row[4])
                data_start = i
            except (ValueError, TypeError):
                pass
                
    if data_start is None: return None
    
    # Optimized bulk loading of numeric data
    data = np.genfromtxt(path, delimiter=",", skip_header=data_start, 
                         usecols=(3, 4), invalid_raise=False)
    
    if data.ndim == 1: data = np.array([data])
    t, x = data[:, 0], data[:, 1]
    
    # Cleaning: Filter out NaNs and Infs
    mask = np.isfinite(t) & np.isfinite(x)
    t, x = t[mask], x[mask]
    
    # Define fs (Sampling Frequency) for FFT scaling
    fs = 1.0 / sample_interval if sample_interval else 1.0 / np.median(np.diff(t))
    return t, x, fs

def run_ids_analysis():
    """
    Main IDS Loop: Scans the directory for CSV files and compares their 
    spectral signatures against the configured thresholds.
    """
    # Find all CSV files excluding baseline/calibration files
    files = [f for f in os.listdir('.') if f.endswith('.CSV') and 'Baseline' not in f]
    
    print(f"--- IDS: UPDATED DIGITAL FOOTPRINT ANALYSIS ---")
    print(f"Auth Freq: {FREQ_LOW}-{FREQ_HIGH}kHz | Max Mag: {MAG_LIMIT} | Max CV: {CV_LIMIT}")
    print("-" * 75)

    for fname in sorted(files):
        try:
            res = load_csv(fname)
            if res is None: continue
            t, x, fs = res
            
            # 1. Frequency Domain Transformation (FFT)
            # Remove DC bias (detrend) so 0Hz doesn't dominate the magnitude
            x_detrend = x - np.mean(x)
            N = len(x_detrend)
            X = np.fft.fft(x_detrend)
            freqs = np.fft.fftfreq(N, d=1.0/fs)
            
            # Work with positive frequencies only
            pos_mask = freqs >= 0
            X_mag = np.abs(X[pos_mask]) * 2.0 / N
            freqs_pos = freqs[pos_mask]
            
            # 2. Feature Extraction 
            # Extract top 10 peaks to characterize the signal's "fingerprint"
            idx = np.argsort(X_mag)[-10:]
            p_freqs = freqs_pos[idx] / 1000.0  # kHz conversion
            p_mags  = X_mag[idx]
            
            # Observed Metrics:
            # Median frequency of dominant peaks
            obs_freq = np.median(p_freqs) 

            # Absolute highest magnitude (peak power)
            obs_mag  = np.max(X_mag)   

            # Complexity (Standard Deviation / Mean)
            obs_cv   = np.std(p_mags) / np.mean(p_mags) if np.mean(p_mags) > 0 else 0

            #  3. Anomaly Detection Engine 
            # Evaluate the observed footprint against the configured limits
            f_pass = (FREQ_LOW <= obs_freq <= FREQ_HIGH)
            m_pass = (obs_mag <= MAG_LIMIT)
            c_pass = (obs_cv <= CV_LIMIT)

            print(f"FILE: {fname}")
            print(f"  [FREQ]: {obs_freq:>8.2f} kHz | [MAG]: {obs_mag:>8.4f} | [CV]: {obs_cv:>8.4f}")

            # 4. Verdict Reporting 
            if f_pass and m_pass and c_pass:
                print("  RESULT: [ PASS ] Authentic Digital Footprint")
            else:
                anomalies = []
                if not f_pass: anomalies.append("Frequency Shift")
                if not m_pass: anomalies.append("Active Power Detected")
                if not c_pass: anomalies.append("Spectral Complexity (Noise)")
                print(f"  RESULT: [ FAIL ] Anomaly: {', '.join(anomalies)}")
            print("-" * 75)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    run_ids_analysis()