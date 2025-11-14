import subprocess, sys, webbrowser, os, time, requests

def run_training():
    if not os.path.exists('data.csv'):
        print("data.csv not found!")
        return False
    return subprocess.run([sys.executable, "dropout_random_forest_clean.py"], check=True).returncode == 0

def wait_for_streamlit(port=8501, timeout=30):
    url = f"http://localhost:{port}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False

def run_dashboard():
    proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port=8501", "--server.headless=true"
    ])
    print("Starting Streamlit… (port 8501)")
    if wait_for_streamlit():
        webbrowser.open('http://localhost:8501')
        print("Dashboard is live!")
    else:
        print("Timed-out waiting for Streamlit.")
        proc.terminate()

def main():
    print("STUDENT DROPOUT SYSTEM")
    input("Press Enter to start…")
    if run_training():
        run_dashboard()
    else:
        print("Training failed – aborting.")

if __name__ == "__main__":
    main()
