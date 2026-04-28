import subprocess
import os
import sys

def run_command(command):
    print(f"Running command: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"Command '{command}' failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def main():
    print("=== STARTING FULL PIPELINE ===")
    
    # Step 1: Preprocessing
    run_command("python src/preprocessing.py")
    
    # Step 2: Training Models
    run_command("python autoencoder_model.py")
    run_command("python random_forest_model.py")
    run_command("python xgboost_model.py")
    run_command("python hybrid_model.py")
    
    # Step 3: Evaluation
    run_command("python evaluate_model.py")
    
    # Step 4: Start Flask
    print("=== PIPELINE COMPLETED. STARTING FLASK APP... ===")
    run_command("python app.py")

if __name__ == "__main__":
    main()
