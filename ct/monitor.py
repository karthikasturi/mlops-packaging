"""
Simple monitoring script that combines drift detection and continuous training
Run this periodically (e.g., via cron job) to monitor and retrain models
"""
import os
import sys
from detect_drift import simulate_drift_detection, save_drift_report
from continuous_training import ContinuousTrainingPipeline

def monitor_and_retrain():
    """
    Main monitoring function that:
    1. Checks for drift
    2. Triggers retraining if needed
    3. Logs the results
    """
    print("="*60)
    print("AUTOMATED MODEL MONITORING AND RETRAINING")
    print("="*60)
    
    # Step 1: Drift Detection
    print("\n[DRIFT DETECTION PHASE]")
    retrain_needed = simulate_drift_detection()
    
    # Save drift report
    save_drift_report(retrain_needed)
    
    # Step 2: Continuous Training (if needed)
    if retrain_needed:
        print("\n[CONTINUOUS TRAINING PHASE]")
        print("Drift detected! Initiating automatic retraining...\n")
        
        pipeline = ContinuousTrainingPipeline()
        success = pipeline.run_continuous_training()
        
        if success:
            print("\n✓ Model successfully retrained and deployed!")
            return 0
        else:
            print("\n⚠️  Retraining completed but old model retained")
            return 1
    else:
        print("\n✓ No drift detected. Model monitoring complete.")
        return 0

if __name__ == "__main__":
    exit_code = monitor_and_retrain()
    sys.exit(exit_code)
