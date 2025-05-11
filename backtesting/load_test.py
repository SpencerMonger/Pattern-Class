# backtesting/load_test.py
import pickle
import os
import sys

# Try adding psutil for memory debugging if available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def print_memory_usage(stage=""):
    """Prints current process and system memory usage if psutil is available."""
    if not PSUTIL_AVAILABLE:
        print(f"[{stage}] psutil not found, cannot report detailed memory.")
        return
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        virtual_mem = psutil.virtual_memory()
        print(f"[{stage}] Process Memory: {mem_info.rss / (1024 * 1024):.2f} MB (RSS)")
        print(f"[{stage}] System Memory : Total={virtual_mem.total / (1024*1024):.2f}MB, Available={virtual_mem.available / (1024*1024):.2f}MB ({virtual_mem.percent}%)")
    except Exception as e:
        print(f"[{stage}] Error getting memory info: {e}")


# Determine paths relative to this script file
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir) # Go up one level to datahead_v3
model_dir = os.path.join(base_dir, "saved_models")
model_filename = "random_forest_NEWsmall_model.pkl" # Ensure this is the correct filename
model_path = os.path.join(model_dir, model_filename)

print(f"--- Starting Model Load Test ---")
print(f"Python Version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Attempting to load model from: {model_path}")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at the specified path.")
    sys.exit(1)

print_memory_usage("Before Load")

try:
    with open(model_path, 'rb') as f:
        print("File opened successfully. Attempting pickle.load()...")
        model = pickle.load(f)
        print("--- Model loaded successfully! ---")
        print(f"Model type: {type(model)}")
        # You could add more checks here, like model.feature_names_in_ if it's sklearn
        print_memory_usage("After Load")

except MemoryError:
    print("--- MemoryError occurred during pickle.load() ---")
    print_memory_usage("During MemoryError")
    # Suggest potential causes even if memory seems high
    print("Possible causes despite high RAM:")
    print(" - Intermittent system resource contention (try restarting).")
    print(" - Internal pickle file structure issue (corruption?).")
    print(" - Python environment subtle change (version, libraries).")

except FileNotFoundError:
     print(f"--- FileNotFoundError: Could not find model file at {model_path} ---")

except Exception as e:
    print(f"--- An unexpected error occurred: {type(e).__name__} ---")
    print(f"Error details: {e}")
    import traceback
    traceback.print_exc()
    print_memory_usage("During Other Error")

print("--- Load Test Script Finished ---") 