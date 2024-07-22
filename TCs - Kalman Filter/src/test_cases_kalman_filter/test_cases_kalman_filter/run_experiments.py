import os
import subprocess
import time
import json

num_antennas_list = ["8", "16", "32", "64"]
noise_levels = ["No noise", "Low noise", "Medium noise", "High noise"]
test_cases = ["TC1", "TC2", "TC3"]

launch_files = {
    "TC1": "webots_launch_tc1.py",
    "TC2": "webots_launch_tc2.py",
    "TC3": "webots_launch_tc3.py"
}

def run_experiments(num_antennas, noise_level, test_case):
    config_file = './test_cases_kalman_filter/config.json'
    with open(config_file, 'w') as f:
        config = {
            "num_antennas": num_antennas,
            "noise_level": noise_level,
            "test_case": test_case
        }
        json.dump(config, f)

    # Run the experiments
    subprocess.run(['colcon', 'build'])
    launch_file = launch_files[test_case]
    process = subprocess.Popen(["bash", "-c", f"source install/local_setup.bash && ros2 launch test_cases_kalman_filter {launch_file}"])
    process.wait()

for test_case in test_cases:
    for num_antennas in num_antennas_list:
        for noise_level in noise_levels:
            run_experiments(num_antennas, noise_level, test_case)
            print("Experiments for ",test_case ,", with ", num_antennas, " antennas and ", noise_level, " done")