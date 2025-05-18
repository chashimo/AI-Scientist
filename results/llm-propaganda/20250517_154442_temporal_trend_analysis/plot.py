import matplotlib.pyplot as plt
import json
import os

def plot_trends(results_file, output_dir):
    # Define labels for each run
    labels = {
        'run_0': 'Baseline',
        'run_1': 'Timestamp Tagging',
        'run_2': 'Time-Series Analysis'
    }

    with open(results_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    time_series_data = {}
    for entry in data:
        date = entry['timestamp'].split('T')[0]
        time_series_data[date] = time_series_data.get(date, 0) + 1

    dates = sorted(time_series_data.keys())
    frequencies = [time_series_data[date] for date in dates]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, frequencies, marker='o', label=labels.get(output_dir, 'Unknown'))
    plt.title('Propaganda Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend_analysis.png"))
    plt.close()

if __name__ == "__main__":
    plot_trends('run_2/results_propaganda.jsonl', 'run_3')
