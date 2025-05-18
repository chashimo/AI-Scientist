import json
import matplotlib.pyplot as plt

# Define the labels for each run
labels = {
    "run_0": "Baseline",
    "run_1": "Diverse Narrative Generation",
    "run_2": "Increased Query Generation"
}

# Load results from each run
results = {}
for run, label in labels.items():
    with open(f"{run}/final_info.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        results[label] = data["youtube_propaganda_detection"]["means"]

# Extract data for plotting
hit_rates = [results[label]["hit_rate"] for label in labels.values()]
hits = [results[label]["hits"] for label in labels.values()]
total_videos = [results[label]["total_videos"] for label in labels.values()]

# Plot hit rates
plt.figure(figsize=(10, 6))
plt.bar(labels.values(), hit_rates, color=['blue', 'orange', 'green'])
plt.title('Hit Rate Comparison Across Runs')
plt.xlabel('Experiment')
plt.ylabel('Hit Rate')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.savefig('hit_rate_comparison.png')
plt.close()

# Plot total hits
plt.figure(figsize=(10, 6))
plt.bar(labels.values(), hits, color=['blue', 'orange', 'green'])
plt.title('Total Hits Comparison Across Runs')
plt.xlabel('Experiment')
plt.ylabel('Total Hits')
plt.grid(axis='y')
plt.savefig('total_hits_comparison.png')
plt.close()

# Plot total videos
plt.figure(figsize=(10, 6))
plt.bar(labels.values(), total_videos, color=['blue', 'orange', 'green'])
plt.title('Total Videos Processed Across Runs')
plt.xlabel('Experiment')
plt.ylabel('Total Videos')
plt.grid(axis='y')
plt.savefig('total_videos_comparison.png')
plt.close()

print("Plots generated: hit_rate_comparison.png, total_hits_comparison.png, total_videos_comparison.png")
