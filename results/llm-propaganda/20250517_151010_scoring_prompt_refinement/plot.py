import json
import matplotlib.pyplot as plt

# Define the labels for each run
labels = {
    "run_0": "Baseline",
    "run_1": "Clarify Scale Anchors",
    "run_2": "Add Negative/Neutral Examples",
    "run_3": "Refine Role Instructions"
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
plt.bar(labels.values(), hit_rates, color='skyblue')
plt.title('Hit Rates for Each Experiment')
plt.xlabel('Experiment')
plt.ylabel('Hit Rate')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('hit_rates.png')
plt.close()

# Plot hits
plt.figure(figsize=(10, 6))
plt.bar(labels.values(), hits, color='lightgreen')
plt.title('Total Hits for Each Experiment')
plt.xlabel('Experiment')
plt.ylabel('Hits')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('hits.png')
plt.close()

# Plot total videos
plt.figure(figsize=(10, 6))
plt.bar(labels.values(), total_videos, color='salmon')
plt.title('Total Videos Analyzed for Each Experiment')
plt.xlabel('Experiment')
plt.ylabel('Total Videos')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('total_videos.png')
plt.close()

print("Plots generated: hit_rates.png, hits.png, total_videos.png")
