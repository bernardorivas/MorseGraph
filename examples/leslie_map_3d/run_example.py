import CMGDB
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import io

# Import saving functions from their modules
import CMGDB.SaveMorseData

from MorseGraph.systems import leslie_map_3d as f

def F(rect):
    return CMGDB.BoxMap(f, rect)

# [[-2.3, -1.8, -0.5], [95.2, 75.1, 72.3]] bounds for seed 42
lower_bounds = [-2.5, -2.0, -0.7]
upper_bounds = [96.0, 76.0, 73.0]

subdiv_min = 39
subdiv_max = 42
subdiv_init = 24
subdiv_limit = 10000

model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit, lower_bounds, upper_bounds, F)

print("Computing Morse graph...")
start_time = time.time()
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
end_time = time.time()
computation_time = end_time - start_time
print(f"Morse graph computation took {computation_time:.2f} seconds.")

# Create a directory to save results
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "leslie_map_3d_results")
os.makedirs(output_dir, exist_ok=True)
print(f"Saving results to {output_dir}")

# --- Save Morse Sets to CSV ---
morse_sets_fname = os.path.join(output_dir, "morse_sets.csv")
CMGDB.SaveMorseData.SaveMorseSets(morse_graph, morse_sets_fname)
print(f"Morse sets saved to {morse_sets_fname}")

# --- Compute Barycenters ---
print("Computing barycenters of Morse sets...")
barycenters = {}
for i in range(morse_graph.num_vertices()):
    morse_set_boxes = morse_graph.morse_set_boxes(i)
    barycenters[i] = []
    if morse_set_boxes:
        dim = len(morse_set_boxes[0]) // 2
        for box in morse_set_boxes:
            barycenter = np.array([(box[j] + box[j + dim]) / 2.0 for j in range(dim)])
            barycenters[i].append(barycenter)

# Save barycenters to a file
barycenters_fname = os.path.join(output_dir, "barycenters.npz")
barycenters_to_save = {f'morse_set_{k}': np.array(v) for k, v in barycenters.items() if v}
if barycenters_to_save:
    np.savez(barycenters_fname, **barycenters_to_save)
    print(f"Barycenters saved to {barycenters_fname}")

# --- Save complete Morse graph computation results ---
print("Saving complete Morse graph data...")
model_params = {
    'subdiv_min': subdiv_min,
    'subdiv_max': subdiv_max,
    'subdiv_init': subdiv_init,
    'subdiv_limit': subdiv_limit,
    'lower_bounds': lower_bounds,
    'upper_bounds': upper_bounds,
}
runtime_info = {
    'computation_time_seconds': computation_time
}
metadata = {
    'model_params': model_params,
    'runtime_info': runtime_info,
    'barycenters': barycenters
}
morse_graph_data_fname = os.path.join(output_dir, "morse_graph_data.mgdb")
CMGDB.SaveMorseGraphData(morse_graph, map_graph, morse_graph_data_fname, metadata=metadata)

def plot_morse_graph_to_ax(morse_graph, ax, title=''):
    if morse_graph is None:
        ax.text(0.5, 0.5, 'Not computed', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
        return
    gv_source = CMGDB.PlotMorseGraph(morse_graph, cmap=plt.cm.cool)
    img_data = gv_source.pipe(format='png')
    img = plt.imread(io.BytesIO(img_data))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

# --- Plotting ---
print("Plotting Morse graph...")
fig_mg, ax_mg = plt.subplots()
plot_morse_graph_to_ax(morse_graph, ax_mg, title='Morse Graph')
plt.savefig(os.path.join(output_dir, "morse_graph.png"))

print("Plotting barycenters of Morse sets...")
fig_barycenters = plt.figure()
ax_barycenters = fig_barycenters.add_subplot(111, projection='3d')

num_morse_sets = morse_graph.num_vertices()
colors = matplotlib.cm.cool(np.linspace(0, 1, num_morse_sets))

for i, points in barycenters.items():
    if not points:
        continue
    data = np.array(points)
    ax_barycenters.scatter(data[:, 0], data[:, 1], data[:, 2], c=[colors[i]], marker='s', s=1, label=f'Morse Set {i}')

ax_barycenters.set_xlabel("x")
ax_barycenters.set_ylabel("y")
ax_barycenters.set_zlabel("z")
ax_barycenters.set_title("Barycenters of Morse Sets")
plt.savefig(os.path.join(output_dir, "barycenters_scatterplot.png"))

print("Displaying plots...")
plt.show()

print("Done.")
