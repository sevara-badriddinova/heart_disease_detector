import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')  # Hide the axes

# Define box positions and labels
boxes = [
    (0.1, 0.7, 0.3, 0.15, "Raw ECG Data\n(300+ rows)"),
    (0.6, 0.7, 0.3, 0.15, "Raw Heart Disease Data\n(300+ rows)"),
    (0.1, 0.45, 0.8, 0.15, "Reset Index & Select First 200 Rows\n(To Align and Test Faster)"),
    (0.1, 0.25, 0.8, 0.15, "Merge ECG + Heart Data by Row Index"),
    (0.1, 0.05, 0.8, 0.15, "Drop or Fill Missing Values (NaN)\nResult: Clean Data for Model")
]

# Draw each box and add label
for (x, y, w, h, label) in boxes:
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", edgecolor="black", facecolor="#cfe2f3")
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10)

# Draw arrows between the boxes
arrow_props = dict(arrowstyle="->", color="black", lw=1.5)
arrows = [((0.25, 0.7), (0.5, 0.625)),
          ((0.75, 0.7), (0.5, 0.625)),
          ((0.5, 0.625), (0.5, 0.525)),
          ((0.5, 0.525), (0.5, 0.4)),
          ((0.5, 0.4), (0.5, 0.22))]

for (start, end) in arrows:
    ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)

# Save the image
plt.tight_layout()
plt.savefig("data_filtering_flow.png")
plt.show()
