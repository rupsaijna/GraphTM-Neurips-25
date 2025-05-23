import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============
# Create figure
# =============
plt.style.use("seaborn-v0_8")

sizes = [2048, 4096, 8192]
clauses = [4, 50, 200, 1000, 2000]

fig, axs = plt.subplots(1, 4, figsize=[10,3], sharey=True)
for i, size in enumerate(sizes):
    for j, clause in enumerate(clauses):
        df = pd.read_csv(f"results_unique_symbols_exp_{clause}_{size}.csv")
        g = sns.lineplot(data=df, x='epoch', y='accuracy', ax=axs[i], label=f"{clause} clauses")
    leg = axs[i].get_legend()
    axs[i].grid(False)
    axs[i].get_legend().remove()
    axs[i].set_title(f"Msg. HV size = {size}", fontsize=12)
axs[3].axis('off')
axs[3].legend(handles=leg.legend_handles,loc='center left')
box = axs[3].get_position()

# Adjust layout
plt.tight_layout()
axs[3].set_position([box.x0 * 1.025, box.y0, box.width * 0.6, box.height])
plt.show()