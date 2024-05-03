import json
import pdb
import matplotlib.pyplot as plt


num_instruct_data = {
    'single-trial summarization': 5000,
    'multi-trial summarization': 2029,
    'query generation': 2161,
    'query expansion': 51000,
    'patient-trial matching': 6368,
    'criteria design': 35951,
    'sutdy arm design': 53548,
    'outcome measure design': 44809
}


# Using the provided hex codes to replicate the gradient effect exactly as in the user's image
hex_codes = [
    "#7400B8", "#6930C3", "#5E60CE", "#5390D9", 
    "#4EA8DE", "#48BFE3", "#56CFE1", "#64DFDF"
]

task_names = list(num_instruct_data.keys())
data_counts = list(num_instruct_data.values())

plt.barh(task_names, data_counts, color=hex_codes)
plt.xscale('log')
plt.xlabel('Number of Data Points')
plt.title('Number of data points per task with specified gradient color')

plt.savefig('bar_plot.png')