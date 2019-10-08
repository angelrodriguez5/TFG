import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from math import ceil

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path, tags):
	# Plot graphs in pairs
	if len(tags) >= 10:
		raise Exception('Max length of Tags is 9')
	pltgrid = 100 * ceil(len(tags)/2) + 20
	
	x_data = [[] for i in range(len(tags))]
	y_data = [[] for i in range(len(tags))]
	
	# Load all scalars and avoid loading more than 1 of the rest
	tf_size_guidance = {
		'compressedHistograms': 1,
		'images': 1,
		'scalars': 0,
		'histograms': 1
	}

	event_acc = EventAccumulator(path, tf_size_guidance)

	print("Loading...")
	event_acc.Reload()

	# Show all scalar tags in the log file
	print("Scalar tags available:")
	print(event_acc.Tags()['scalars'])
	
	# Load data corresponding to chosen tags
	for i, tag in enumerate(tags):
		try:
			event = event_acc.Scalars(tag)
			# Separate list of tuples (time, step, val) into three lists
			w_times, steps, values = zip(*event)
			# Set the current plot to its position in the grid
			plt.subplot(pltgrid + i + 1)
			plt.plot(steps, values)
			# plt.xlabel("Steps")
			# plt.ylabel("Values")
			plt.title(tag)
		except Exception:
			# Requested invalid tag
			print("Tag \"%s\" does not exist " % tag)
			
	plt.show()


if __name__ == '__main__':
	log_file = "..\experiment_logs\\testing\events.out.tfevents.1570544722.ymir"

	# List of tags to print
	tags = ["loss", "val_loss", "recall50", "val_recall", "precision", "val_precision"]

	plot_tensorflow_log(log_file, tags)