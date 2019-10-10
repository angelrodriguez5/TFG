import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from math import ceil

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(paths_dic, tags):
	# Plot graphs in pairs
	if len(tags) >= 10:
		raise Exception('Max length of Tags is 9')
	pltgrid = 100 * ceil(len(tags)/2) + 20
	
	# Load all scalars
	tf_size_guidance = {
		'compressedHistograms': 1,
		'images': 1,
		'scalars': 0,
		'histograms': 1
	}

	print("Loading...")
	# Dictionary of log_name: event accumulator
	event_accs = {}
	for k, v in paths_dic.items():
		event_accs[k] = EventAccumulator(v, tf_size_guidance)
		event_accs[k].Reload()
	print("-----")

	# Show all scalar tags in the log file
	# print("Scalar tags available:")
	# print(event_accs[k].Tags()['scalars'])

	# Load data corresponding to chosen tags
	for i, tag in enumerate(tags):
		try:
			# Set the current plot to its position in the grid
			plt.subplot(pltgrid + i + 1)
			plt.title(tag)

			# Plot the same tag in the same subplot for each log file
			for name, event_acc in event_accs.items():
				event = event_acc.Scalars(tag)
				# Separate list of tuples (time, step, val) into three lists
				w_times, steps, values = zip(*event)
				# Plot evolution of values with respect to steps
				plt.plot(steps, values, label=name)

			# Show legend on the first subplot, it is the same for the rest
			if i == 0:
				plt.legend(loc='upper right', frameon=True)

		except Exception:
			# Requested invalid tag
			print("Tag \"%s\" does not exist in log with name \"%s\"" % (tag, name))

	plt.show()


if __name__ == '__main__':
	# Dictionary of user-defined log names and their directories
	# All log files will overlap in each graph and the legend will show the name given by this dictionary
	log_files = {"lre-5": "C:\\Users\\pickl\\Documents\\UDC2018\\TFG-NoGit\\experiment_logs\\learningRate\\lr-5\\events.out.tfevents.1570546151.ymir",
				 "lre-6": "C:\\Users\\pickl\\Documents\\UDC2018\\TFG-NoGit\\experiment_logs\\learningRate\\lr-6\\events.out.tfevents.1570615720.ymir"}

	# List of tags to print for each log
	tags = ["loss", "val_loss", "recall50", "val_recall", "precision", "val_precision", "neg_test_#FP", "neg_test_mFP"]

	plot_tensorflow_log(log_files, tags)