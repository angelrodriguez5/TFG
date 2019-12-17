import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from math import ceil
from statistics import mean, stdev

import matplotlib as mpl
import matplotlib.pyplot as plt

import glob

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
	for name, path in paths_dic.items():
		# There should only be a file with ymir extension in the directory
		logfile = glob.glob("%s\\*.ymir" % path)[0]
		event_accs[name] = EventAccumulator(logfile, tf_size_guidance)
		event_accs[name].Reload()
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
				# Print best f1 scores of checkpoint epochs
				checkpoint_interval = 10 # Checkpoint every 10th epoch
				if (tag == "val_f1"):
					chkpts = values[::checkpoint_interval]
					print("%s: 	Maximum f1 Score is: %f" % (name, max(chkpts)))
					print(" 	Achieved in epoch %d" % (chkpts.index(max(chkpts)) * checkpoint_interval))

			# Show legend on the first subplot, it is the same for the rest
			if i == 0:
				plt.legend(loc='upper right', frameon=True)

		except Exception:
			# Requested invalid tag
			print("Tag \"%s\" does not exist in log with name \"%s\"" % (tag, name))

	plt.subplots_adjust(hspace=0.5)
	plt.show()


def plot_crossvalidation_logs(paths_dic, tags):
	# Plot graphs in pairs
	if len(tags) >= 10:
		raise Exception('Max length of Tags is 9')
	pltgrid = 100 * len(tags) + 10
	
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
	for name, path in paths_dic.items():
		# There should only be a file with ymir extension in the directory
		logfiles = glob.glob("%s\\*.ymir" % path)
		tmp = []
		for f in logfiles:
			event_acc = EventAccumulator(f, tf_size_guidance)
			event_acc.Reload()
			tmp.append(event_acc)
		event_accs[name] = tmp
	print("-----")

	plt.figure(figsize=[7,9])
	# Load data corresponding to chosen tags
	for i, tag in enumerate(tags):
		try:
			# Set the current plot to its position in the grid
			plt.subplot(pltgrid + i + 1)
			plt.title(tag)
			# Extract data from event accumulators (should be the same length)
			for name, events in event_accs.items():
				data = []
				for acc in events:
					event = acc.Scalars(tag)
					# Separate list of tuples (time, step, val) into three lists
					w_times, steps, values = zip(*event)
					data.append(values)

				# Calculate mean and std deviation
				m = [mean(x) for x in zip(*data)]
				sd = [stdev(x) for x in zip(*data)]

				plt.plot(steps, m, label=name)


			# Show legend on the first subplot, it is the same for the rest
			if i == 0:
				plt.legend(loc='best', frameon=True)

		except Exception as e:
			# Requested invalid tag
			print(e)
			print("Tag \"%s\" does not exist in log with name \"%s\"" % (tag, name))

	plt.subplots_adjust(hspace=0.35)
	plt.show()

if __name__ == '__main__':
	'''
	# One log at a time
	exp_dir = "C:\\Users\\pickl\\Documents\\UDC2018\\TFG-NoGit\\experiment_logs\\"
	exp1 = exp_dir + "crossvalidation\\tttvx"
	exp2 = exp_dir + "crossvalidation\\xtttv"
	exp3 = exp_dir + "crossvalidation\\vxttt"
	exp4 = exp_dir + "crossvalidation\\tvxtt"
	exp5 = exp_dir + "crossvalidation\\ttvxt"
	# Dictionary of user-defined log names and their directories
	# All log files will overlap in each graph and the legend will show the name given by this dictionary
	log_files = {"tttvx": exp1,
				 "xtttv": exp2,
				 "vxttt": exp3,
				 "tvxtt": exp4,
				 "ttvxt": exp5}

	# List of tags to print for each log
	tags = ["tra_loss", "val_loss", "tra_recall", "val_recall", "tra_precision", "val_precision", "val_f1", "neg_test_#FP"]
	plot_tensorflow_log(log_files, tags)

	''' 

	# Cross validation 
	exp_dir = "C:\\Users\\pickl\\Documents\\UDC2018\\TFG-NoGit\\experiment_logs\\"
	exp1 = exp_dir + "Augmented_final"
	exp2 = exp_dir + "NotAugmented_final"

	# Dictionary of user-defined log names and their directories
	# All log files will overlap in each graph and the legend will show the name given by this dictionary
	log_files = {"Final config": exp1}

	tags = ["val_recall", "val_precision"]
	# tags = ["tra_loss", "tra_recall", "tra_precision", "tra_f1"]
	plot_crossvalidation_logs(log_files, tags)
