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
		logfiles = sorted(glob.glob("%s\\*.ymir" % path))
		tmp = []
		for f in logfiles:
			event_acc = EventAccumulator(f, tf_size_guidance)
			event_acc.Reload()
			tmp.append(event_acc)
		event_accs[name] = tmp

	folds = ["tttvx","xtttv","vxttt","tvxtt","ttvxt"]
	
	# Extract data from event accumulators (should be the same length)
	for name, events in event_accs.items():
		data = []
		for acc, fold in zip(events, folds):
			
			f1s = None
			rcs = None

			for i, tag in enumerate(tags):
				event = acc.Scalars(tag)
				# Separate list of tuples (time, step, val) into three lists
				w_times, steps, values = zip(*event)
				data.append(values)

				# Print best f1 scores of checkpoint epochs
				checkpoint_interval = 10 # Checkpoint every 10th epoch
				print("*****************")
				if (tag == "val_f1"):
					f1s = values[::checkpoint_interval]
					print("%s: 	Maximum f1 Score is: %f" % (name, max(f1s)))
					print(" 	Achieved in fold %s epoch %d" % (fold, (f1s.index(max(f1s)) * checkpoint_interval)))

				# Print best recalls
				if (tag == "val_recall"):
					rcs = values[::checkpoint_interval]
					print("%s: 	Maximum recall is: %f" % (name, max(rcs)))
					print(" 	Achieved in fold %s epoch %d" % (fold, (rcs.index(max(rcs)) * checkpoint_interval)))

				if (f1s is not None) and (rcs is not None):
					print("*****************")
					f1s = np.array(f1s)
					rcs = np.array(rcs)
					sums = (f1s + rcs).tolist()
					test_run = sums.index(max(sums))
					print("%s: 	Maximum sum of metrics is: %f" % (name, max(sums)))
					print(" 	Achieved in fold %s epoch %d" % (fold, test_run * checkpoint_interval))
					print()
					
					print("Test metrics of best epoch:")
					# We use test_run - 1 because in epoch 0 we do not test
					# Save test values in the best epoch
					event = acc.Scalars("test_recall")
					w_times, steps, values = zip(*event)
					print("			Epoch : %d" % steps[test_run - 1])
					print("			Recall : %f" % values[test_run - 1])

					event = acc.Scalars("test_precision")
					w_times, steps, values = zip(*event)
					print("			Precision : %f" % values[test_run - 1])

					event = acc.Scalars("test_f1")
					w_times, steps, values = zip(*event)
					print("			F1 : %f" % values[test_run - 1])

					event = acc.Scalars("neg_test_#FP")
					w_times, steps, values = zip(*event)
					print("			FP : %d" % values[test_run - 1])
					


if __name__ == '__main__':
	# Cross validation 
	exp_dir = "C:\\Users\\pickl\\Documents\\UDC2018\\TFG-NoGit\\experiment_logs\\"
	exp1 = exp_dir + "Augmented_final"
	exp2 = exp_dir + "NotAugmented_final"
	# Dictionary of user-defined log names and their directories
	# All log files will overlap in each graph and the legend will show the name given by this dictionary
	log_files = {"Augmented": exp1}

	tags = ["val_recall", "val_f1"]
	plot_crossvalidation_logs(log_files, tags)
