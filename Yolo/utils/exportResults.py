import os

def exportResults(experimentName):
    print("Creating experiment directory...")
    dirname = "/home/angel/experiments/%s" % (experimentName)
    os.system("mkdir %s" % (dirname))
    
    print("Moving experiment results to directory...")
    os.system("mv checkpoints/ output/ analytics/ logs/ %s" % (dirname))

    print("Don't forget to fill in the experiment description.txt in the experiment's directory")
    os.system("touch %s/description.txt" % (dirname))

    print("Done!")

