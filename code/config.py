import os

HOME = os.path.expanduser("~")
DOCUMENTS = "%s/Documents" % HOME
RESOURCES = "%s/nlp_resources" % DOCUMENTS

REUTERS_DIR = "%s/reuters21578" % RESOURCES
REUTERS_TRAIN = "%s/reuters_train.npy" % RESOURCES
REUTERS_TEST = "%s/reuters_test.npy" % RESOURCES
REUTERS_META = "%s/reuters_meta.txt" % RESOURCES
