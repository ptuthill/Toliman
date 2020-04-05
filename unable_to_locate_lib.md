# Fixing unable to locate lib

dir_name = os.getcwd().split('/')[-1]
parent_path = os.getcwd()[:-(len(dir_name) + 1)]
sys.path.append(parent_path)

This adds the parent directory to the path so taht lib can be acessed in the same way - no change to importing code