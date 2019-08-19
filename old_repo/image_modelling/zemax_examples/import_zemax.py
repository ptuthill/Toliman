import numpy as np
import codecs
import os
import sys

for filename in sys.argv[1:]:
    with codecs.open(filename, encoding='utf-16') as f:
        data = np.loadtxt(f, skiprows=21)
    # Export it as a more convenient (and compact) numpy binary file
    np.save(os.path.splitext(filename)[0], data)
