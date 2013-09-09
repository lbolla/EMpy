__author__ = 'Lorenzo Bolla'

# packages required by EMpy
# stop if not present
dependencies = ['numpy', 'scipy', 'matplotlib']

# packages required only by some functions of EMpy (not fundamental)
# warn if not present
light_dependencies = ['bvp', 'scipy.sparse.linalg']

for dep in dependencies:
    try:
        exec('import ' + dep)
    except:
        raise ImportError(dep + ' not found')

for dep in light_dependencies:
    try:
        exec('import ' + dep)
    except:
        print('WARNING -- ' + dep + ' not found. Continue anyway.')
