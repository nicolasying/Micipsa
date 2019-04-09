from distutils.core import setup
from Cython.Build import cythonize
import numpy

# setup(name='orthonormalize app', 
#     ext_modules=cythonize("orthonormalize_lib.pyx"),
#     include_dirs=[numpy.get_include()]
#     )

# setup(name='ridge module', 
#     ext_modules=cythonize("./ridge_lib.pyx"),
#     include_dirs=[numpy.get_include()]
#     )

# setup(name='ridge_all_in_once module', 
#     ext_modules=cythonize("./ridge_all_lib.pyx"),
#     include_dirs=[numpy.get_include()]
#     )

# setup(name='lasso_all_in_once module', 
# ext_modules=cythonize("./lasso_lib.pyx"),
# include_dirs=[numpy.get_include()]
# )

# setup(name='dim_search_with_glm module', 
# ext_modules=cythonize("./GLM_dim_search_lib.pyx"),
# include_dirs=[numpy.get_include()]
# )

setup(name='dim_alpha_search module', 
ext_modules=cythonize("./dim_alpha_search_lib.pyx"),
include_dirs=[numpy.get_include()]
)