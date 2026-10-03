"""Compression topics. Importing this package registers every algorithm.

Each topic folder holds that method's scripts, figures, and algorithm code.
The engine looks methods up by name and does not define them.
"""

from compression_topics.checksparse.algorithms import checksparse_l1 as _checksparse_l1  # noqa: F401
from compression_topics.checksparse.algorithms import checksparse_row as _checksparse_row  # noqa: F401
from compression_topics.qjl.algorithms import qjl as _qjl  # noqa: F401
from compression_topics.quantize.algorithms import quantize as _quantize  # noqa: F401
from compression_topics.sparsify.algorithms import sparsify_nm as _sparsify_nm  # noqa: F401
from compression_topics.spatial.algorithms import pair_pool as _pair_pool  # noqa: F401
from compression_topics.spatial.algorithms import residual_pool as _residual_pool  # noqa: F401
from compression_topics.spatial.algorithms import spatial as _spatial  # noqa: F401
from compression_topics.vector.algorithms import vector_compress as _vector_compress  # noqa: F401
from compression_topics.vector.algorithms import vector_compress_pair as _vector_compress_pair  # noqa: F401
