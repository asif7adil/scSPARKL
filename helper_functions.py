# helper functions

from pyspark.sql.functions import col, explode, lit
from pyspark.sql.functions import create_map
from pyspark.sql import DataFrame
from typing import Iterable 
from itertools import chain

def melt_spark(
    df: DataFrame, 
    id_vars: Iterable[str],
    value_vars: Iterable[str],
    var_name: str="variable", value_name: str="value") -> DataFrame:
    
    """
    Convert :class:`DataFrame` from wide to long format.
    """
    # Create map<key: value>
    vars_and_vals = create_map(
        list(chain.from_iterable([
            [lit(c), col(c)] for c in value_vars])))
    _tmp = df.select(*id_vars, explode(vars_and_vals)) \
        .withColumnRenamed('key', var_name) \
        .withColumnRenamed('value', value_name)
    return _tmp