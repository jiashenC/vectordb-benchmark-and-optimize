import h5py
import pandas as pd

from evadb.catalog.catalog_type import ColumnType, NdArrayType
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe


class HDF5Reader(AbstractFunction):
    @setup(cacheable=False, function_type="Reader", batchable=False)
    def setup(self):
        self.f = h5py.File("./sift-128-euclidean.hdf5")

    @property
    def name(self):
        return "HDF5Reader"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["key", "index"],
                column_types=[NdArrayType.STR, NdArrayType.STR],
                column_shapes=[(1,), (1,)]
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["feature"],
                column_types=[
                    NdArrayType.FLOAT32
                ],
                column_shapes=[(1, 128)]
            )
        ],
    )
    def forward(self, input_df):
        key_df = input_df.iloc[:, 0]
        index_df = input_df.iloc[:, 1]

        feature_list = []
        for key, index in zip(key_df, index_df):
            index = int(index)
            feature = self.f[key][index]
            feature = feature.reshape(1, -1)
            feature_list.append(feature)

        df = pd.DataFrame({"feature": feature_list})
        return df
