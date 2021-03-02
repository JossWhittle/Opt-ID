# Copyright 2017 Diamond Light Source
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


# External Imports
import numbers
from beartype import beartype
import typing as typ
import jax.numpy as jnp
import pandas as pd
import pandera as pa

# Opt-ID Imports


TVector = typ.Union[jnp.ndarray, typ.Sequence[numbers.Real]]


class Candidate:

    @beartype
    def __init__(self,
            name: str,
            vector: TVector):
        """
        Construct a Candidate instance.

        :param name:
            String name for the candidate.

        :param vector:
            Field vector for the magnet.
        """

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        if not isinstance(vector, jnp.ndarray):
            vector = jnp.array(vector, dtype=jnp.float32)

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != jnp.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        self._vector = vector

    @staticmethod
    @beartype
    def from_dataframe(
            df: pd.DataFrame,
            name: str = 'name',
            x: str = 'x',
            z: str = 'z',
            s: str = 's'):
        """
        Parse pandas Dataframe to produce a list of candidates.

        :param df:
            Pandas Dataframe with the name and vector data.

        :param name:
            String column name for the name field.

        :param x:
            String column name for the x vector field.

        :param z:
            String column name for the z vector field.

        :param s:
            String column name for the s vector field.

        :return:
            List of Candidate instances.
        """

        schema = pa.DataFrameSchema({
            name: pa.Column(pa.String, pa.Check((lambda col: (len(col.unique()) == len(col))),
                                                error='candidate names must be unique'), coerce=True),
            x: pa.Column(pa.Float, coerce=True),
            z: pa.Column(pa.Float, coerce=True),
            s: pa.Column(pa.Float, coerce=True)
        })

        df = schema.validate(df)

        return [Candidate(name=row[name], vector=(float(row[x]), float(row[z]), float(row[s])))
                for _, row in df.sort_values(name).iterrows()]

    @property
    @beartype
    def name(self) -> str:
        return self._name

    @property
    @beartype
    def vector(self) -> jnp.ndarray:
        return self._vector
