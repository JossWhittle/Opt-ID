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
import jax
import jax.numpy as jnp


@jax.jit
def transform_points(lattice, matrix):
    """
    Apply a 4x4 affine transformation to a lattice of points in XZS.

    :param lattice:
        A tensor of points in XZS.

    :param matrix:
        A single 4x4 affine matrix.

    :return:
        A tensor of points in XZS.
    """
    lattice = jnp.concatenate([lattice, jnp.ones(lattice.shape[:-1] + (1,))], axis=-1)
    return (lattice @ matrix)[..., :-1]


@jax.jit
def transform_vectors(lattice, matrix):
    """
    Apply a 4x4 affine transformation to a lattice of vectors in XZS.

    :param lattice:
        A tensor of vectors in XZS.

    :param matrix:
        A single 4x4 affine matrix.

    :return:
        A tensor of vectors in XZS.
    """
    lattice = jnp.concatenate([lattice, jnp.zeros(lattice.shape[:-1] + (1,))], axis=-1)
    return (lattice @ matrix)[..., :-1]


@jax.jit
def radians(degrees):
    """
    Convert degrees to radians.

    :param degrees:
        Angle in degrees.

    :return:
        Angle in radians.
    """
    return degrees * (jnp.pi / 180.0)


@jax.jit
def rotate_x(theta):
    """
    Create a 4x4 affine matrix representing a rotation on the X-axis.

    :param theta:
        Angle in radians to rotate by.

    :return:
        An 4x4 affine matrix.
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[ 1,  0,  0,  0],
                      [ 0,  c, -s,  0],
                      [ 0,  s,  c,  0],
                      [ 0,  0,  0,  1]],
                     dtype=jnp.float32).T


@jax.jit
def rotate_z(theta):
    """
    Create a 4x4 affine matrix representing a rotation on the Z-axis.

    :param theta:
        Angle in radians to rotate by.

    :return:
        An 4x4 affine matrix.
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[ c,  0,  s,  0],
                      [ 0,  1,  0,  0],
                      [-s,  0,  c,  0],
                      [ 0,  0,  0,  1]],
                     dtype=jnp.float32).T


@jax.jit
def rotate_s(theta):
    """
    Create a 4x4 affine matrix representing a rotation on the S-axis.

    :param theta:
        Angle in radians to rotate by.

    :return:
        An 4x4 affine matrix.
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[ c, -s,  0,  0],
                      [ s,  c,  0,  0],
                      [ 0,  0,  1,  0],
                      [ 0,  0,  0,  1]],
                     dtype=jnp.float32).T


@jax.jit
def scale(x, z, s):
    """
    Create a 4x4 affine matrix representing a set of orthogonal scale transformations.

    :param x:
        Scaling coefficient on the X-axis.

    :param z:
        Scaling coefficient on the Z-axis.

    :param s:
        Scaling coefficient on the S-axis.

    :return:
        An 4x4 affine matrix.
    """
    return jnp.array([[x, 0, 0, 0],
                      [0, z, 0, 0],
                      [0, 0, s, 0],
                      [0, 0, 0, 1]],
                     dtype=jnp.float32).T


@jax.jit
def translate(x, z, s):
    """
    Create a 4x4 affine matrix representing a set of orthogonal translation transformations.

    :param x:
        Translation offset on the X-axis.

    :param z:
        Translation offset on the Z-axis.
        
    :param s:
        Translation offset on the S-axis.

    :return:
        An 4x4 affine matrix.
    """
    return jnp.array([[1, 0, 0, x],
                      [0, 1, 0, z],
                      [0, 0, 1, s],
                      [0, 0, 0, 1]],
                     dtype=jnp.float32).T
