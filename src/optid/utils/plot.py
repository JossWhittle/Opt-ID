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
from beartype import beartype
import numpy as np
import numbers
import typing as typ
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Opt-ID Imports
from ..device import \
    Device, Beam


TLimit = typ.Optional[typ.Tuple[numbers.Real, numbers.Real, int]]


@beartype
def setup_axes3d(fig, rows: int, cols: int, subplot: int, title: str = '', proj: str = 'persp',
                 elev: int = 10, azim: int = -80,
                 x: TLimit = (-50, 50, 5), z: TLimit = (-50, 50, 5), s: TLimit = (-50, 50, 5)):

    ax = fig.add_subplot(rows, cols, subplot, projection='3d')
    ax.set_proj_type(proj)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])
    ax.xaxis.pane.fill, ax.yaxis.pane.fill, ax.zaxis.pane.fill = False, False, False
    ax.set_xlabel('X Axis')
    ax.set_zlabel('Z Axis (Transverse)')
    ax.set_ylabel('S Axis')
    ax.set_title(title, fontdict={'size': 22})

    if x is not None:
        ax.set_xlim3d(*x[:2])
        ax.set_xticks(np.linspace(*x))

    if z is not None:
        ax.set_zlim3d(*z[:2])
        ax.set_zticks(np.linspace(*z))

    if s is not None:
        ax.set_ylim3d(*s[:2])
        ax.set_yticks(np.linspace(*s))

    return ax


@beartype
def plot_geometry(ax, vertices, polyhedra, color, axes3d: bool = False):
    if axes3d:
        ax.add_collection3d(Poly3DCollection(
            [[vertices[vertex, [0, 2, 1]] for vertex in face] for faces in polyhedra for face in faces],
            facecolors=[color], edgecolors=['k'], linewidth=0.5, alpha=1))
        ax.plot(*vertices[:, [0, 2, 1]].T, ' .', color='k', alpha=0)
    else:
        ax.add_collection(PatchCollection(
            [Polygon([vertices[vertex, [2, 1]] for vertex in face]) for faces in polyhedra for face in faces],
            facecolors=[color], edgecolors=['k'], linewidth=0.5, alpha=1))
        ax.plot(*vertices[:, [2, 1]].T, ' .', color='k', alpha=0)


@beartype
def plot_device(device: Device, *args, ax=None, cmap=plt.get_cmap('tab10'), axes3d: bool = False, **kargs):

    if ax is None:
        ax = plt.gca()

    colors = dict()

    for beam in device.beams.values():
        for idx, slot in enumerate(beam.slots):

            color_key = f'{slot.slot_type.qualified_name}'
            if color_key in colors:
                color = colors[color_key]
            else:
                color = colors[color_key] = cmap(len(colors))

            matrix = slot.world_matrix(*args, **kargs)
            geometry = slot.slot_type.magnet_type.geometry.transform(matrix)

            plot_geometry(ax, geometry.vertices, geometry.polyhedra, color, axes3d)


@beartype
def plot_beam(beam: Beam, *args, ax=None, cmap=plt.get_cmap('tab10'), axes3d: bool = False, **kargs):

    if ax is None:
        ax = plt.gca()

    colors = dict()

    for idx, slot in enumerate(beam.slots):

        color_key = f'{slot.slot_type.qualified_name}'
        if color_key in colors:
            color = colors[color_key]
        else:
            color = colors[color_key] = cmap(len(colors))

        matrix = slot.world_matrix(*args, **kargs)
        geometry = slot.slot_type.magnet_type.geometry.transform(matrix)

        plot_geometry(ax, geometry.vertices, geometry.polyhedra, color, axes3d)
