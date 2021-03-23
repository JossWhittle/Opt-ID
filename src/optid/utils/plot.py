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
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Opt-ID Imports
from ..constants import VECTOR_X, VECTOR_Z, VECTOR_S, VECTOR_ZERO
from ..core.affine import transform_points, transform_rescaled_vectors
from ..device import MagnetSlot #Device

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
def plot_geometry(ax, vertices, polyhedra, color, alpha: numbers.Real, edgecolor = 'k', axes3d: bool = False):
    if axes3d:
        ax.add_collection3d(Poly3DCollection(
            [[vertices[vertex, [0, 2, 1]] for vertex in face] for faces in polyhedra for face in faces],
            facecolors=[color], edgecolors=[edgecolor], linewidth=0.5, alpha=alpha))
        ax.plot(*vertices[:, [0, 2, 1]].T, ' .', color='k', alpha=0)
    else:

        s0, z0 = np.min(vertices, axis=0)[[2, 1]]
        s1, z1 = np.max(vertices, axis=0)[[2, 1]]

        ax.add_collection(PatchCollection(
                [Polygon([[s0, z0], [s1, z0], [s1, z1], [s0, z1]])],
                facecolors=[color], edgecolors=[edgecolor], linewidth=0.5, alpha=alpha))

        # ax.add_collection(PatchCollection(
        #     [Polygon([vertices[vertex, [2, 1]] for vertex in face]) for faces in polyhedra for face in faces],
        #     facecolors=[color], edgecolors=[edgecolor], linewidth=0.5, alpha=alpha))
        ax.plot(*vertices[:, [2, 1]].T, ' .', color='k', alpha=0)


@beartype
def plot_device(
        device,
        *args,
        beams: typ.Optional[typ.Sequence[str]] = None,
        cmap=plt.get_cmap('tab10'),
        ax=None,
        edgecolor = 'k',
        alpha: numbers.Real = 1,
        axes3d: bool = False,
        **kargs):

    if ax is None:
        ax = plt.gca()

    colors = dict()
    legend = list()
    for beam in device.beams.values():

        if (beams is not None) and (beam.name not in beams):
            continue

        for idx, slot in enumerate(beam.slots):

            color_key = f'{slot.slot_type.qualified_name}'
            if color_key in colors:
                color = colors[color_key]
            else:
                color = colors[color_key] = cmap(len(colors))

                label = f'{slot.slot_type.qualified_name}' \
                    if isinstance(slot, MagnetSlot) else slot.slot_type.qualified_name

                legend += [mpatches.Patch(color=color, label=label)]

            geometry = slot.geometry.transform(slot.world_matrix(*args, **kargs))

            plot_geometry(ax=ax, vertices=geometry.vertices, polyhedra=geometry.polyhedra,
                          color=color, edgecolor=edgecolor, alpha=alpha, axes3d=axes3d)

    ax.legend(handles=legend)


@beartype
def plot_device_direction_matrices(
        device,
        *args,
        beams: typ.Optional[typ.Sequence[str]] = None,
        ax=None,
        axes3d: bool = False,
        **kargs):

    if ax is None:
        ax = plt.gca()

    x_color, z_color, s_color = 'rgb'

    def plot_vector(vertices, color, style):
        if axes3d:
            ax.plot(*vertices[:, [0, 2, 1]].T, style, color=color, alpha=1)
        else:
            ax.plot(*vertices[:, [2, 1]].T, style, color=color, alpha=1)

    for beam in device.beams.values():

        if (beams is not None) and (beam.name not in beams):
            continue

        for idx, slot in enumerate(beam.slots):

            if not isinstance(slot, MagnetSlot):
                continue

            matrix = slot.world_matrix(*args, **kargs)
            origin = transform_points(VECTOR_ZERO, matrix)

            bmin, bmax = slot.slot_type.bounds
            size = np.min(bmax - bmin) / 2.2

            plot_vector(np.stack([origin, origin + (transform_rescaled_vectors(slot.magnet.vector, matrix) * size)], axis=0), 'k', '-o')
            plot_vector(np.stack([origin, origin + (transform_rescaled_vectors(VECTOR_X, matrix) * size)], axis=0),
                        x_color, '-.')
            plot_vector(np.stack([origin, origin + (transform_rescaled_vectors(VECTOR_Z, matrix) * size)], axis=0),
                        z_color, '-.')
            plot_vector(np.stack([origin, origin + (transform_rescaled_vectors(VECTOR_S, matrix) * size)], axis=0),
                        s_color, '-.')

    ax.legend(handles=[mpatches.Patch(color=x_color, label=f'X'),
                       mpatches.Patch(color=z_color, label=f'Z'),
                       mpatches.Patch(color=s_color, label=f'S')])
