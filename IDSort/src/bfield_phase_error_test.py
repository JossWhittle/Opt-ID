import unittest, os, shutil
from collections import namedtuple

import json, h5py, random
import numpy as np

from ..src.magnets import Magnets, MagLists

from ..src.field_generator import generate_reference_magnets,   \
                                 generate_bfield,              \
                                 calculate_bfield_phase_error

from ..src.logging_utils import logging, getLogger, setLoggerLevel
logger = getLogger(__name__)


if __name__ == "__main__":

    # Enable debug logging
    setLoggerLevel(logger, 4)

    rng_seed = 6456
    random.seed(rng_seed)
    test_dir = f'/home/pje39613/work/phase_error_{rng_seed}'
    os.makedirs(test_dir, exist_ok=True)

    # inp == Inputs
    # exp == Expected Outputs
    # obs == Observed Outputs

    inp_path = '/home/pje39613/work/magnets'

    # Prepare input file paths
    inp_json_path   = os.path.join(inp_path, 'cpmu.json')
    inp_mag_path    = os.path.join(inp_path, 'cpmu.mag')
    inp_h5_path     = os.path.join(inp_path, 'cpmu.h5')

    try:

        # Attempt to load the ID json data
        try:
            logger.info('Loading ID info from json [%s]', inp_json_path)
            with open(inp_json_path, 'r') as fp:
                info = json.load(fp)

        except Exception as ex:
            logger.error('Failed to load ID info from json [%s]', inp_json_path, exc_info=ex)
            raise ex

        # Attempt to load the ID's lookup table for the eval points defined in the JSON file
        try:
            logger.info('Loading ID lookup table [%s]', inp_h5_path)
            with h5py.File(inp_h5_path, 'r') as fp:
                lookup = {}
                for beam in info['beams']:
                    logger.debug('Loading beam [%s]', beam['name'])
                    lookup[beam['name']] = fp[beam['name']][...]

        except Exception as ex:
            logger.error('Failed to load ID lookup table [%s]', inp_h5_path, exc_info=ex)
            raise ex

        # Attempt to load the real magnet data
        try:
            logger.info('Loading ID magnets [%s]', inp_mag_path)
            magnet_sets = Magnets()
            magnet_sets.load(inp_mag_path)

        except Exception as ex:
            logger.error('Failed to load ID info from json [%s]', inp_mag_path, exc_info=ex)
            raise ex

        # From loaded data construct a perfect magnet array that the loss will be computed with respect to
        logger.info('Constructing perfect reference magnets to shadow real magnets and ideal bfield')
        ref_magnet_sets  = generate_reference_magnets(magnet_sets)
        ref_magnet_lists = MagLists(ref_magnet_sets)
        ref_magnet_lists.shuffle_all()
        ref_bfield       = generate_bfield(info, ref_magnet_lists, ref_magnet_sets, lookup)

        # Execute the function under test for perfect reference magnets
        obs_ref_phase_error, obs_ref_trajectories = calculate_bfield_phase_error(info, ref_bfield,
            debug_path=os.path.join(test_dir, 'fix_ref.npz'))

        # Execute the function under test for real magnets (with no optimization applied, expect values to be poor)
        magnet_lists = MagLists(magnet_sets)
        magnet_lists.shuffle_all()
        bfield       = generate_bfield(info, magnet_lists, magnet_sets, lookup)
        obs_phase_error, obs_trajectories = calculate_bfield_phase_error(info, bfield,
            debug_path=os.path.join(test_dir, 'fix_real.npz'))

    # Use (except + else) instead of (finally) so that output files can be inspected if the test fails
    except Exception as ex: raise ex
    else:

        pass

