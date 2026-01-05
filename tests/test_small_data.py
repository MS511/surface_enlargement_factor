"""
test_small_data.py

Tests for small topography data in both list and matrix forms.
Focus: triangular SEF (surface area via triangulation).
"""

import os
import math

import numpy as np
import pandas as pd
import pytest

from src.reader import Reader
from src.measurement import Measurement


@pytest.fixture
def data_path():
    return os.path.join(os.path.dirname(__file__), 'data')


def test_small_list_3x3(data_path):
    file_list_3x3 = os.path.join(data_path, "small_list_3x3.txt")
    reader = Reader(
        file_structure='list',
        input_units={'x': 'm', 'y': 'm', 'z': 'm'},
        convert_units=False,
        read_options={
            'delimiter': r'\s+',
            'header': None,
            'comment': '#',
            'engine': 'python'
        }
    )
    data, settings = reader.read(file_list_3x3)
    meas = Measurement(data, settings)
    meas.correct_bending(method='mean')

    meas.calculate_surface_area(method='triangular')
    assert 'triangular' in meas.results

    tri_area = meas.results['triangular'][0]
    assert tri_area > 0


def test_small_matrix_3x3(data_path):
    file_matrix_3x3 = os.path.join(data_path, "small_matrix_3x3.txt")
    reader = Reader(
        file_structure='matrix',
        input_units={'z': 'm'},
        convert_units=False,
        spatial_info={'width': 2.0, 'height': 3.0},
        read_options={
            'delimiter': r'\s+',
            'header': None,
            'comment': '#',
            'engine': 'python'
        }
    )

    data, settings = reader.read(file_matrix_3x3)
    meas = Measurement(data, settings)
    meas.correct_bending(method='mean')

    meas.calculate_surface_area(method='triangular')
    assert 'triangular' in meas.results

    tri_area = meas.results['triangular'][0]
    assert tri_area > 0


def test_triangular_flat_plane_sef_is_one():
    # 3x4 grid: extents map to projected area width*height
    n_rows, n_cols = 3, 4
    z = np.zeros((n_rows, n_cols), dtype=float)
    data = pd.DataFrame(z)

    settings = {'width': 6.0, 'height': 2.0}
    meas = Measurement(data, settings)
    meas.calculate_surface_area('triangular')

    sef = meas.results['triangular'][2]
    assert sef == pytest.approx(1.0, abs=1e-12)


def test_triangular_inclined_plane_matches_analytic_factor():
    # z(x,y) = a*x + b*y => area scale = sqrt(1+a^2+b^2) relative to projected
    n_rows, n_cols = 20, 30
    width = 3.0
    height = 2.0

    dx = width / (n_cols - 1)
    dy = height / (n_rows - 1)

    a = 0.4
    b = -0.7

    x = np.arange(n_cols) * dx
    y = np.arange(n_rows) * dy
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='xy')
    z = a * x_mesh + b * y_mesh

    data = pd.DataFrame(z)
    settings = {'width': width, 'height': height}
    meas = Measurement(data, settings)
    meas.calculate_surface_area('triangular')

    sef = meas.results['triangular'][2]
    expected = math.sqrt(1.0 + a * a + b * b)

    # triangulation introduces small discretization error; keep tight but realistic tolerance
    assert sef == pytest.approx(expected, rel=5e-4, abs=0.0)
