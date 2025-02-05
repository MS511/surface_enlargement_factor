"""
test_small_data.py

Tests for small 3x3 topography data in both list and matrix forms.
"""

import pytest
import os
from src.reader import Reader
from src.measurement import Measurement


@pytest.fixture
def data_path():
    return os.path.join(os.path.dirname(__file__), 'data')


def test_small_list_3x3(data_path):
    file_list_3x3 = os.path.join(data_path, "small_list_3x3.txt")
    methods = ['linear', 'triangular']
    reader = Reader(file_structure='list',
                    input_units={'x': 'm', 'y': 'm', 'z': 'm'},
                    convert_units=False,
                    read_options={
                        'delimiter': r'\s+',
                        'header': None,
                        'comment': '#',
                        'engine': 'python'
                    })
    data, settings = reader.read(file_list_3x3)
    meas = Measurement(data, settings)
    meas.correct_bending(method='mean')
    for method in methods:
        meas.calculate_surface_area(method=method)
        if method == 'linear':
            assert meas.results[method][-1] == 3.1666666666666665
        elif method == 'triangular':
               assert meas.results[method][-1] == 2.641678499864457

        assert method in meas.results

        tri_area = meas.results[method][0]
        assert tri_area > 0


def test_small_matrix_3x3(data_path):
    file_matrix_3x3 = os.path.join(data_path, "small_matrix_3x3.txt")
    methods = ['linear', 'triangular']
    reader = Reader(file_structure='matrix',
                    input_units={'z': 'm'},
                    convert_units=False,
                    spatial_info={'width': 2.0, 'height': 3.0},
                    read_options={
                        'delimiter': r'\s+',
                        'header': None,
                        'comment': '#',
                        'engine': 'python'
                    })

    data, settings = reader.read(file_matrix_3x3)
    meas = Measurement(data, settings)
    meas.correct_bending(method='mean')
    for method in methods:
        meas.calculate_surface_area(method=method)

        if method == 'linear':
            assert meas.results[method][-1] == 3.4814814814814814
        elif method == 'triangular':
            assert meas.results[method][-1] == 2.8082905685532675

        assert method in meas.results

        tri_area = meas.results[method][0]
        assert tri_area > 0
