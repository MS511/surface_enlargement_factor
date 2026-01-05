"""
reader.py

This module provides the Reader class which loads topography data
from either a list-based file (x, y, z columns) or a matrix-based file (z only).
It can optionally convert data from certain physical units into a consistent scale.
"""

import pandas as pd
import numpy as np

UNIT_MAPPING = {
    'm': 1.0,
    'meter': 1.0,
    'meters': 1.0,
    'mm': 1e-3,
    'millimeter': 1e-3,
    'millimeters': 1e-3,
    'µm': 1e-6,
    'micrometer': 1e-6,
    'micrometers': 1e-6,
    'nm': 1e-9,
    'nanometer': 1e-9,
    'nanometers': 1e-9
}


class Reader:
    def __init__(
            self,
            file_structure='auto',
            input_units=None,
            convert_units=True,
            spatial_info=None,
            config=None,
            read_options=None
    ):
        self.file_structure = file_structure
        self.input_units = input_units or {}
        self.convert_units = convert_units
        self.spatial_info = spatial_info or {}
        self.config = config or {}
        self.read_options = read_options or {}
        self.data = None
        self.settings = {}

    def _convert_value(self, value: float, unit_str: str) -> float:
        if not self.convert_units:
            return value
        factor = UNIT_MAPPING.get(unit_str.lower(), 1.0)
        return value * factor

    def read(self, f_in: str):
        if self.file_structure == 'auto':
            if 'matrix' in f_in.lower():
                structure = 'matrix'
            else:
                structure = 'list'
        else:
            structure = self.file_structure

        default_opts = {
            'delimiter': '\t',
            'header': None,
            'comment': '#'
        }
        options = {**default_opts, **self.read_options}

        try:
            df = pd.read_csv(f_in, **options)
        except Exception as e:
            msg = (
                f"[Error] Could not read file '{f_in}' with options {options}. "
                f"Cause: {e}"
            )
            raise ValueError(msg)

        if df.shape[1] == 0:
            try:
                with open(f_in, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
                skiprows = options.get('skiprows', 0)
                lines = lines[skiprows:]
                delim = options.get('delimiter', ';')
                data_list = [l.split(delim) for l in lines]
                df = pd.DataFrame(data_list)
            except Exception as e:
                msg = f"[Error] Fallback read also failed for file '{f_in}'. Cause: {e}"
                raise ValueError(msg)

        df = df.apply(pd.to_numeric, errors='coerce').dropna(how='any')

        if structure == 'list':
            if df.shape[1] != 3:
                raise ValueError(
                    f"[Error] Expected exactly 3 columns for list input, got {df.shape[1]}."
                )
            df.columns = ['x', 'y', 'z']
            df['x'] = df['x'] - df['x'].min()
            df['y'] = df['y'] - df['y'].min()
            df['z'] = df['z'] - df['z'].min()

            for col in ['x', 'y', 'z']:
                unit_str = self.input_units.get(col, 'm')
                df[col] = df[col].apply(lambda val: self._convert_value(val, unit_str))

            width = df['x'].max()
            height = df['y'].max()
            self.settings['width'] = width
            self.settings['height'] = height

            pivot_table = df.pivot(index='y', columns='x', values='z')
            pivot_table.ffill(inplace=True)
            pivot_table.bfill(inplace=True)

            self.data = pivot_table

        elif structure == 'matrix':
            mat = df.to_numpy().astype(float)
            unit_z = self.input_units.get('z', 'm')
            mat = np.vectorize(lambda val: self._convert_value(val, unit_z))(mat)
            df_z = pd.DataFrame(mat)
            self.data = df_z

            num_rows, num_cols = df_z.shape

            if 'width' in self.spatial_info and 'height' in self.spatial_info:
                width = self.spatial_info['width']
                height = self.spatial_info['height']
                if self.convert_units:
                    x_unit = self.input_units.get('x', 'm')
                    y_unit = self.input_units.get('y', 'm')
                    width = self._convert_value(width, x_unit)
                    height = self._convert_value(height, y_unit)
            elif 'delta_x' in self.spatial_info and 'delta_y' in self.spatial_info:
                dx = self.spatial_info['delta_x']
                dy = self.spatial_info['delta_y']
                if self.convert_units:
                    x_unit = self.input_units.get('x', 'm')
                    y_unit = self.input_units.get('y', 'm')
                    dx = self._convert_value(dx, x_unit)
                    dy = self._convert_value(dy, y_unit)

                # Interpret delta_x/delta_y as grid spacing between points.
                # Total extent is (N-1) * delta.
                width = (num_cols - 1) * dx if num_cols > 1 else 0.0
                height = (num_rows - 1) * dy if num_rows > 1 else 0.0
            else:
                raise ValueError(
                    "[Error] For matrix input, provide either {width, height} or {delta_x, delta_y}."
                )

            self.settings['width'] = width
            self.settings['height'] = height
        else:
            raise ValueError(f"[Error] Invalid file_structure '{structure}'. Must be 'list' or 'matrix'.")

        self.settings['f_name'] = f_in
        self.settings['input_units'] = self.input_units
        self.settings['units'] = self.input_units

        return self.data, self.settings
