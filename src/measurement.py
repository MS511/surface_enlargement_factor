"""
measurement.py

Contains the 'Measurement' class, which represents a 2D topography
and provides surface area calculations, bending corrections, and plotting routines.
Internally uses dimensionless or consistent scales if convert_units=True.
Output is labeled with placeholders [unit_x], [unit_y], [unit_z].

Coordinate convention (internal):
- data shape = (n_rows, n_cols)
- x-axis spans columns (n_cols), total extent = settings['width']
- y-axis spans rows (n_rows), total extent = settings['height']
- grid spacings: dx = width/(n_cols-1), dy = height/(n_rows-1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import cmasher as cmr
import matplotlib

matplotlib.rcParams['font.size'] = 11


class Measurement:
    def __init__(self, data: pd.DataFrame, settings: dict):
        self.data = data.fillna(data.mean())
        self.settings = settings
        self._get_width_and_projected_area()
        self.results = {}

    def _get_width_and_projected_area(self):
        """Derive grid spacing from total physical extents.

        settings['width'] / ['height'] are interpreted as total extents (first-to-last point).
        Therefore dx/dy use (N-1) intervals.
        """
        try:
            n_rows, n_cols = self.data.shape
            if n_rows < 2 or n_cols < 2:
                raise ValueError("Need at least 2x2 data to define dx/dy from extents.")

            total_width = float(self.settings['width'])
            total_height = float(self.settings['height'])

            # x spans columns, y spans rows
            self.width_x = total_width / (n_cols - 1)
            self.width_y = total_height / (n_rows - 1)
            self.projected_area = total_width * total_height
        except KeyError as e:
            print(f"[Warning] Missing key {e} in settings. Fallback: dx=dy=1 and projected area from shape.")
            n_rows, n_cols = self.data.shape
            self.width_x = 1.0
            self.width_y = 1.0
            self.projected_area = float((n_cols - 1) * (n_rows - 1)) if n_rows > 1 and n_cols > 1 else float(n_rows * n_cols)
        except ValueError as e:
            print(f"[Warning] {e} Fallback: dx=dy=1 and projected area from shape.")
            n_rows, n_cols = self.data.shape
            self.width_x = 1.0
            self.width_y = 1.0
            self.projected_area = float((n_cols - 1) * (n_rows - 1)) if n_rows > 1 and n_cols > 1 else float(n_rows * n_cols)

    def correct_bending(self, method: str = 'mean', poly_degree: int = 2):
        if self.data.shape[0] < 2 or self.data.shape[1] < 2:
            print("[Error] Bending correction not meaningful for data < 2x2.")
            return

        y_mean = self.data.mean(axis=1)

        if method == 'mean':
            correction = np.vstack([y_mean.values for _ in range(self.data.shape[1])]).T
            self.data = self.data - correction

        elif method == 'poly':
            x_idx = np.arange(len(y_mean))
            try:
                p = np.polyfit(x_idx, y_mean, poly_degree)
                y_fit = np.polyval(p, x_idx)
                correction = np.vstack([y_fit for _ in range(self.data.shape[1])]).T
                self.data = self.data - correction
            except np.linalg.LinAlgError as err:
                print(f"[Error] Polynomial fitting failed: {err}")
                return
        else:
            print(f"[Error] Unknown bending-correction method: '{method}'. Use 'mean' or 'poly'.")
            return

        overall_min = self.data.min().min()
        self.data = self.data - overall_min

    def calculate_surface_area(self, method: str = 'triangular'):
        if method == 'triangular':
            self._surface_area_triangular_mean_diagonals()
        else:
            raise ValueError(f"[Error] Unknown method '{method}'. Valid: 'triangular'.")

    @staticmethod
    def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Vectorized triangle area via cross product: 0.5*|| (b-a) x (c-a) ||."""
        cross = np.cross(b - a, c - a)
        return 0.5 * np.linalg.norm(cross, axis=-1)

    def _surface_area_triangular_mean_diagonals(self):
        """Surface area via triangulation, averaging both possible diagonals per cell.

        For each grid cell with corners:
          p00 (row i, col j), p10 (i+1,j), p01 (i,j+1), p11(i+1,j+1)
        we compute:
          diag A: triangles (p00,p10,p11) + (p00,p11,p01)
          diag B: triangles (p00,p10,p01) + (p10,p11,p01)
        and take the mean of both cell areas.

        This reduces directional bias compared to a single fixed diagonal.
        """
        df = self.data
        n_rows, n_cols = df.shape
        if n_rows < 2 or n_cols < 2:
            raise ValueError("[Error] Need at least 2x2 data for triangular surface area.")

        dx = float(self.width_x)
        dy = float(self.width_y)

        z = df.to_numpy(dtype=float)

        # Coordinates: x along columns, y along rows
        x_coords = np.arange(n_cols) * dx
        y_coords = np.arange(n_rows) * dy
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing='xy')
        positions = np.dstack((x_mesh, y_mesh, z))  # shape (n_rows, n_cols, 3)

        # Cell corner arrays (each shape (n_rows-1, n_cols-1, 3))
        p00 = positions[:-1, :-1]
        p10 = positions[1:, :-1]
        p01 = positions[:-1, 1:]
        p11 = positions[1:, 1:]

        # Diagonal A (p00->p11)
        area_a = self._triangle_area(p00, p10, p11) + self._triangle_area(p00, p11, p01)
        # Diagonal B (p10->p01)
        area_b = self._triangle_area(p00, p10, p01) + self._triangle_area(p10, p11, p01)

        cell_area = 0.5 * (area_a + area_b)
        total_area = float(np.nansum(cell_area))

        projected_area = float(self.projected_area)
        enlargement_factor = total_area / projected_area

        self.results['triangular'] = [total_area, projected_area, enlargement_factor]

    def prompt_results(self):
        file_name = self.settings.get('f_name', 'Unknown')
        print(f"\nAnalyzed file: {file_name}")

        table_data = []
        for method, (area_calc, area_proj, factor) in self.results.items():
            table_data.append([
                method,
                f"{area_calc:8.4g}",
                f"{factor * 100:6.2f}"
            ])

        print(tabulate(
            table_data,
            headers=["Method", "Surface Area", "Enlargement Factor [%]"],
            tablefmt="orgtbl"
        ))

        if self.results:
            area_proj = self.results[list(self.results.keys())[0]][1]
        else:
            area_proj = self.projected_area

        print(f"The projected area is {area_proj:.8f} [unit_x]*[unit_y].")

    def plot_topography(self, show: bool = False):
        data = self.data
        w = self.settings.get('width', data.shape[1])
        h = self.settings.get('height', data.shape[0])

        if data.shape[0] > 1 and data.shape[1] > 1:
            data_plot = data.iloc[:-1, :-1]
        else:
            data_plot = data

        n_x, n_y = data_plot.shape
        x_vals = np.linspace(0, w, n_y)
        y_vals = np.linspace(0, h, n_x)
        x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')

        z_array = data_plot.to_numpy()
        z_min = np.nanmin(z_array)
        adjusted_data = z_array - z_min

        fig, ax = plt.subplots(figsize=(6, 6 / 1.618))
        contour = ax.contourf(x_mesh, y_mesh, adjusted_data.T, levels=100, cmap=cmr.savanna)
        cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.015)

        ax.set_xlabel("x [unit_x]")
        ax.set_ylabel("y [unit_y]")
        cbar.set_label("z [unit_z]")

        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.close()
