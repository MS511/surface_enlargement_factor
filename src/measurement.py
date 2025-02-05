"""
measurement.py

Contains the 'Measurement' class, which represents a 2D topography
and provides surface area calculations, bending corrections, and plotting routines.
Internally uses dimensionless or consistent scales if convert_units=True.
Output is labeled with placeholders [unit_x], [unit_y], [unit_z].
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
        try:
            self.width_x = self.settings['width'] / self.data.shape[0]
            self.width_y = self.settings['height'] / self.data.shape[1]
            self.projected_area = self.settings['width'] * self.settings['height']
        except KeyError as e:
            print(f"[Warning] Missing key {e} in settings. Fallback: 1 per row/column.")
            self.width_x = 1.0
            self.width_y = 1.0
            self.projected_area = float(self.data.shape[0] * self.data.shape[1])

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
        if method == 'linear':
            self._surface_area_linear()
        elif method == 'triangular':
            self._surface_area_triangular_optimized()
        else:
            raise ValueError(f"[Error] Unknown method '{method}'. Valid: 'linear'/'triangular'.")

    def _surface_area_linear(self):
        width_x = self.width_x
        width_y = self.width_y
        projected_area = self.projected_area

        arr = self.data.to_numpy()

        diff_x = np.abs(np.diff(arr, axis=0))
        additional_surf_x = np.nansum(diff_x) * width_x

        diff_y = np.abs(np.diff(arr, axis=1))
        additional_surf_y = np.nansum(diff_y) * width_y

        total_area = projected_area + additional_surf_x + additional_surf_y
        enlargement_factor = total_area / projected_area

        self.results['linear'] = [total_area, projected_area, enlargement_factor]

    def _surface_area_triangular_optimized(self):
        width_x = self.width_x
        width_y = self.width_y
        df = self.data

        df_extended = pd.concat([df.iloc[:, 0], df], axis=1, ignore_index=True)
        df_extended = pd.concat([df_extended, df.iloc[:, -1]], axis=1, ignore_index=True)
        df_extended.loc[-1] = df_extended.loc[0]
        df_extended.index = df_extended.index + 1
        df_extended.sort_index(inplace=True)

        n_x, n_y = df_extended.shape
        x = np.arange(n_x) * width_x
        y = np.arange(n_y) * width_y

        x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
        positions = np.dstack((x_mesh, y_mesh, df_extended.to_numpy()))

        idx_i, idx_j = np.meshgrid(range(1, n_x), range(1, n_y), indexing='ij')

        pos_center = positions[idx_i, idx_j]
        pos_ul = positions[idx_i - 1, idx_j - 1]
        pos_uc = positions[idx_i, idx_j - 1]
        pos_lc = positions[idx_i - 1, idx_j]

        vec1_1 = pos_uc - pos_center
        vec2_1 = pos_ul - pos_center
        cross1 = np.cross(vec1_1, vec2_1)

        vec1_2 = pos_lc - pos_center
        vec2_2 = pos_ul - pos_center
        cross2 = np.cross(vec1_2, vec2_2)

        area1 = 0.5 * np.linalg.norm(cross1, axis=2)
        area2 = 0.5 * np.linalg.norm(cross2, axis=2)
        total_area = np.nansum(area1 + area2)

        projected_area = self.projected_area
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
