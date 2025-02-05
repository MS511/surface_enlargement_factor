"""
writer.py

The Writer class appends surface analysis results to a text file in the 'output' folder.
If the folder or file do not exist, it tries to create them.
"""

import os


class Writer:
    def __init__(self, measurement, method: str):
        self.measurement = measurement
        self.add_to_results(method)

    def _create_path_res(self):
        path_src = self.measurement.settings['f_name']
        path_res = os.path.join('output', *path_src.split('/')[1:-1])
        os.makedirs(path_res, exist_ok=True)

    def _create_file_res(self):
        path_src = self.measurement.settings['f_name']
        path_res = os.path.join('output', *path_src.split('/')[1:-1], 'results.txt')
        header = "area_proj\tarea_calc\tsesf\n"
        with open(path_res, 'w+', encoding='utf-8') as f_res:
            f_res.write(header)

    def add_to_results(self, method: str):
        path_src = self.measurement.settings['f_name']
        path_res = os.path.join('output', *path_src.split('/')[1:-1], 'results.txt')

        if not os.path.exists(os.path.dirname(path_res)):
            try:
                self._create_path_res()
            except OSError as e:
                print(f"[Error] Could not create directory '{os.path.dirname(path_res)}': {e}")
                return

        if not os.path.isfile(path_res):
            self._create_file_res()

        results = self.measurement.results
        if method not in results:
            print(f"[Error] Method '{method}' not found in measurement results.")
            return

        area_calc = results[method][0]
        area_proj = results[method][1]
        sesf = results[method][2]

        line_to_add = f"{area_proj:8.4g}\t{area_calc:8.4g}\t{sesf:8.4g}\n"

        try:
            with open(path_res, 'a', encoding='utf-8') as f_res:
                f_res.write(line_to_add)
        except OSError as e:
            print(f"[Error] Could not write to file '{path_res}': {e}")
