from src.reader import Reader
from src.measurement import Measurement
from src.writer import Writer


def main():
    filepath = 'sample_list.txt'

    # only supported method
    sef_method = 'triangular'

    input_units = {'x': 'µm', 'y': 'µm', 'z': 'µm'}
    # sample_matrix.txt: units -> x, y: µm, z: mm
    # sample_list.txt: units -> x, y, z: µm
    # sample_list_2.txt: units -> x, y, z: µm

    read_options = {
        'delimiter': '\t',
        'header': None,
        'comment': '#',
        'decimal': '.',
        'engine': 'python'
    }

    spatial_info = {'width': 886.59,
                    'height': 662.42}

    reader = Reader(file_structure='auto',  # 'auto' or 'matrix' or 'list'
                    input_units=input_units,
                    convert_units=True,
                    read_options=read_options,
                    spatial_info=spatial_info)

    data, settings = reader.read(filepath)
    meas = Measurement(data, settings)
    meas.correct_bending(method='mean')
    meas.calculate_surface_area(method=sef_method)
    meas.prompt_results()
    Writer(meas, method=sef_method)
    meas.plot_topography(show=False)


if __name__ == '__main__':
    main()
