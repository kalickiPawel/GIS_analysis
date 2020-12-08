from laboratories import Interpolation


if __name__ == '__main__':
    ipt_data = {
        'data_path': './data',
        'filename': 'wraki utm.txt',
        'spacing': 0.1,
        'window_type': 1,
        'window_size': 0.2,
        'num_min_points': 2
    }
    print("Witaj w programie")
    for ipt in ipt_data:
        input_tmp = input(f'Read {ipt} (default: {ipt_data[ipt]}): ')
        ipt_data[ipt] = input_tmp if input_tmp != '' else ipt_data[ipt]

    first = Interpolation(
        data_path=ipt_data['data_path'],
        filename=ipt_data['filename'],
        spacing=ipt_data['spacing'],
        window_type=ipt_data['window_type'],  # -> 0: square, 1: circle
        window_size=ipt_data['window_size'],
        num_min_points=ipt_data['num_min_points']
    )
