from laboratories import Interpolation


if __name__ == '__main__':
    ipt_data = {
        'input': ['./data', 'wraki utm.txt'],
        'spacing': 0.1,
        'window_type': 1,
        'window_size': 0.2,
        'num_min_points': 2,
        'output': ['./output', 'out.csv'],
        'save_format': 'csv'
    }
    print("Witaj w programie")
    for ipt in ipt_data:
        if ipt is 'input' or ipt is 'output':
            input_tmp = ['', '']
            input_tmp[0] = input(f'Read {ipt}_path (default: {ipt_data[ipt][0]}): ')
            input_tmp[1] = input(f'Read {ipt}_file (default: {ipt_data[ipt][1]}): ')
            ipt_data[ipt][0] = input_tmp[0] if input_tmp[0] != '' else ipt_data[ipt][0]
            ipt_data[ipt][1] = input_tmp[1] if input_tmp[1] != '' else ipt_data[ipt][1]
        else:
            input_tmp = input(f'Read {ipt} (default: {ipt_data[ipt]}): ')
            ipt_data[ipt] = input_tmp if input_tmp != '' else ipt_data[ipt]

    first = Interpolation(
        input=ipt_data['input'],
        spacing=ipt_data['spacing'],
        window_type=ipt_data['window_type'],  # -> 0: square, 1: circle
        window_size=ipt_data['window_size'],
        num_min_points=ipt_data['num_min_points'],
        output=ipt_data['output'],
        save_format=ipt_data['save_format']
    )
    # TODO: user input -> type of method the interpolation(ma, idw, kriging)
    # TODO: user input -> if method is IDW -> set power value
    # TODO: user input -> if method is Kriging -> set method of variogram
