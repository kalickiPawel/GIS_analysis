from laboratories import Interpolation


if __name__ == '__main__':
    first = Interpolation(
        data_path="./data",
        filename='wraki utm.txt',
        spacing=0.1,
        window_type=0,
        window_size=0.1,
        num_min_points=2
    )
    # print(vars(first))

