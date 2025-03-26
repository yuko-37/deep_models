import h5py
import os
import numpy as np
from typing import Union


def load_datasets():
    train_ds = h5py.File('datasets/train_signs.h5')
    train_x = train_ds['train_set_x']
    train_y = train_ds['train_set_y']

    test_ds = h5py.File('datasets/test_signs.h5')
    test_x = test_ds['test_set_x']
    test_y = test_ds['test_set_y']

    return train_x, train_y, test_x, test_y


def print_attrs(name, group):
    if len(group.attrs) > 0:
        print(f"Attributes for {name}: ")
        for key, val in group.attrs.items():
            print(f"  {key}: {val}")
    # else:
        # print(f"No attributes for {name}")


def explore_group(group: Union[h5py.File, h5py.Group], prefix="",
                  level=0, max_level=3):
    if level > max_level:
        print(f"Max recursion level reached")
        return

    indent = "  " * level
    print_attrs(prefix, group)

    for name, item in group.items():
        item_path = f"{prefix}/{name}" if prefix else name

        if isinstance(item, h5py.Group):
            print(f"{indent}Group: {item_path}/")
            explore_group(item, item_path, level+1)

        elif isinstance(item, h5py.Dataset):
            dataset_info = f"{indent}Dataset: {item_path}"
            dataset_info += f", Shape: {item.shape}, Type: {item.dtype}"

            if item.chunks:
                dataset_info += f", Chunks: {item.chunks}"
            if item.compression:
                dataset_info += f", Compression: {item.compression}"

            print(dataset_info)

            print_attrs(item_path, item)

        else:
            print(f"{indent}Unknown item: {item_path}, Type: {type(item)}")


def analyze_h5_file(filepath):
    try:
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"File: {filepath}")
        print(f"Size: {file_size:.2f} MB")

        with h5py.File(filepath) as file:
            print("\nFile Structure:")
            explore_group(file)

            groups = []
            datasets = []

            def count_items(name, obj):
                if isinstance(obj, h5py.Group) and name != '':
                    groups.append(name)
                elif isinstance(obj, h5py.Dataset):
                    datasets.append(name)

            file.visititems(count_items)

            print(f"\nSummary:")
            print(f"  Total groups: {len(groups)}")
            print(f"  Total datasets: {len(datasets)}")

            total_size = 0
            first_items = []

            for name in datasets:
                dataset = file[name]
                size = np.prod(dataset.shape) * dataset.dtype.itemsize
                total_size += size
                print(f"  {name}: {size / (1024 * 1024):.2f} MB")
                first_items.append((name, dataset[0]))

            print(f"\n  Total data size: {total_size / (1024 * 1024):.2f} MB\n")


    except Exception as e:
        print(f"Error analyzing file: {str(e)}")


if __name__ == '__main__':
    analyze_h5_file('datasets/test_signs.h5')

