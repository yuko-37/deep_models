import pickle
import os


def save_to_file(parameters, layers_dims, filename='parameters/parameters.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({
            'parameters': parameters,
            'layer_dims': layers_dims
        }, f)

    print(f"Parameters saved to {filename}.")


def load_from_file(filename='parameters/parameters.pkl'):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        parameters = data['parameters']
        layers_dims = data['layer_dims']

        print(f"Parameters loaded from {filename}.")
        return parameters
    except Exception as e:
        print(f"Error loading parameters from {filename}: {e}")
        raise e


def parameters_dont_exist(filename='parameters/parameters.pkl'):
    return not os.path.isfile(filename)
