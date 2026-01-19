import pickle


def save_to(parameters, layers_dims, filename):
    with open(filename, 'wb') as f:
        pickle.dump({
            'parameters': parameters,
            'layer_dims': layers_dims
        }, f)

    print(f"Parameters saved to {filename}.")


def load_from(filename):
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
