"""
Utility script to convert model weights files to be compatible across multiple ML frameworks
"""
from scipy.io import loadmat


# Reference: https://sefiks.com/2019/07/15/how-to-convert-matlab-models-to-keras/
def matlab_to_tensorflow(mat_weights_file, tf_model, include_top=True):
    model = loadmat(mat_weights_file, matlab_compatible=False, struct_as_record=False)
    net = model['net'][0][0]
    model_layers = net.layers
    model_layers = model_layers[0]
    num_model_layers = model_layers.shape[0]

    # model compiled in Tensorflow
    model_layer_names = [layer.name for layer in tf_model.layers]

    for i in range(num_model_layers):
        model_layer = model_layers[i][0, 0].name[0]
        if model_layer in model_layer_names:
            if include_top:
                if 'conv' in model_layer or 'fc' in model_layer:
                    print(f'{i}. {model_layer}')
                    model_index = model_layer_names.index(model_layer)
                    weights = model_layers[i][0, 0].weights[0, 0]
                    bias = model_layers[i][0, 0].weights[0, 1]
                    tf_model.layers[model_index].set_weights([weights, bias[:, 0]])
            else:
                if 'conv' in model_layer:
                    print(f'{i}. {model_layer}')
                    model_index = model_layer_names.index(model_layer)
                    weights = model_layers[i][0, 0].weights[0, 0]
                    bias = model_layers[i][0, 0].weights[0, 1]
                    tf_model.layers[model_index].set_weights([weights, bias[:, 0]])

    tf_model.save_weights('vgg-face-weights-no-top.h5')
    # print(model_weights)


def matlab_to_pytorch():
    pass


def tensorflow_to_pytorch():
    pass


def pytorch_to_tensorflow():
    pass