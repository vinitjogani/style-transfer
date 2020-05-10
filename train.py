from loss import loss
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
import tensorflow as tf
from load import load_images, deprocess


model = vgg19.VGG19(weights='imagenet', include_top=False)
feature_map = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=dict([(l.name, l.output) for l in model.layers])
)
style_layer_names = [
    'block1_conv1', 'block2_conv1',
    'block3_conv1', 'block4_conv1',
    'block5_conv1'
]
content_layer_name = 'block5_conv2'


@tf.function
def grad(base, style, combo):
    with tf.GradientTape() as tape:
        l = loss(base, style, combo, style_layer_names, content_layer_name, feature_map)
    return l, tape.gradient(l, combo)


if __name__ == '__main__':

    base_input, style_input = load_images("base.png", "style.jpg")

    combo = tf.Variable(base_input)
    optimizer = tf.optimizers.SGD(
        tf.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1.,
            decay_steps=100,
            decay_rate=0.99
        )
    )

    for i in range(5001):
        l, g = grad(base_input, style_input, combo)
        optimizer.apply_gradients([(g, combo)])

        if (i+1) % 100 == 0:
            plt.imshow(deprocess(combo.numpy()))
            plt.show()

            print(f"Iteration {i}, loss={l.numpy()}")
