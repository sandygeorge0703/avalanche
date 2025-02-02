import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_cnn_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Layer Details: (height, width, channels)
    layers = [
        {'name': 'Input', 'size': (224, 224, 3)},
        {'name': 'Conv2D', 'size': (224, 224, 32)},
        {'name': 'Conv2D', 'size': (222, 222, 32)},
        {'name': 'MaxPool2D', 'size': (111, 111, 32)},
        {'name': 'Conv2D', 'size': (111, 111, 64)},
        {'name': 'Conv2D', 'size': (109, 109, 64)},
        {'name': 'MaxPool2D', 'size': (54, 54, 64)},
        {'name': 'Conv2D', 'size': (54, 54, 64)},
        {'name': 'AdaptiveMaxPool2D', 'size': (1, 1, 64)},
        {'name': 'Flatten', 'size': (1, 1, 64)},  # Flatten is typically represented as the number of features.
        {'name': 'Fully Connected', 'size': (1, 1, 3)}  # Output: 3 classes
    ]

    x_position = 0.05  # Starting position for the first layer

    # Loop through each layer and plot it
    for i, layer in enumerate(layers):
        height, width, channels = layer['size']

        # Scaling the boxes: Use width * channels for the width of the box and height for the height of the box
        box_width = width * 0.003  # Scale the width by a factor of 0.003 for readability
        box_height = height * 0.003  # Scale the height by a factor of 0.003 for readability

        # Box color coding
        if 'Conv' in layer['name']:
            color = 'lightgreen'
            edge_color = 'green'
        elif 'MaxPool' in layer['name']:
            color = 'lightcoral'
            edge_color = 'red'
        elif 'AdaptiveMaxPool' in layer['name']:
            color = 'lightyellow'
            edge_color = 'yellow'
        elif 'Flatten' in layer['name']:
            color = 'lavender'
            edge_color = 'purple'
        else:
            color = 'lightblue'
            edge_color = 'blue'

        # Draw the box with scaled dimensions
        ax.add_patch(patches.FancyBboxPatch((x_position, 0.7), box_width, box_height, boxstyle="round,pad=0.05",
                                            edgecolor=edge_color, facecolor=color, lw=2))

        # Dynamically adjust font size based on box height and width
        font_size = min(box_width, box_height) * 0.3  # Increase the multiplier for better readability
        font_size = max(font_size, 10)  # Ensure font size is not too small (minimum font size is 10)

        # Add the text outside the box, and place it above
        label_x = x_position + box_width / 2  # Label aligned horizontally at the center of the box
        label_y = 0.85  # Place the label a bit above the box

        ax.text(label_x, label_y, layer["name"], ha='center', va='center', fontsize=font_size)

        # Draw arrows pointing to the corresponding box
        ax.annotate('', xy=(x_position + box_width / 2, 0.75), xytext=(label_x, label_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        # Move the x_position for the next layer (spacing between layers)
        x_position += box_width + 0.05  # Add some extra space between the layers

    # Set plot limits and hide axes
    ax.set_xlim(0, x_position)
    ax.set_ylim(0, 1)
    ax.axis('off')  # Hide the axes
    plt.title("Simple CNN Architecture", fontsize=16)
    plt.show()


# Call the function to plot the CNN architecture
plot_cnn_architecture()
