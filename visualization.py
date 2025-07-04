import matplotlib.pyplot as plt
import networkx as nx

# Define the structure of the network
def visualize_network_structure(input_values, hidden_layer_outputs, output_values, weights_1, weights_2):
    G = nx.DiGraph()  # Create a directed graph

    # Add input layer nodes
    for i in range(len(input_values)):
        G.add_node(f"Input {i + 1}", pos=(0, i))  # Number as Input 1, Input 2, etc.

    # Add hidden layer nodes
    for i in range(len(hidden_layer_outputs)):
        G.add_node(f"Hidden {i + 1}", pos=(1, i))  # Number as Hidden 1, Hidden 2, etc.

    # Add output layer nodes
    for i in range(len(output_values)):
        G.add_node(f"Output {i + 1}", pos=(2, i))  # Number as Output 1, Output 2, etc.

    # Add edges and set edge labels
    for i in range(len(input_values)):
        for j in range(len(hidden_layer_outputs)):
            G.add_edge(f"Input {i + 1}", f"Hidden {j + 1}", weight=weights_1[i + j * len(input_values)])

    for i in range(len(hidden_layer_outputs)):
        for j in range(len(output_values)):
            G.add_edge(f"Hidden {i + 1}", f"Output {j + 1}", weight=weights_2[i])

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Visualize the network
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)

    plt.axis('off')
    plt.show()
