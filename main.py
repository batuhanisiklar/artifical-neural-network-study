import math
from visualization import visualize_network_structure
import random

def calculate_total_error(output_errors):
    total_error = 0
    if isinstance(output_errors, list):
        for error in output_errors:
            total_error += abs(error)  # Sum absolute errors
    else:
        total_error = abs(output_errors)
    return total_error

# Function to calculate the sum of weights in the hidden layer
def hidden_layer_weight_sum(input_values, weights, hidden_layer_count):
    total_values = []
    for i in range(hidden_layer_count):
        total = 0
        for j in range(len(input_values)):
            total += input_values[j] * weights[j * hidden_layer_count + i]
        total_values.append(round(total, 3))  # Round to 3 decimal places
    return total_values

def hidden_layer_output(values):
    output_values = []
    for value in values:
        new_value = 1 / (1 + math.e ** -value)
        output_values.append(round(new_value, 3))
    return output_values

def calculate_error(output, expected_output):
    error_values = []
    for i in range(len(output)):
        new_value = output[i] * (1 - output[i]) * (expected_output[i] - output[i])
        error_values.append(round(new_value, 3))
    return error_values

def update_weights(weights, input_values, error_values, learning_rate):
    for i in range(len(error_values)):
        new_value = weights[i] + (learning_rate * input_values[i] * error_values[i])
        weights[i] = round(new_value, 3)
    return weights

# Training loop
def backpropagation_training(input_values, expected_output, weights_1, weights_2, learning_rate, hidden_layer_count):
    hidden_layer_outputs = []

    # 1. Forward propagation
    hidden_layer_sum = hidden_layer_weight_sum(input_values, weights_1, hidden_layer_count)
    hidden_layer_out = hidden_layer_output(hidden_layer_sum)
    hidden_layer_outputs.append(hidden_layer_out)

    new_hidden_sum = hidden_layer_weight_sum(hidden_layer_out, weights_2, len(expected_output))
    new_hidden_out = hidden_layer_output(new_hidden_sum)

    # 2. Calculate error at output layer
    output_error = calculate_error(new_hidden_out, expected_output)

    # Print input and hidden layer outputs
    print(f"Input Values: {input_values}")
    print(f"Hidden Layer Output: {hidden_layer_out}")
    print(f"Output Error: {output_error}")

    # Update weights
    weights_2 = update_weights(weights_2, hidden_layer_out, output_error, learning_rate)

    # Calculate hidden layer error
    hidden_error = calculate_error(new_hidden_out, expected_output)

    # Update hidden layer weights
    weights_1 = update_weights(weights_1, input_values, hidden_error, learning_rate)

    total_error = calculate_total_error(output_error)
    # Print updated weights
    print(f"Updated Weights 1: {weights_1}")
    print(f"Updated Weights 2: {weights_2}")
    print(f"Total Error : {total_error}\n")
    return hidden_layer_outputs, total_error

hidden_layer_count = int(input("Enter the number of hidden layers: "))

input_values = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]
expected_outputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
learning_rate = 0.1

# Randomly generate weights
weights_1 = [round(random.uniform(-1, 4), 3) for _ in range(hidden_layer_count * len(input_values))]
print(f"\nFirst Layer Weights: {weights_1}")

weights_2 = [round(random.uniform(-1, 4), 3) for _ in range(hidden_layer_count * len(expected_outputs))]
print(f"Second Layer Weights: {weights_2}\n")

total_errors = 0

# Training process
for i in range(len(input_values)):
    hidden_layer_outputs, total_error = backpropagation_training(
        input_values[i], expected_outputs[i], weights_1, weights_2, learning_rate, hidden_layer_count
    )
    total_errors += total_error  # Add total error at each step

print(f"Total Error = {total_errors}\n")

# Visualize the network structure
visualize_network_structure(input_values, hidden_layer_outputs[-1], expected_outputs, weights_1, weights_2)
