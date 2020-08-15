#include <chrono>
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>

#include "matrix.h"

float random_float(float min, float max)
{
	static std::mt19937_64 g1(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<float> dis(min, max);
	return dis(g1);
}

float sigmoid(float input)
{
	return 1.f / (1.f + expf(-input));
}

struct neural_net
{
	struct layer
	{
		const unsigned size;
		matrix weights;
		matrix biases;

		layer(unsigned size, unsigned prev_layer_size) : size(size), biases(size, 1), weights(size, prev_layer_size)
		{
			init();
		}

		void init()
		{
			for (unsigned i = 0; i < size; i++)
			{
				biases.at(0, i) = random_float(-1.f, 1.f);

				for (unsigned j = 0; j < weights.get_height(); j++)
				{
					weights.at(j, i) = random_float(-1.f, 1.f);
				}
			}
		}
	};

	std::vector<layer> layers;
	unsigned input_layer_size;

	neural_net(unsigned num_layers, unsigned* layer_sizes)
	{
		assert(num_layers > 1);
		assert(layer_sizes != nullptr);
		
		input_layer_size = layer_sizes[0];

		layers.reserve(num_layers - 1);

		//Init layers
		for (unsigned i = 1; i < num_layers; i++)
		{
			layers.emplace_back(layer_sizes[i], layer_sizes[i - 1]);
		}
	}

	matrix activation_function(matrix input) const
	{
		assert(input.get_height() == 1);

		for (unsigned i = 0; i < input.get_width(); i++)
		{
			input.at(0, i) = sigmoid(input.at(0, i));
		}
		return input;
	}

	matrix run(const matrix& input) const
	{
		assert(input.get_width() == input_layer_size);
		assert(input.get_height() == 1);

		matrix result = input;

		for (const layer& l : layers)
		{
			result = activation_function(result * l.weights + l.biases);
		}

		return result;
	}

	std::vector<matrix> run_ext_output(const matrix& input) const
	{
		assert(input.get_width() == input_layer_size);
		assert(input.get_height() == 1);

		std::vector<matrix> result;
		result.reserve(layers.size() + 1);
		result.push_back(input);

		for (const layer& l : layers)
		{
			result.push_back(activation_function(result.back() * l.weights + l.biases));
		}

		return result;
	}

	bool backpropagation(const matrix& input, const matrix& required_output, float rate, float threshold)
	{
		assert(input.is_alive() && required_output.is_alive());
		assert(input.get_width() == input_layer_size);
		assert(input.get_height() == 1);
		assert(required_output.get_width() == layers.back().size);
		assert(required_output.get_height() == 1);
		assert(rate > 0);
		assert(threshold > 0);

		std::vector<matrix> values = run_ext_output(input); // Calculate initial neurons activation values
		
		matrix x = values.back() - required_output; // Delta
		
		if ((x * transpose(x)).at(0, 0) < threshold) // dot(x, x) - sum of squared deltas < threshold
			return false;

		for (unsigned i = values.size() - 1; i > 0; i--) // For every layer starting from the last
		{
			x = hadamard_product(x, hadamard_product(values[i], (1.f - values[i]))); // Sigmoid partial derivative

			layers[i - 1].biases = layers[i - 1].biases - x * rate; // Calculate new bias

			matrix weights_derivatives = transpose(values[i - 1]) * x; // Weights partial derivative

			x = transpose(layers[i - 1].weights * transpose(x)); // Neuron connection partial derivative

			layers[i - 1].weights = layers[i - 1].weights - weights_derivatives * rate; // Calculate new weights
		}

		return true;
	}
};

void print(const matrix& values)
{
	for (unsigned i = 0; i < values.get_width(); i++)
	{
		std::cout << values.at(0, i) << " ";
	}
	std::cout << std::endl;
}

int main()
{
	const unsigned num_layers = 4;
	unsigned layer_sizes[num_layers] = {3, 4, 3, 2};

	neural_net test(num_layers, layer_sizes);

	matrix input(3, 1);
	input.at(0, 0) = 0.5f;
	input.at(0, 1) = 0.2f;
	input.at(0, 2) = 0.8f;

	matrix required_output(2,1);
	required_output.at(0, 0) = 1.f;
	required_output.at(0, 1) = 0.f;

	// Print initial output
	std::cout << "Initial output:" << std::endl;
	print(test.run(input));
	std::cout << std::endl;

	// Run backpropagation
	int i = 0;
	while(test.backpropagation(input, required_output, exp(i * 2.f), 1e-9f)) i++;

	// Calculate Sum of square deltas
	matrix result = test.run(input);
	matrix delta = required_output - result;
	float delta_square_sum = (delta * transpose(delta)).at(0, 0);

	// Print results
	std::cout << "Num iterations: " << i << std::endl;
	std::cout << "Sum of squared deltas : " << delta_square_sum << std::endl;
	print(result);

	return 0;
}
