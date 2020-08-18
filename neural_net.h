#pragma once
#include <vector>
#include <fstream>

#include "matrix.h"

class neural_net
{
	struct layer
	{
		const unsigned size;
		matrix weights;
		matrix biases;

		layer(unsigned size, unsigned prev_layer_size) : size(size), biases(size, 1, 0.f), weights(size, prev_layer_size, 0.f) {}
		void init();
	};

	std::vector<layer> layers;
	unsigned input_layer_size;

public:
	neural_net(const unsigned num_layers, const unsigned* const layer_sizes);

	matrix run(const matrix& input) const;

	std::vector<matrix> run_ext_output(const matrix& input) const;

	void backpropagation(const matrix& input, const matrix& required_output, float rate);

	bool save_to_file(const char* const file_name);

	bool load_from_file(const char* const file_name);

	static float calculate_error(matrix values, matrix required_values);

	static float sigmoid(float input)
	{
		return 1.f / (1.f + expf(-input));
	}

	static float sigmoid_derivative(float input)
	{
		return input * (1 - input);
	}

	static float relu(float input)
	{
		return input > 0 ? input : 0;
	}

	static float relu_derivative(float input)
	{
		return input > 0 ? 1 : 0;
	}

	static matrix activation_function(matrix input);

	static matrix activation_function_derivative(matrix input);
};
