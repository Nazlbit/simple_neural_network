#pragma once
#include <vector>
#include <fstream>

#include "matrix.h"

class neural_net
{
public:
	enum class optimization_method
	{
		none, momentum
	};

private:
	struct layer
	{
		unsigned size = 0;
		unsigned prev_layer_size = 0;
		matrix weights;
		matrix biases;

		layer() = default;
		layer(unsigned size, unsigned prev_layer_size) : size(size), prev_layer_size(prev_layer_size), weights(size, prev_layer_size), biases(size, 1) {}
		layer(unsigned size, unsigned prev_layer_size, float init_val) : size(size), prev_layer_size(prev_layer_size), weights(size, prev_layer_size, init_val), biases(size, 1, init_val) {}


		void init();
	};

	std::vector<layer> layers;
	unsigned input_layer_size;

public:

	neural_net(const unsigned num_layers, const unsigned* const layer_sizes);
	neural_net(const char* const file_name);

	matrix run(matrix input) const;

	std::vector<matrix> run_ext_output(matrix input) const;

	std::vector<layer> backpropagation(const matrix& input, const matrix& required_output);

	void backpropagation(const matrix& input, const matrix& required_output, float rate);

	void train_batch(const matrix& input, const matrix& required_output, unsigned iter_num, float rate);

	void train_stochastic(const matrix& input, const matrix& required_output, unsigned iter_num, float rate);

	void train_mini_batch(const std::vector<matrix>& input, const std::vector<matrix>& required_output, unsigned iter_num, float rate);

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
		return input > 0 ? 1.f : 0.f;
	}

	static matrix activation_function(matrix input);

	static matrix activation_function_derivative(matrix input);
};
