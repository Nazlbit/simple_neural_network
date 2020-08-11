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
		unsigned size = 0;
		matrix<float> weights;
		matrix<float> biases;

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
	unsigned input_layer_size = 0;

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

	matrix<float> activation_function(matrix<float> input)
	{
		assert(input.get_height() == 1);

		for (unsigned i = 0; i < input.get_width(); i++)
		{
			input.at(0, i) = sigmoid(input.at(0, i));
		}
		return input;
	}

	matrix<float> run(matrix<float> input)
	{
		assert(input.get_width() == input_layer_size);
		assert(input.get_height() == 1);

		for (const layer& l : layers)
		{
			input = activation_function(input * l.weights + l.biases);
		}
		return input;
	}
};

int main()
{
	const unsigned num_layers = 4;
	unsigned layer_sizes[num_layers] = {3, 4, 3, 2};

	neural_net test(num_layers, layer_sizes);

	matrix<float> input(3, 1);
	input.at(0, 0) = 0.5f;
	input.at(0, 1) = 0.2f;
	input.at(0, 2) = 0.8f;

	matrix<float> output = test.run(input);

	return 0;
}
