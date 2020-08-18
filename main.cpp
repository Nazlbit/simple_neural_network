#include <iostream>

#include "neural_net.h"
#include "auxiliary.h"

void print(const matrix& values)
{
	for (unsigned j = 0; j < values.get_height(); j++)
	{
		for (unsigned i = 0; i < values.get_width(); i++)
		{
			std::cout << values.at(j, i) << " ";
		}
		std::cout << '\n';
	}
}

void digits()
{
	//Load data

	struct
	{
		binary_data data;
		int32_t magic_number;
		int32_t items_num;
	} labels;
	struct
	{
		binary_data data;
		int32_t magic_number;
		int32_t items_num;
		int32_t num_of_rows;
		int32_t num_of_columns;
		
	} images;

	std::cout << "Loading data into memory...\n";

	//Data source is http://yann.lecun.com/exdb/mnist/
	labels.data = read_file("train-labels.idx1-ubyte");
	images.data = read_file("train-images.idx3-ubyte");

	if (!labels.data.get_data() || !images.data.get_data())
	{
		std::cout << "ERROR: couldn't load data!\n";
		return;
	}

	std::cout << "Processing data...\n";

	labels.magic_number = *(int32_t*)labels.data.get_data();
	labels.items_num = *(int32_t*)(labels.data.get_data() + 4);

	images.magic_number = *(int32_t*)images.data.get_data();
	images.items_num = *(int32_t*)(images.data.get_data() + 4);
	images.num_of_rows = *(int32_t*)(images.data.get_data() + 8);
	images.num_of_columns = *(int32_t*)(images.data.get_data() + 12);

	if (is_little_endian())
	{
		swap_byte_order((char*)&labels.magic_number, sizeof(int32_t));
		swap_byte_order((char*)&labels.items_num, sizeof(int32_t));
		swap_byte_order((char*)&images.magic_number, sizeof(int32_t));
		swap_byte_order((char*)&images.items_num, sizeof(int32_t));
		swap_byte_order((char*)&images.num_of_rows, sizeof(int32_t));
		swap_byte_order((char*)&images.num_of_columns, sizeof(int32_t));
	}

	if (labels.magic_number != 0x00000801 || images.magic_number != 0x00000803)
	{
		std::cout << "ERROR: magic numbers don't match!\n";
		return;
	}

	if (labels.items_num != images.items_num)
	{
		std::cout << "ERROR: data sizes don't match\n";
		return;
	}

	const unsigned input_layer_size = images.num_of_rows * images.num_of_columns;
	const unsigned items_num = images.items_num;
	const unsigned items_num_test = 100;

	matrix input(input_layer_size, items_num);
	matrix required_output(10, items_num, 0.f);

	matrix input_test(input_layer_size, items_num_test);
	matrix required_output_test(10, items_num_test, 0.f);

	const float mul = 1.f / 255;

	//Init train data
	for (int32_t i = 0; i < items_num; i++)
	{
		//Input layer

		char* img_start_addr = images.data.get_data() + i * input_layer_size + 16;
		
		for (unsigned j = 0; j < input_layer_size; j++)
		{
			input.at(i, j) = *reinterpret_cast<uint8_t*>(img_start_addr + j) * mul;
		}

		//Output layer

		required_output.at(i, *reinterpret_cast<uint8_t*>(labels.data.get_data() + i + 8)) = 1.f;
	}

	//Init test data
	for (int32_t i = 0; i < items_num_test; i++)
	{
		//Input layer

		char* img_start_addr = images.data.get_data() + i * input_layer_size + 16;

		for (unsigned j = 0; j < input_layer_size; j++)
		{
			input_test.at(i, j) = *reinterpret_cast<uint8_t*>(img_start_addr + j) * mul; //Shifted by items_num images
		}

		//Output layer

		required_output_test.at(i, *reinterpret_cast<uint8_t*>(labels.data.get_data() + i + 8)) = 1.f;
	}

	const unsigned num_layers = 4;
	const unsigned layer_sizes[num_layers] = {input_layer_size, 16, 16, 10};
	neural_net net(num_layers, layer_sizes);

	float error = neural_net::calculate_error(net.run(input_test), required_output_test);
	unsigned iter = 0;

	std::cout << "Training...\n";

	while(error > 0.01f)
	{
		iter++;
		std::cout << "Iteration #" << iter << '\n';

		net.backpropagation(input, required_output, 0.0001f);
		error = neural_net::calculate_error(net.run(input_test), required_output_test);
		std::cout << "Error: " << error << '\n';
	}

	std::cout << "TEST:\n";

	matrix result = net.run(input_test);
	error = neural_net::calculate_error(result, required_output_test);

	print(result);
	std::cout << '\n';
	print(required_output_test);
	std::cout << "Error: " << error << '\n';

	for (int i = 0; i < 5 && !net.save_to_file("digits_net.bin"); i++);
}

void simple_example()
{
	const unsigned num_layers = 4;
	const unsigned layer_sizes[num_layers] = { 3, 4, 3, 2 };

	neural_net net(num_layers, layer_sizes);

	matrix input(3, 1);
	input.at(0, 0) = 0.5f;
	input.at(0, 1) = 0.2f;
	input.at(0, 2) = 0.8f;

	matrix required_output(2, 1);
	required_output.at(0, 0) = 1.f;
	required_output.at(0, 1) = 0.f;

	//Print initial output
	std::cout << "Initial output:\n";
	print(net.run(input));
	std::cout << '\n';

	//Run backpropagation
	for (int i = 0; i < 100; i++)
	{
		std::cout << "Iteration #" << i << '\n';

		net.backpropagation(input, required_output, 20.f);

		//Calculate Error
		matrix result = net.run(input);
		matrix delta = required_output - result;
		float delta_square_sum = (delta * transpose(delta)).at(0, 0);

		//Print results
		std::cout << "Error : " << delta_square_sum << '\n';
		print(result);
	}

	//Test file saving and loading functionality
	std::cout << "Test file saving and loading functionality\n";
	net.save_to_file("simple_example.bin");
	net.load_from_file("simple_example.bin");
	print(net.run(input));
}

int main()
{
	simple_example();
	//Takes a lot of time to train
	//digits();
	
	return 0;
}
