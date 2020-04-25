/*
 * Copyright (c) 2017-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"
#include <atomic>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <algorithm>

#define _GNU_SOURCE
#include <sched.h>

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

static unsigned int inferences = 0;

/** Example demonstrating how to implement ResNet50's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
class GraphResNet50Example: public Example {
public:

	GraphResNet50Example() :
			cmd_parser(), common_opts(cmd_parser), common_params(), graph(0,
					"ResNet50") {
	}

	bool do_setup(int argc, char **argv) override
	{
		// Parse arguments
		cmd_parser.parse(argc, argv);

		// Consume common parameters
		common_params = consume_common_graph_parameters(common_opts);

		// Return when help menu is requested
		if (common_params.help) {
			cmd_parser.print_help(argv[0]);
			return false;
		}

		// Checks
		ARM_COMPUTE_EXIT_ON_MSG(
				arm_compute::is_data_type_quantized_asymmetric(
						common_params.data_type),
				"QASYMM8 not supported for this graph");
		ARM_COMPUTE_EXIT_ON_MSG(
				common_params.data_type == DataType::F16
						&& common_params.target == Target::NEON,
				"F16 NEON not supported for this graph");

		// Print parameter values
		//std::cout << common_params << std::endl;

		// Get trainable parameters data path
		std::string data_path = common_params.data_path;

		// Create a preprocessor object
		const std::array<float, 3> mean_rgb { { 122.68f, 116.67f, 104.01f } };
		std::unique_ptr<IPreprocessor> preprocessor =
				arm_compute::support::cpp14::make_unique<CaffePreproccessor>(
						mean_rgb, false /* Do not convert to BGR */);

		// Create input descriptor
		const TensorShape tensor_shape = permute_shape(
				TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW,
				common_params.data_layout);
		TensorDescriptor input_descriptor = TensorDescriptor(tensor_shape,
				common_params.data_type).set_layout(common_params.data_layout);

		// Set weights trained layout
		const DataLayout weights_layout = DataLayout::NCHW;

		graph << common_params.target << common_params.fast_math_hint
				<< InputLayer(input_descriptor,
						get_input_accessor(common_params,
								std::move(preprocessor),
								false /* Do not convert to BGR */))
				<< ConvolutionLayer(7U, 7U, 64U,
						get_weights_accessor(data_path,
								"/cnn_data/resnet50_model/conv1_weights.npy",
								weights_layout),
						std::unique_ptr<arm_compute::graph::ITensorAccessor>(
								nullptr), PadStrideInfo(2, 2, 3, 3)).set_name(
						"conv1/convolution")
				<< BatchNormalizationLayer(
						get_weights_accessor(data_path,
								"/cnn_data/resnet50_model/conv1_BatchNorm_moving_mean.npy"),
						get_weights_accessor(data_path,
								"/cnn_data/resnet50_model/conv1_BatchNorm_moving_variance.npy"),
						get_weights_accessor(data_path,
								"/cnn_data/resnet50_model/conv1_BatchNorm_gamma.npy"),
						get_weights_accessor(data_path,
								"/cnn_data/resnet50_model/conv1_BatchNorm_beta.npy"),
						0.0000100099996416f).set_name("conv1/BatchNorm")
				<< ActivationLayer(
						ActivationLayerInfo(
								ActivationLayerInfo::ActivationFunction::RELU)).set_name(
						"conv1/Relu")
				<< PoolingLayer(
						PoolingLayerInfo(PoolingType::MAX, 3,
								PadStrideInfo(2, 2, 0, 1, 0, 1,
										DimensionRoundingType::FLOOR))).set_name(
						"pool1/MaxPool");

		add_residual_block(data_path, "block1", weights_layout, 64, 3, 2);
		add_residual_block(data_path, "block2", weights_layout, 128, 4, 2);
		add_residual_block(data_path, "block3", weights_layout, 256, 6, 2);
		add_residual_block(data_path, "block4", weights_layout, 512, 3, 1);

		graph
				<< PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name(
						"pool5")
				<< ConvolutionLayer(1U, 1U, 1000U,
						get_weights_accessor(data_path,
								"/cnn_data/resnet50_model/logits_weights.npy",
								weights_layout),
						get_weights_accessor(data_path,
								"/cnn_data/resnet50_model/logits_biases.npy"),
						PadStrideInfo(1, 1, 0, 0)).set_name(
						"logits/convolution")
				<< FlattenLayer().set_name("predictions/Reshape")
				<< SoftmaxLayer().set_name("predictions/Softmax")
				<< OutputLayer(get_output_accessor(common_params, 5));

		// Finalize graph
		GraphConfig config;
		config.num_threads = common_params.threads;
		config.use_tuner = common_params.enable_tuner;
		graph.finalize(common_params.target, config);

		return true;
	}
	void do_run() override
	{
		for (unsigned int i = 0; i < ::inferences; i++) {
			graph.run();
		}
	}

private:
	CommandLineParser cmd_parser;
	CommonGraphOptions common_opts;
	CommonGraphParams common_params;
	Stream graph;

	void add_residual_block(const std::string &data_path,
			const std::string &name, DataLayout weights_layout,
			unsigned int base_depth, unsigned int num_units,
			unsigned int stride) {
		for (unsigned int i = 0; i < num_units; ++i) {
			std::stringstream unit_path_ss;
			unit_path_ss << "/cnn_data/resnet50_model/" << name << "_unit_"
					<< (i + 1) << "_bottleneck_v1_";
			std::stringstream unit_name_ss;
			unit_name_ss << name << "/unit" << (i + 1) << "/bottleneck_v1/";

			std::string unit_path = unit_path_ss.str();
			std::string unit_name = unit_name_ss.str();

			unsigned int middle_stride = 1;

			if (i == (num_units - 1)) {
				middle_stride = stride;
			}

			SubStream right(graph);
			right
					<< ConvolutionLayer(1U, 1U, base_depth,
							get_weights_accessor(data_path,
									unit_path + "conv1_weights.npy",
									weights_layout),
							std::unique_ptr<arm_compute::graph::ITensorAccessor>(
									nullptr), PadStrideInfo(1, 1, 0, 0)).set_name(
							unit_name + "conv1/convolution")
					<< BatchNormalizationLayer(
							get_weights_accessor(data_path,
									unit_path
											+ "conv1_BatchNorm_moving_mean.npy"),
							get_weights_accessor(data_path,
									unit_path
											+ "conv1_BatchNorm_moving_variance.npy"),
							get_weights_accessor(data_path,
									unit_path + "conv1_BatchNorm_gamma.npy"),
							get_weights_accessor(data_path,
									unit_path + "conv1_BatchNorm_beta.npy"),
							0.0000100099996416f).set_name(
							unit_name + "conv1/BatchNorm")
					<< ActivationLayer(
							ActivationLayerInfo(
									ActivationLayerInfo::ActivationFunction::RELU)).set_name(
							unit_name + "conv1/Relu")

					<< ConvolutionLayer(3U, 3U, base_depth,
							get_weights_accessor(data_path,
									unit_path + "conv2_weights.npy",
									weights_layout),
							std::unique_ptr<arm_compute::graph::ITensorAccessor>(
									nullptr),
							PadStrideInfo(middle_stride, middle_stride, 1, 1)).set_name(
							unit_name + "conv2/convolution")
					<< BatchNormalizationLayer(
							get_weights_accessor(data_path,
									unit_path
											+ "conv2_BatchNorm_moving_mean.npy"),
							get_weights_accessor(data_path,
									unit_path
											+ "conv2_BatchNorm_moving_variance.npy"),
							get_weights_accessor(data_path,
									unit_path + "conv2_BatchNorm_gamma.npy"),
							get_weights_accessor(data_path,
									unit_path + "conv2_BatchNorm_beta.npy"),
							0.0000100099996416f).set_name(
							unit_name + "conv2/BatchNorm")
					<< ActivationLayer(
							ActivationLayerInfo(
									ActivationLayerInfo::ActivationFunction::RELU)).set_name(
							unit_name + "conv1/Relu")

					<< ConvolutionLayer(1U, 1U, base_depth * 4,
							get_weights_accessor(data_path,
									unit_path + "conv3_weights.npy",
									weights_layout),
							std::unique_ptr<arm_compute::graph::ITensorAccessor>(
									nullptr), PadStrideInfo(1, 1, 0, 0)).set_name(
							unit_name + "conv3/convolution")
					<< BatchNormalizationLayer(
							get_weights_accessor(data_path,
									unit_path
											+ "conv3_BatchNorm_moving_mean.npy"),
							get_weights_accessor(data_path,
									unit_path
											+ "conv3_BatchNorm_moving_variance.npy"),
							get_weights_accessor(data_path,
									unit_path + "conv3_BatchNorm_gamma.npy"),
							get_weights_accessor(data_path,
									unit_path + "conv3_BatchNorm_beta.npy"),
							0.0000100099996416f).set_name(
							unit_name + "conv2/BatchNorm");

			if (i == 0) {
				SubStream left(graph);
				left
						<< ConvolutionLayer(1U, 1U, base_depth * 4,
								get_weights_accessor(data_path,
										unit_path + "shortcut_weights.npy",
										weights_layout),
								std::unique_ptr<
										arm_compute::graph::ITensorAccessor>(
										nullptr), PadStrideInfo(1, 1, 0, 0)).set_name(
								unit_name + "shortcut/convolution")
						<< BatchNormalizationLayer(
								get_weights_accessor(data_path,
										unit_path
												+ "shortcut_BatchNorm_moving_mean.npy"),
								get_weights_accessor(data_path,
										unit_path
												+ "shortcut_BatchNorm_moving_variance.npy"),
								get_weights_accessor(data_path,
										unit_path
												+ "shortcut_BatchNorm_gamma.npy"),
								get_weights_accessor(data_path,
										unit_path
												+ "shortcut_BatchNorm_beta.npy"),
								0.0000100099996416f).set_name(
								unit_name + "shortcut/BatchNorm");

				graph
						<< BranchLayer(BranchMergeMethod::ADD, std::move(left),
								std::move(right)).set_name(unit_name + "add");
			} else if (middle_stride > 1) {
				SubStream left(graph);
				left
						<< PoolingLayer(
								PoolingLayerInfo(PoolingType::MAX, 1,
										PadStrideInfo(middle_stride,
												middle_stride, 0, 0), true)).set_name(
								unit_name + "shortcut/MaxPool");

				graph
						<< BranchLayer(BranchMergeMethod::ADD, std::move(left),
								std::move(right)).set_name(unit_name + "add");
			} else {
				SubStream left(graph);
				graph
						<< BranchLayer(BranchMergeMethod::ADD, std::move(left),
								std::move(right)).set_name(unit_name + "add");
			}

			graph
					<< ActivationLayer(
							ActivationLayerInfo(
									ActivationLayerInfo::ActivationFunction::RELU)).set_name(
							unit_name + "Relu");
		}
	}
};

int get_temp() {
	std::ofstream file;
	file.open("/sys/class/thermal/thermal_zone0/temp");
	int temp;
	file << temp;
	file.close();
	return temp;
}

static std::atomic_uint *run_cpu_small, *run_cpu_big, *run_gpu, *val;

struct _config {
	std::string name;
	int argc;
	std::vector<std::string> argv;
	std::atomic_uint *run; // run or sleep
	double time_taken;
};

struct units {
	bool cpu_small;
	bool cpu_big;
	bool gpu;
};

/* maximum rise in temperatures
 {'cpu_big': 20563.2,
 'cpu_big_cpu_small': 21168.0,
 'cpu_small': 3628.8,
 'gpu': 9223.2,
 'gpu_cpu_big': 22096.8,
 'gpu_cpu_big_cpu_small': 22960.8,
 'gpu_cpu_small': 12009.6}
 */
static std::map<units, double> delta_temps = {
// {{small, big, gpu}, delta_temp}
		{ { 0, 1, 0 }, 20563.2 }, { { 1, 1, 0 }, 21168.0 }, { { 1, 0, 0 },
				3628.8 }, { { 0, 0, 1 }, 9223.2 }, { { 0, 1, 1 }, 22096.8 }, { {
				1, 1, 1 }, 22960.8 }, { { 1, 0, 1 }, 12009.6 } };

static const int TL = 80000;

/** Main program for ResNet50
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 *
 * @return Return code
 */
int main(int argc, char **argv) {
	// Command Line Parsing
	CommandLineParser parser;

	SimpleOption<unsigned int> *imagesOption = parser.add_option<
			SimpleOption<unsigned int>>("n", 100);
	ToggleOption *helpOption = parser.add_option<ToggleOption>("help");
	helpOption->set_help("Help");
	imagesOption->set_help("Images");

	parser.parse(argc, argv);

	bool help = helpOption->is_set() ? helpOption->value() : false;
	if (help) {
		parser.print_help(argv[0]);
	}

	unsigned int images = imagesOption->value();

	// Set up flags
	run_cpu_small = static_cast<std::atomic_uint*>(mmap(NULL,
			sizeof *run_cpu_small,
			PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
	run_cpu_big = static_cast<std::atomic_uint*>(mmap(NULL, sizeof *run_cpu_big,
	PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
	run_gpu = static_cast<std::atomic_uint*>(mmap(NULL, sizeof *run_gpu,
	PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
	val = static_cast<std::atomic_uint*>(mmap(NULL, sizeof *val,
	PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));

	// Create configs for child processes
	/*time taken
	 {'cpu_big': 0.014421939849853516,
	 'cpu_small': 0.021512556076049804,
	 'gpu': 0.025214433670043945}*/
	double cpu_small_time = 0.014421939849853516, cpu_big_time =
			0.021512556076049804, gpu_time = 0.025214433670043945;
	_config configs[] = { { "cpu_small", 3, { argv[0], "--target=NEON",
			"--threads=4" }, run_cpu_small, cpu_small_time },
			{ "cpu_big", 3, { argv[0], "--target=NEON", "--threads=4" },
					run_cpu_big, cpu_big_time }, { "gpu", 2, { argv[0],
					"--target=CL" }, run_gpu, gpu_time } };

	// Create Child Processes
	int pid = fork();
	std::vector<double> temperatures;
	if (pid > 0) {
		double min_delta_temp = std::min_element(delta_temps.begin(),
				delta_temps.end(), [](const auto &l, const auto &r) {
					return l.second < r.second;
				});
		usleep(5000);
		while (*val < images) {
			double temp = get_temp();
			temperatures.push_back(temp);
			if (temp + min_delta_temp < TL) {
				double delta_temp = TL - temp;
				bool min_cpu_small = false;
				bool min_cpu_big = false;
				bool min_gpu = false;
				double min_total_time = 99999999;
				for (int i = 1; i <= 7; i++) {
					bool cpu_small = i & 1;
					bool cpu_big = i & 2;
					bool gpu = i & 4;
					// combination must have current target as running.
					if ((*run_cpu_small && !cpu_small)
							|| (*run_cpu_big && !cpu_big)
							|| (*run_gpu && !gpu)) {
						continue;
					}
					if (delta_temps[ { cpu_small, cpu_big, gpu }]
							< delta_temp) {
						double total_time = 0;
						if (cpu_small)
							total_time += cpu_small_time;
						if (cpu_big)
							total_time += cpu_big_time;
						if (gpu)
							total_time += gpu_time;
						if (total_time < min_total_time) {
							min_cpu_small = cpu_small;
							min_cpu_big = cpu_big;
							min_gpu = gpu;
						}
					}
				}
				*run_cpu_small = min_cpu_small ? 1 : 0;
				*run_cpu_big = min_cpu_big ? 1 : 0;
				*run_gpu = min_gpu ? 1 : 0;
			}
			usleep(100);
		}
		for (auto &temp : temperatures) {
			std::cout << temp << "\n";
		}
	} else {
		for (auto config : configs) {
			int pid = fork();
			if (pid == 0) {
				cpu_set_t mask;
				if (config.name == "cpu_small") {
					CPU_ZERO(&mask);
					for (int i = 0; i <= 3; i++)
						CPU_SET(i, &mask);
					sched_setaffinity(0, sizeof(mask), &mask);
				} else if (config.name == "cpu_big") {
					CPU_ZERO(&mask);
					for (int i = 4; i <= 7; i++)
						CPU_SET(i, &mask);
					sched_setaffinity(0, sizeof(mask), &mask);
				}
				while (true) {
					if (*val >= images)
						break;
					(*val)++;
					if (config.run) {
						arm_compute::utils::run_example<GraphResNet50Example>(
								config.argc, argv);
					}
				}
				exit(1);
			}

		}

	}
}
