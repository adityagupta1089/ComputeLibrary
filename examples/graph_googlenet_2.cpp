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

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

static unsigned int inferences = 0;

/** Example demonstrating how to implement AlexNet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
class GraphGooglenetExample  : public Example
{
public:
    

    GraphAlexnetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "GoogleNet")
    {
    }
    
    static std::mutex finalize_mutex;
    
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        //cmd_parser.parse(argc, argv);

        // Consume common parameters
        //common_params = consume_common_graph_parameters(common_opts);
        
        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        //std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(227U, 227U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
              << ConvolutionLayer(
                  7U, 7U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv1/conv1_7x7_s2_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv1/conv1_7x7_s2_b.npy"),
                  PadStrideInfo(2, 2, 3, 3))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f))
              << ConvolutionLayer(
                  1U, 1U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 192U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        graph << get_inception_node(data_path, "inception_3a", weights_layout, 64, std::make_tuple(96U, 128U), std::make_tuple(16U, 32U), 32U);
        graph << get_inception_node(data_path, "inception_3b", weights_layout, 128, std::make_tuple(128U, 192U), std::make_tuple(32U, 96U), 64U);
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        graph << get_inception_node(data_path, "inception_4a", weights_layout, 192, std::make_tuple(96U, 208U), std::make_tuple(16U, 48U), 64U);
        graph << get_inception_node(data_path, "inception_4b", weights_layout, 160, std::make_tuple(112U, 224U), std::make_tuple(24U, 64U), 64U);
        graph << get_inception_node(data_path, "inception_4c", weights_layout, 128, std::make_tuple(128U, 256U), std::make_tuple(24U, 64U), 64U);
        graph << get_inception_node(data_path, "inception_4d", weights_layout, 112, std::make_tuple(144U, 288U), std::make_tuple(32U, 64U), 64U);
        graph << get_inception_node(data_path, "inception_4e", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U);
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
        graph << get_inception_node(data_path, "inception_5a", weights_layout, 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U);
        graph << get_inception_node(data_path, "inception_5b", weights_layout, 384, std::make_tuple(192U, 384U), std::make_tuple(48U, 128U), 128U);
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 7, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL)))
              << FullyConnectedLayer(
                  1000U,
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_b.npy"))
              << SoftmaxLayer()
		<< OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_file  = common_params.tuner_file;
        graph.finalize(common_params.target, config);
        return true;
    }
    void do_run() override
    {
	for (unsigned int i = 0; i < ::inferences; i++){
            graph.run();
	}
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;
};
BranchLayer get_inception_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                   unsigned int a_filt,
                                   std::tuple<unsigned int, unsigned int> b_filters,
                                   std::tuple<unsigned int, unsigned int> c_filters,
                                   unsigned int d_filt)
    {
        std::string total_path = "/cnn_data/googlenet_model/" + param_path + "/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, a_filt,
                get_weights_accessor(data_path, total_path + "1x1_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, std::get<0>(b_filters),
                get_weights_accessor(data_path, total_path + "3x3_reduce_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "3x3_reduce_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                3U, 3U, std::get<1>(b_filters),
                get_weights_accessor(data_path, total_path + "3x3_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "3x3_b.npy"),
                PadStrideInfo(1, 1, 1, 1))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                1U, 1U, std::get<0>(c_filters),
                get_weights_accessor(data_path, total_path + "5x5_reduce_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "5x5_reduce_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                5U, 5U, std::get<1>(c_filters),
                get_weights_accessor(data_path, total_path + "5x5_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "5x5_b.npy"),
                PadStrideInfo(1, 1, 2, 2))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)))
            << ConvolutionLayer(
                1U, 1U, d_filt,
                get_weights_accessor(data_path, total_path + "pool_proj_w.npy", weights_layout),
                get_weights_accessor(data_path, total_path + "pool_proj_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }
};
struct _config {
    bool                        execute;
    std::string                 name;
    int                         argc;
    std::vector<std::string>    argv;
};

static std::atomic_uint* val;

/** Main program for AlexNet
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 *
 * @return Return code
 */
int main(int argc, char **argv)
{
    // Command Line Parsing
    CommandLineParser parser;

    ToggleOption*      cpuOption    = parser.add_option<ToggleOption>("cpu");
    ToggleOption*      gpuOption    = parser.add_option<ToggleOption>("gpu");
    ToggleOption*      helpOption   = parser.add_option<ToggleOption>("help");
    SimpleOption<unsigned int>* imagesOption     = parser.add_option<SimpleOption<unsigned int>>("n", 100);
    SimpleOption<unsigned int>* inferencesOption = parser.add_option<SimpleOption<unsigned int>>("i", 1); 

    cpuOption->set_help("CPU");
    gpuOption->set_help("GPU");
    helpOption->set_help("Help");
    imagesOption->set_help("Images");
    inferencesOption->set_help("Inferences");
    
    parser.parse(argc, argv);
    
    ARM_COMPUTE_EXIT_ON_MSG(!helpOption->is_set() && !cpuOption->is_set() && !gpuOption->is_set(), "No target given, add --cpu or --gpu");
    
    bool help = helpOption->is_set() ? helpOption->value() : false;
    if (help) {
        parser.print_help(argv[0]);
    }
    
    bool cpu = cpuOption->is_set() ? cpuOption->value() : false;
    bool gpu = gpuOption->is_set() ? gpuOption->value() : false;

    unsigned int images     = imagesOption->value();
                 inferences = inferencesOption->value();

    std::cout << "CPU:" << cpu << ", GPU:" << gpu << ", Images:" << images <<"\n";
    
    // Create configs for child processes
    _config configs[] = {
        {cpu, "CPU", 3, {argv[0], "--target=NEON", "--threads=4"}},
        {gpu, "GPU", 2, {argv[0], "--target=CL"}}
    };
    
    // Start Timer
    auto tbegin = std::chrono::high_resolution_clock::now();
    
    // Shared counter
    val = static_cast<std::atomic_uint*>(mmap(NULL, sizeof *val, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    *val = 0;

    // Create Child Processes
    for (auto config: configs) {
        if (config.execute) {
            int pid = fork();
            if (pid == 0) {
                std::cout << "Started " << config.name << "[" << getpid() << "]" << std::endl;
                char* argv[config.argc];
                for(int i = 0; i < config.argc; i++) {
                    argv[i] = new char[config.argv[i].length() + 1];
                    strcpy(argv[i], config.argv[i].c_str());
                }
                int processed = 0;
                while (true) {
                    if (*val >= images) break;
                    //std::cout << config.name << " : " << (*val)++ << std::endl;
                    processed++;
                    (*val)++;
                    arm_compute::utils::run_example<GraphGooglenetExample>(config.argc, argv); 
                }
                std::cout << "Completed " << config.name << ": " << processed << " inferences" << std::endl;
                exit(1);
            }
        }
    }

    // Wait for children to complete
    for (auto config: configs) {
        if (config.execute) {
            int status;
            pid_t pid = wait(&status);
            std::cout << "Finished [" << pid << "]" << std::endl;
        }
    }

    // Calculate Time taken
    auto tend = std::chrono::high_resolution_clock::now();
    double gross = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
    double cost = gross / images;
    std::cout << cost << " per image" << std::endl;
    cost /= inferences;
    std::cout << cost << " per inference" << std::endl; 
}
