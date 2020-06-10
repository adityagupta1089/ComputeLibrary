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

#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <sched.h>

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;
using namespace std;
using namespace std::chrono;

static unsigned int images     = 100;
static unsigned int inferences = 1;
static atomic_uint *run_cpu_small, *run_cpu_big, *run_gpu, *val = 0;

/** Example demonstrating how to implement ResNet50's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
class GraphResNet50Example : public Example
{
    public:
    GraphResNet50Example()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "ResNet50")
    {
    }

    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");
        ARM_COMPUTE_EXIT_ON_MSG(common_params.data_type == DataType::F16 && common_params.target == Target::NEON, "F16 NEON not supported for this graph");

        // Print parameter values
        //cout << common_params << endl;

        // Get trainable parameters data path
        string data_path = common_params.data_path;

        // Create a preprocessor object
        const array<float, 3> mean_rgb{
            { 122.68f, 116.67f, 104.01f }
        };
        unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb, false /* Do not convert to BGR */);

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, move(preprocessor), false /* Do not convert to BGR */))
              << ConvolutionLayer(7U, 7U, 64U, get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_weights.npy", weights_layout), unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 3, 3)).set_name("conv1/convolution")
              << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_moving_mean.npy"), get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_moving_variance.npy"), get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_gamma.npy"), get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_beta.npy"), 0.0000100099996416f).set_name("conv1/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv1/Relu")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))).set_name("pool1/MaxPool");

        add_residual_block(data_path, "block1", weights_layout, 64, 3, 2);
        add_residual_block(data_path, "block2", weights_layout, 128, 4, 2);
        add_residual_block(data_path, "block3", weights_layout, 256, 6, 2);
        add_residual_block(data_path, "block4", weights_layout, 512, 3, 1);

        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("pool5") << ConvolutionLayer(1U, 1U, 1000U, get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_weights.npy", weights_layout), get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_biases.npy"), PadStrideInfo(1, 1, 0, 0)).set_name("logits/convolution") << FlattenLayer().set_name("predictions/Reshape") << SoftmaxLayer().set_name("predictions/Softmax") << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        graph.finalize(common_params.target, config);

        return true;
    }
    void do_run() override
    {
        for(unsigned int i = 0; i < ::inferences; i++)
        {
            graph.run();
        }
    }

    private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    void add_residual_block(const string &data_path, const string &name, DataLayout weights_layout, unsigned int base_depth, unsigned int num_units, unsigned int stride)
    {
        for(unsigned int i = 0; i < num_units; ++i)
        {
            stringstream unit_path_ss;
            unit_path_ss << "/cnn_data/resnet50_model/" << name << "_unit_" << (i + 1) << "_bottleneck_v1_";
            stringstream unit_name_ss;
            unit_name_ss << name << "/unit" << (i + 1) << "/bottleneck_v1/";

            string unit_path = unit_path_ss.str();
            string unit_name = unit_name_ss.str();

            unsigned int middle_stride = 1;

            if(i == (num_units - 1))
            {
                middle_stride = stride;
            }

            SubStream right(graph);
            right << ConvolutionLayer(1U, 1U, base_depth, get_weights_accessor(data_path, unit_path + "conv1_weights.npy", weights_layout), unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0)).set_name(unit_name + "conv1/convolution") << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_moving_mean.npy"), get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_moving_variance.npy"), get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_gamma.npy"), get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_beta.npy"), 0.0000100099996416f).set_name(unit_name + "conv1/BatchNorm") << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv1/Relu")

                  << ConvolutionLayer(3U, 3U, base_depth, get_weights_accessor(data_path, unit_path + "conv2_weights.npy", weights_layout), unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(middle_stride, middle_stride, 1, 1)).set_name(unit_name + "conv2/convolution") << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_moving_mean.npy"), get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_moving_variance.npy"), get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_gamma.npy"), get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_beta.npy"), 0.0000100099996416f).set_name(unit_name + "conv2/BatchNorm") << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv1/Relu")

                  << ConvolutionLayer(1U, 1U, base_depth * 4, get_weights_accessor(data_path, unit_path + "conv3_weights.npy", weights_layout), unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0)).set_name(unit_name + "conv3/convolution") << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_moving_mean.npy"), get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_moving_variance.npy"), get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_gamma.npy"), get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_beta.npy"), 0.0000100099996416f).set_name(unit_name + "conv2/BatchNorm");

            if(i == 0)
            {
                SubStream left(graph);
                left << ConvolutionLayer(1U, 1U, base_depth * 4, get_weights_accessor(data_path, unit_path + "shortcut_weights.npy", weights_layout), unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0)).set_name(unit_name + "shortcut/convolution") << BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_moving_mean.npy"), get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_moving_variance.npy"), get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_gamma.npy"), get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_beta.npy"), 0.0000100099996416f).set_name(unit_name + "shortcut/BatchNorm");

                graph << BranchLayer(BranchMergeMethod::ADD, move(left), move(right)).set_name(unit_name + "add");
            }
            else if(middle_stride > 1)
            {
                SubStream left(graph);
                left << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 1, PadStrideInfo(middle_stride, middle_stride, 0, 0), true)).set_name(unit_name + "shortcut/MaxPool");

                graph << BranchLayer(BranchMergeMethod::ADD, move(left), move(right)).set_name(unit_name + "add");
            }
            else
            {
                SubStream left(graph);
                graph << BranchLayer(BranchMergeMethod::ADD, move(left), move(right)).set_name(unit_name + "add");
            }

            graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Relu");
        }
    }
};
class GraphAlexnetExample : public Example
{
    public:
    GraphAlexnetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "AlexNet")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Set default layout if needed
        if(!common_opts.data_layout->is_set() && common_params.target == Target::NEON)
        {
            common_params.data_layout = DataLayout::NCHW;
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
              // Layer 1
              << ConvolutionLayer(
                     11U, 11U, 96U,
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_b.npy"),
                     PadStrideInfo(4, 4, 0, 0))
                     .set_name("conv1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu1")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm1")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool1")
              // Layer 2
              << ConvolutionLayer(
                     5U, 5U, 256U,
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_b.npy"),
                     PadStrideInfo(1, 1, 2, 2), 2)
                     .set_name("conv2")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm2")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool2")
              // Layer 3
              << ConvolutionLayer(
                     3U, 3U, 384U,
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_b.npy"),
                     PadStrideInfo(1, 1, 1, 1))
                     .set_name("conv3")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3")
              // Layer 4
              << ConvolutionLayer(
                     3U, 3U, 384U,
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_b.npy"),
                     PadStrideInfo(1, 1, 1, 1), 2)
                     .set_name("conv4")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4")
              // Layer 5
              << ConvolutionLayer(
                     3U, 3U, 256U,
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_b.npy"),
                     PadStrideInfo(1, 1, 1, 1), 2)
                     .set_name("conv5")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool5")
              // Layer 6
              << FullyConnectedLayer(
                     4096U,
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_b.npy"))
                     .set_name("fc6")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu6")
              // Layer 7
              << FullyConnectedLayer(
                     4096U,
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_b.npy"))
                     .set_name("fc7")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu7")
              // Layer 8
              << FullyConnectedLayer(
                     1000U,
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_b.npy"))
                     .set_name("fc8")
              // Softmax
              << SoftmaxLayer().set_name("prob")
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
        graph.run();
    }

    private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;
};
class GraphMobilenetExample : public Example
{
    public:
    GraphMobilenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "MobileNetV1")
    {
        // Add model id option
        model_id_opt = cmd_parser.add_option<SimpleOption<int>>("model-id", 0);
        model_id_opt->set_help("Mobilenet model id (0: 1.0_224, else: 0.75_160");
    }
    GraphMobilenetExample(const GraphMobilenetExample &) = delete;
    GraphMobilenetExample &operator=(const GraphMobilenetExample &) = delete;
    GraphMobilenetExample(GraphMobilenetExample &&)                 = default; // NOLINT
    GraphMobilenetExample &operator=(GraphMobilenetExample &&) = default;      // NOLINT
    ~GraphMobilenetExample() override                          = default;
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(common_params.data_type == DataType::F16 && common_params.target == Target::NEON, "F16 NEON not supported for this graph");

        // Print parameter values
        //std::cout << common_params << std::endl;

        // Get model parameters
        int model_id = model_id_opt->value();

        // Create input descriptor
        unsigned int spatial_size = (model_id == 0 || common_params.data_type == DataType::QASYMM8) ? 224 : 160;

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(spatial_size, spatial_size, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set graph hints
        graph << common_params.target
              << DepthwiseConvolutionMethod::Optimized3x3
              << common_params.fast_math_hint;

        // Create core graph
        if(arm_compute::is_data_type_float(common_params.data_type))
        {
            create_graph_float(input_descriptor, model_id);
        }
        else
        {
            create_graph_qasymm(input_descriptor);
        }

        // Create common tail
        graph << ReshapeLayer(TensorShape(1001U)).set_name("Reshape")
              << SoftmaxLayer().set_name("Softmax")
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
        graph.run();
    }

    private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    SimpleOption<int> *model_id_opt{ nullptr };
    CommonGraphParams  common_params;
    Stream             graph;

    void create_graph_float(TensorDescriptor &input_descriptor, int model_id)
    {
        float       depth_scale = (model_id == 0) ? 1.f : 0.75;
        std::string model_path  = (model_id == 0) ? "/cnn_data/mobilenet_v1_1_224_model/" : "/cnn_data/mobilenet_v1_075_160_model/";

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>();

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Add model path to data path
        if(!data_path.empty())
        {
            data_path += model_path;
        }

        graph << InputLayer(input_descriptor,
                            get_input_accessor(common_params, std::move(preprocessor), false))
              << ConvolutionLayer(
                     3U, 3U, 32U * depth_scale,
                     get_weights_accessor(data_path, "Conv2d_0_weights.npy", DataLayout::NCHW),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
                     .set_name("Conv2d_0")
              << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "Conv2d_0_BatchNorm_moving_mean.npy"),
                     get_weights_accessor(data_path, "Conv2d_0_BatchNorm_moving_variance.npy"),
                     get_weights_accessor(data_path, "Conv2d_0_BatchNorm_gamma.npy"),
                     get_weights_accessor(data_path, "Conv2d_0_BatchNorm_beta.npy"),
                     0.001f)
                     .set_name("Conv2d_0/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name("Conv2d_0/Relu6");
        graph << get_dwsc_node_float(data_path, "Conv2d_1", 64 * depth_scale, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_2", 128 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_3", 128 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_4", 256 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_5", 256 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_6", 512 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_7", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_8", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_9", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_10", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_11", 512 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_12", 1024 * depth_scale, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << get_dwsc_node_float(data_path, "Conv2d_13", 1024 * depth_scale, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::CEIL), PadStrideInfo(1, 1, 0, 0));
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("Logits/AvgPool_1a")
              << ConvolutionLayer(
                     1U, 1U, 1001U,
                     get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_weights.npy", DataLayout::NCHW),
                     get_weights_accessor(data_path, "Logits_Conv2d_1c_1x1_biases.npy"),
                     PadStrideInfo(1, 1, 0, 0))
                     .set_name("Logits/Conv2d_1c_1x1");
    }

    void create_graph_qasymm(TensorDescriptor &input_descriptor)
    {
        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Quantization info taken from the AndroidNN QASYMM8 MobileNet example
        const QuantizationInfo in_quant_info  = QuantizationInfo(0.0078125f, 128);
        const QuantizationInfo mid_quant_info = QuantizationInfo(0.0784313753247f, 128);

        const std::vector<QuantizationInfo> conv_weights_quant_info = {
            QuantizationInfo(0.031778190285f, 156), // conv0
            QuantizationInfo(0.00604454148561f, 66) // conv14
        };

        const std::vector<QuantizationInfo> depth_weights_quant_info = {
            QuantizationInfo(0.254282623529f, 129),  // dwsc1
            QuantizationInfo(0.12828284502f, 172),   // dwsc2
            QuantizationInfo(0.265911251307f, 83),   // dwsc3
            QuantizationInfo(0.0985597148538f, 30),  // dwsc4
            QuantizationInfo(0.0631204470992f, 54),  // dwsc5
            QuantizationInfo(0.0137207424268f, 141), // dwsc6
            QuantizationInfo(0.0817828401923f, 125), // dwsc7
            QuantizationInfo(0.0393880493939f, 164), // dwsc8
            QuantizationInfo(0.211694166064f, 129),  // dwsc9
            QuantizationInfo(0.158015936613f, 103),  // dwsc10
            QuantizationInfo(0.0182712618262f, 137), // dwsc11
            QuantizationInfo(0.0127998134121f, 134), // dwsc12
            QuantizationInfo(0.299285322428f, 161)   // dwsc13
        };

        const std::vector<QuantizationInfo> point_weights_quant_info = {
            QuantizationInfo(0.0425766184926f, 129),  // dwsc1
            QuantizationInfo(0.0250773020089f, 94),   // dwsc2
            QuantizationInfo(0.015851572156f, 93),    // dwsc3
            QuantizationInfo(0.0167811904103f, 98),   // dwsc4
            QuantizationInfo(0.00951790809631f, 135), // dwsc5
            QuantizationInfo(0.00999817531556f, 128), // dwsc6
            QuantizationInfo(0.00590536883101f, 126), // dwsc7
            QuantizationInfo(0.00576109671965f, 133), // dwsc8
            QuantizationInfo(0.00830461271107f, 142), // dwsc9
            QuantizationInfo(0.0152327232063f, 72),   // dwsc10
            QuantizationInfo(0.00741417845711f, 125), // dwsc11
            QuantizationInfo(0.0135628981516f, 142),  // dwsc12
            QuantizationInfo(0.0338749065995f, 140)   // dwsc13
        };

        graph << InputLayer(input_descriptor.set_quantization_info(in_quant_info),
                            get_weights_accessor(data_path, "/cnn_data/mobilenet_qasymm8_model/" + common_params.image))
              << ConvolutionLayer(
                     3U, 3U, 32U,
                     get_weights_accessor(data_path, "/cnn_data/mobilenet_qasymm8_model/Conv2d_0_weights.npy"),
                     get_weights_accessor(data_path, "/cnn_data/mobilenet_qasymm8_model/Conv2d_0_bias.npy"),
                     PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR),
                     1, conv_weights_quant_info.at(0), mid_quant_info)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_1", 64U, PadStrideInfo(1U, 1U, 1U, 1U), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(0), point_weights_quant_info.at(0));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_2", 128U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(1),
                                      point_weights_quant_info.at(1));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_3", 128U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(2),
                                      point_weights_quant_info.at(2));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_4", 256U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(3),
                                      point_weights_quant_info.at(3));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_5", 256U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(4),
                                      point_weights_quant_info.at(4));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_6", 512U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(5),
                                      point_weights_quant_info.at(5));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_7", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(6),
                                      point_weights_quant_info.at(6));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_8", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(7),
                                      point_weights_quant_info.at(7));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_9", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(8),
                                      point_weights_quant_info.at(8));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_10", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(9),
                                      point_weights_quant_info.at(9));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_11", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(10),
                                      point_weights_quant_info.at(10));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_12", 1024U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(11),
                                      point_weights_quant_info.at(11));
        graph << get_dwsc_node_qasymm(data_path, "Conv2d_13", 1024U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(12),
                                      point_weights_quant_info.at(12))
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
              << ConvolutionLayer(
                     1U, 1U, 1001U,
                     get_weights_accessor(data_path, "/cnn_data/mobilenet_qasymm8_model/Logits_Conv2d_1c_1x1_weights.npy"),
                     get_weights_accessor(data_path, "/cnn_data/mobilenet_qasymm8_model/Logits_Conv2d_1c_1x1_bias.npy"),
                     PadStrideInfo(1U, 1U, 0U, 0U), 1, conv_weights_quant_info.at(1));
    }

    BranchLayer get_dwsc_node_float(const std::string &data_path, std::string &&param_path,
                                    unsigned int  conv_filt,
                                    PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info)
    {
        std::string total_path = param_path + "_";
        SubStream   sg(graph);
        sg << DepthwiseConvolutionLayer(
                  3U, 3U,
                  get_weights_accessor(data_path, total_path + "depthwise_depthwise_weights.npy", DataLayout::NCHW),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  dwc_pad_stride_info)
                  .set_name(total_path + "depthwise/depthwise")
           << BatchNormalizationLayer(
                  get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, total_path + "depthwise_BatchNorm_beta.npy"),
                  0.001f)
                  .set_name(total_path + "depthwise/BatchNorm")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(total_path + "depthwise/Relu6")
           << ConvolutionLayer(
                  1U, 1U, conv_filt,
                  get_weights_accessor(data_path, total_path + "pointwise_weights.npy", DataLayout::NCHW),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  conv_pad_stride_info)
                  .set_name(total_path + "pointwise/Conv2D")
           << BatchNormalizationLayer(
                  get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_mean.npy"),
                  get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_moving_variance.npy"),
                  get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_gamma.npy"),
                  get_weights_accessor(data_path, total_path + "pointwise_BatchNorm_beta.npy"),
                  0.001f)
                  .set_name(total_path + "pointwise/BatchNorm")
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f)).set_name(total_path + "pointwise/Relu6");

        return BranchLayer(std::move(sg));
    }

    BranchLayer get_dwsc_node_qasymm(const std::string &data_path, std::string &&param_path,
                                     const unsigned int conv_filt,
                                     PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info,
                                     QuantizationInfo depth_weights_quant_info, QuantizationInfo point_weights_quant_info)
    {
        std::string total_path = "/cnn_data/mobilenet_qasymm8_model/" + param_path + "_";
        SubStream   sg(graph);

        sg << DepthwiseConvolutionLayer(
                  3U, 3U,
                  get_weights_accessor(data_path, total_path + "depthwise_weights.npy"),
                  get_weights_accessor(data_path, total_path + "depthwise_bias.npy"),
                  dwc_pad_stride_info, depth_weights_quant_info)
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f))
           << ConvolutionLayer(
                  1U, 1U, conv_filt,
                  get_weights_accessor(data_path, total_path + "pointwise_weights.npy"),
                  get_weights_accessor(data_path, total_path + "pointwise_bias.npy"),
                  conv_pad_stride_info, 1, point_weights_quant_info)
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f));

        return BranchLayer(std::move(sg));
    }
};
class GraphSqueezenetExample : public Example
{
    public:
    GraphSqueezenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "SqueezeNetV1")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");
        ARM_COMPUTE_EXIT_ON_MSG(common_params.data_type == DataType::F16 && common_params.target == Target::NEON, "F16 NEON not supported for this graph");

        // Print parameter values
        // std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
              << ConvolutionLayer(
                     7U, 7U, 96U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_b.npy"),
                     PadStrideInfo(2, 2, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
              << ConvolutionLayer(
                     1U, 1U, 16U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire2", weights_layout, 64U, 64U);
        graph << ConvolutionLayer(
                     1U, 1U, 16U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire3", weights_layout, 64U, 64U);
        graph << ConvolutionLayer(
                     1U, 1U, 32U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire4", weights_layout, 128U, 128U);
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
              << ConvolutionLayer(
                     1U, 1U, 32U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire5", weights_layout, 128U, 128U);
        graph << ConvolutionLayer(
                     1U, 1U, 48U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire6", weights_layout, 192U, 192U);
        graph << ConvolutionLayer(
                     1U, 1U, 48U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire7", weights_layout, 192U, 192U);
        graph << ConvolutionLayer(
                     1U, 1U, 64U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire8", weights_layout, 256U, 256U);
        graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
              << ConvolutionLayer(
                     1U, 1U, 64U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << get_expand_fire_node(data_path, "fire9", weights_layout, 256U, 256U);
        graph << ConvolutionLayer(
                     1U, 1U, 1000U,
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_b.npy"),
                     PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
              << FlattenLayer()
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
        graph.run();
    }

    private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    BranchLayer get_expand_fire_node(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                                     unsigned int expand1_filt, unsigned int expand3_filt)
    {
        std::string total_path = "/cnn_data/squeezenet_v1.0_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                   1U, 1U, expand1_filt,
                   get_weights_accessor(data_path, total_path + "expand1x1_w.npy", weights_layout),
                   get_weights_accessor(data_path, total_path + "expand1x1_b.npy"),
                   PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                   3U, 3U, expand3_filt,
                   get_weights_accessor(data_path, total_path + "expand3x3_w.npy", weights_layout),
                   get_weights_accessor(data_path, total_path + "expand3x3_b.npy"),
                   PadStrideInfo(1, 1, 1, 1))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }
};
class GraphGooglenetExample : public Example
{
    public:
    GraphGooglenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "GoogleNet")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");
        ARM_COMPUTE_EXIT_ON_MSG(common_params.data_type == DataType::F16 && common_params.target == Target::NEON, "F16 NEON not supported for this graph");

        // Print parameter values
        // std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
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
        graph.run();
    }

    private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

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

int get_temp()
{
    ifstream file("/sys/class/thermal/thermal_zone0/temp");

    int temp = -1;
    file >> temp;
    file.close();
    return temp;
}

#define CPU_SMALL 1
#define CPU_BIG 2
#define GPU 4

static const int ALL = (CPU_BIG | CPU_SMALL | GPU);

struct _config
{
    int    id;
    string name;
    int    argc;
    char **argv;
};

// v3
static std::map<string, std::map<int, double>> delta_temps{
    { "resnet50",
      { { CPU_BIG, 23824.8 },
        { CPU_BIG | CPU_SMALL, 23391.8 },
        { CPU_SMALL, 4103.9 },
        { GPU, 10259.9 },
        { GPU | CPU_BIG, 23155.1 },
        { GPU | CPU_BIG | CPU_SMALL, 25315.1 },
        { GPU | CPU_SMALL, 10454.3 } } },
    { "alexnet",
      { { CPU_BIG, 3996 },
        { CPU_BIG | CPU_SMALL, 13662 },
        { CPU_SMALL, 777.6 },
        { GPU, 8586 },
        { GPU | CPU_BIG, 17355.6 },
        { GPU | CPU_BIG | CPU_SMALL, 19180.8 },
        { GPU | CPU_SMALL, 10486.8 } } },
    { "googlenet",
      { { CPU_BIG, 4849.2 },
        { CPU_BIG | CPU_SMALL, 8121.6 },
        { CPU_SMALL, 442.8 },
        { GPU, 9828 },
        { GPU | CPU_BIG, 14666.4 },
        { GPU | CPU_BIG | CPU_SMALL, 15238.8 },
        { GPU | CPU_SMALL, 11458.8 } } },
    { "mobilenet",
      { { CPU_BIG, 3434.4 },
        { CPU_BIG | CPU_SMALL, 5950.8 },
        { CPU_SMALL, 334.8 },
        { GPU, 5335.2 },
        { GPU | CPU_BIG, 6804 },
        { GPU | CPU_BIG | CPU_SMALL, 7138.8 },
        { GPU | CPU_SMALL, 6933.6 } } },
    { "squeezenet",
      { { CPU_BIG, 1803.6 },
        { CPU_BIG | CPU_SMALL, 4806 },
        { CPU_SMALL, 172.8 },
        { GPU, 7041.6 },
        { GPU | CPU_BIG, 7268.4 },
        { GPU | CPU_BIG | CPU_SMALL, 8456.4 },
        { GPU | CPU_SMALL, 6588 } } }
};
//v3.1
static std::map<string, std::map<int, std::map<int, double>>> delta_temps2{
    { "resnet50",
      { { CPU_SMALL, { { 65000, 4968 }, { 70000, 1416 }, { 75000, 3624 }, { 80000, 1728 } } },
        { CPU_BIG, { { 65000, 18144 }, { 75000, 11088 }, { 80000, 8112 }, { 85000, 4998.86 } } },
        { CPU_BIG | CPU_SMALL, { { 65000, 23328 }, { 80000, 8070.55 }, { 85000, 5481 } } },
        { GPU, { { 65000, 16459.2 }, { 70000, 11324.6 }, { 80000, 3888 } } },
        { GPU | CPU_SMALL, { { 65000, 20088 }, { 70000, 11760 }, { 75000, 11784 }, { 80000, 7776 } } },
        { GPU | CPU_BIG, { { 65000, 24624 }, { 70000, 18360 }, { 75000, 13024.8 }, { 80000, 11880 } } },
        { GPU | CPU_BIG | CPU_SMALL, { { 65000, 25272 }, { 70000, 17403.4 }, { 75000, 10998 } } } } },
    { "alexnet",
      { { CPU_SMALL, { { 50000, 3240 }, { 55000, 648 } } },
        { CPU_BIG, { { 55000, 12204 }, { 65000, 4384.8 }, { 70000, 1458 } } },
        { CPU_BIG | CPU_SMALL, { { 55000, 17280 }, { 60000, 12471.6 } } },
        { GPU, { { 60000, 8586 } } },
        { GPU | CPU_SMALL, { { 60000, 10393.4 }, { 65000, 11016 } } },
        { GPU | CPU_BIG, { { 60000, 17496 }, { 65000, 17145 } } },
        { GPU | CPU_BIG | CPU_SMALL, { { 60000, 22032 }, { 65000, 19030.7 } } } } },
    { "mobilenet",
      { { CPU_SMALL, { { 60000, 334.8 } } },
        { CPU_BIG, { { 60000, 5832 }, { 65000, 5338.29 }, { 70000, 1786.91 } } },
        { CPU_BIG | CPU_SMALL, { { 60000, 6264 }, { 65000, 5872.5 } } },
        { GPU, { { 60000, 5316 }, { 65000, 5508 } } },
        { GPU | CPU_SMALL, { { 60000, 6933.6 } } },
        { GPU | CPU_BIG, { { 60000, 6720 }, { 65000, 7560 } } },
        { GPU | CPU_BIG | CPU_SMALL, { { 60000, 6871.5 }, { 65000, 8208 } } } } },
    { "squeezenet",
      { { CPU_SMALL, { { 60000, 172.8 } } },
        { CPU_BIG, { { 60000, 5400 }, { 65000, 1712.57 }, { 70000, 324 } } },
        { CPU_BIG | CPU_SMALL, { { 60000, 4536 }, { 65000, 4873.5 } } },
        { GPU, { { 60000, 8460 }, { 65000, 6433.71 } } },
        { GPU | CPU_SMALL, { { 60000, 6156 }, { 65000, 6636 } } },
        { GPU | CPU_BIG, { { 60000, 7560 }, { 65000, 7253.05 } } },
        { GPU | CPU_BIG | CPU_SMALL, { { 60000, 14688 }, { 65000, 8128.42 } } } } },
    { "googlenet",
      { { CPU_SMALL, { { 65000, 442.8 } } },
        { CPU_BIG, { { 65000, 12096 }, { 70000, 4360.5 }, { 75000, 1512 } } },
        { CPU_BIG | CPU_SMALL, { { 60000, 21168 }, { 65000, 9806.4 }, { 70000, 6588 } } },
        { GPU, { { 60000, 7128 }, { 65000, 9970.11 } } },
        { GPU | CPU_SMALL, { { 60000, 10977.9 }, { 65000, 14184 } } },
        { GPU | CPU_BIG, { { 60000, 15029.1 }, { 65000, 7776 } } },
        { GPU | CPU_BIG | CPU_SMALL, { { 60000, 15238.8 } } } } },

};

//v4
struct _param
{
    // T(t) = T(0) + b (1 - exp(-t / c))
    double b;
    double c;
};
static std::map<string, std::map<int, _param>> params{
    { "resnet50", { { CPU_BIG, { 1491286.948, 312.361 } }, { CPU_BIG | CPU_SMALL, { 15394.862, 1.425 } }, { CPU_SMALL, { 16143.513, 6.668 } }, { GPU, { 7523.656, 0.460 } }, { GPU | CPU_BIG, { 7871.859, 100.244 } }, { GPU | CPU_BIG | CPU_SMALL, { 8628.666, 0.205 } }, { GPU | CPU_SMALL, { 7404.387, 0.375 } } } },
    { "alexnet", { { CPU_BIG, { 10163.468, 0.915 } }, { CPU_BIG | CPU_SMALL, { 4257.586, 0.168 } }, { CPU_SMALL, { 1219.760, 111.250 } }, { GPU, { 5018.921, 0.265 } }, { GPU | CPU_BIG, { 7382.180, 0.193 } }, { GPU | CPU_BIG | CPU_SMALL, { 7985.138, 0.179 } }, { GPU | CPU_SMALL, { 6264.900, 0.244 } } } },
    { "mobilenet", { { CPU_BIG, { 3200867.422, 470.219 } }, { CPU_BIG | CPU_SMALL, { 2429.987, 26.176 } }, { CPU_SMALL, { 7794853.559, 15217.394 } }, { GPU, { 7738.174, 0.649 } }, { GPU | CPU_BIG, { 6660.583, 0.487 } }, { GPU | CPU_BIG | CPU_SMALL, { 6758.224, 0.393 } }, { GPU | CPU_SMALL, { 6943.555, 0.393 } } } },
    { "squeezenet", { { CPU_BIG, { 5501734.796, 566.798 } }, { CPU_BIG | CPU_SMALL, { 2228.417, 0.083 } }, { CPU_SMALL, { 7790995.764, 8856.862 } }, { GPU, { 6642.212, 0.951 } }, { GPU | CPU_BIG, { 7346.346, 0.380 } }, { GPU | CPU_BIG | CPU_SMALL, { 6436.448, 0.108 } }, { GPU | CPU_SMALL, { 3280.501, 0.318 } } } },
    { "googlenet", { { CPU_BIG, { 162062144.827, 16459.387 } }, { CPU_BIG | CPU_SMALL, { 5986.627, 0.285 } }, { CPU_SMALL, { 10144784.934, 9184.374 } }, { GPU, { 8600.225, 0.575 } }, { GPU | CPU_BIG, { 9148.494, 0.221 } }, { GPU | CPU_BIG | CPU_SMALL, { 11291.538, 0.197 } }, { GPU | CPU_SMALL, { 8836.057, 0.470 } } } },
};

static std::map<string, std::map<int, double>> time_takens = {
    { "resnet50", { { CPU_SMALL, 2.39182 }, { CPU_BIG, 2.51716 }, { GPU, 3.32532 } } },
    { "alexnet", { { CPU_SMALL, 1.53179 }, { CPU_BIG, 1.4873 }, { GPU, 1.45514 } } },
    { "googlenet", { { CPU_SMALL, 0.862833 }, { CPU_BIG, 0.787568 }, { GPU, 1.48101 } } },
    { "mobilenet", { { CPU_SMALL, 0.481851 }, { CPU_BIG, 0.473925 }, { GPU, 0.660021 } } },
    { "squeezenet", { { CPU_SMALL, 0.421066 }, { CPU_BIG, 0.447795 }, { GPU, 0.86846 } } }
};

char **convert(vector<string> argv_list)
{
    char **argv = new char *[argv_list.size()];
    for(unsigned int i = 0; i < argv_list.size(); i++)
    {
        argv[i] = new char[argv_list[i].length() + 1];
        strcpy(argv[i], argv_list[i].c_str());
    }
    return argv;
}

static char **cpu_config = convert({ "", "--target=NEON", "--threads=4" });
static char **gpu_config = convert({ "", "--target=CL" });

_config configs[] = {
    { CPU_SMALL, "CPU Small", 3, cpu_config },
    { CPU_BIG, "CPU Big", 3, cpu_config },
    { GPU, "GPU", 2, gpu_config }
};
std::map<int, int> configs_idx{
    { CPU_SMALL, 0 },
    { CPU_BIG, 1 },
    { GPU, 2 }
};

double fit_temp(double temp, _param param, double time)
{
    // T(t) = T(0) + b (1 - exp(-t / c))
    return temp + param.b * (1 - exp(-time / param.c));
}

//https://stackoverflow.com/a/24413884/1835779
struct MyStreamingHelper
{
    MyStreamingHelper(std::ostream &out1,
                      std::ostream &out2)
        : out1_(out1), out2_(out2)
    {
    }
    std::ostream &out1_;
    std::ostream &out2_;
};

template <typename T>
MyStreamingHelper &operator<<(MyStreamingHelper &h, T const &t)
{
    h.out1_ << t;
    h.out2_ << t;
    return h;
}

MyStreamingHelper &operator<<(MyStreamingHelper &h, ostream &(*f)(ostream &))
{
    h.out1_ << f;
    h.out2_ << f;
    return h;
}

void run_graph(string graph, int argc, char **argv)
{
    if(graph == "resnet50")
    {
        arm_compute::utils::run_example<GraphResNet50Example>(argc, argv);
    }
    else if(graph == "alexnet")
    {
        arm_compute::utils::run_example<GraphAlexnetExample>(argc, argv);
    }
    else if(graph == "mobilenet")
    {
        arm_compute::utils::run_example<GraphMobilenetExample>(argc, argv);
    }
    else if(graph == "squeezenet")
    {
        arm_compute::utils::run_example<GraphSqueezenetExample>(argc, argv);
    }
    else if(graph == "googlenet")
    {
        arm_compute::utils::run_example<GraphGooglenetExample>(argc, argv);
    }
}

void profile_time(MyStreamingHelper &h, string graph)
{
    h << "***\nProfiling Time\n";
    std::map<int, double> samples{
        { CPU_BIG, 0 },
        { CPU_SMALL, 0 },
        { GPU, 0 }
    };
    int       j = 1;
    const int T = 10;
    for(const auto &kv : samples)
    {
        int  config_id = kv.first;
        auto config    = configs[configs_idx[config_id]];
        h << "[" << j++ << "/" << samples.size() << "] Processing " << config.name << "\n";
        for(int i = 0; i < T; i++)
        {
            usleep(1000000);
            auto start = high_resolution_clock::now();

            run_graph(graph, config.argc, config.argv);

            auto   end      = high_resolution_clock::now();
            double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
            h << config.name << " [" << i + 1 << "/" << T << "] " << duration << "\n";
            samples[config_id] += duration;
        }
        h << config.name << " Average " << samples[config_id] / T << "\n";
        time_takens[graph][config_id] = samples[config_id] / T;
    }
}

void profile_temp(MyStreamingHelper &h, unsigned int TL, unsigned int dt, string graph)
{
    h << "***\nProfiling Temperature\n";
    atomic_uint *done = static_cast<atomic_uint *>(mmap(NULL, sizeof *done,
                                                        PROT_READ | PROT_WRITE,
                                                        MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    srand((unsigned)time(NULL));
    for(unsigned int i = 1; i <= ALL; i++)
    {
        cout << "Sleeping until temp < 65000\n";
        while(get_temp() > 65000)
        {
            cout << "\r" << get_temp() << std::flush;
            usleep(1000000);
        }
        cout << "\n";
        h << "[" << i << "/" << ALL << "] "
          << "Processing " << ((i & CPU_BIG) ? "cpu_big_" : "")
          << ((i & CPU_SMALL) ? "cpu_small_" : "")
          << ((i & GPU) ? "gpu" : "") << "\n";
        const int T = 20;
        std::map<int, vector<int>> temp_rise;
        for(int j = 0; j < T; j++)
        {
            *done = 0;
            if(j % 10 == 0 && j > 0)
            {
                // increase temperature after every 10 runs.
                cout << "Running temperature load (100 inferences on CPU BIG)\n";
                int pid = fork();
                if(pid > 0)
                {
                    while(*done == 0)
                    {
                        usleep(1000000);
                    }
                }
                else
                {
                    cpu_set_t mask;
                    CPU_ZERO(&mask);
                    for(int i = 4; i <= 7; i++)
                        CPU_SET(i, &mask);
                    int ret = sched_setaffinity(0, sizeof(mask), &mask);
                    if(ret == -1)
                        printf("Affinity failed [%d]\n", ret);
                    auto config = configs[CPU_BIG];
                    inferences  = 100;
                    run_graph(graph, config.argc, config.argv);
                    *done = 1;
                    exit(0);
                }
            }
            *done   = 0;
            int pid = fork();
            if(pid > 0)
            {
                int temp       = get_temp();
                int max_temp   = temp;
                int start_temp = temp;
                while(*done < i)
                {
                    usleep(dt);
                    int curr_temp = get_temp();
                    max_temp      = max(max_temp, curr_temp);
                }
                int max_delta_temp = max_temp - temp;
                delta_temps[graph][i] += max_delta_temp;
                int index = (int)round((double)start_temp / 5000.0) * 5000;
                h << "[" << j + 1 << "/" << T << "] "
                  << ((i & CPU_BIG) ? "cpu_big_" : "")
                  << ((i & CPU_SMALL) ? "cpu_small_" : "")
                  << ((i & GPU) ? "gpu" : "")
                  << ": start_temp = " << start_temp
                  << ", index = " << index
                  << ", max_temp = " << max_delta_temp << "\n";
                temp_rise[index].push_back(max_delta_temp);
            }
            else
            {
                for(_config &config : configs)
                {
                    int pid = fork();
                    if(pid == 0)
                    {
                        if(i & config.id)
                        {
                            cpu_set_t mask;
                            CPU_ZERO(&mask);
                            if(config.id == CPU_SMALL)
                            {
                                for(int i = 0; i <= 3; i++)
                                    CPU_SET(i, &mask);
                                sched_setaffinity(0, sizeof(mask), &mask);
                            }
                            else if(config.id == CPU_BIG)
                            {
                                for(int i = 4; i <= 7; i++)
                                    CPU_SET(i, &mask);
                                sched_setaffinity(0, sizeof(mask), &mask);
                            }
                            run_graph(graph, config.argc, config.argv);
                            (*done) += config.id;
                        }
                        exit(0);
                    }
                }
                exit(0);
            }
        }
        delta_temps[graph][i] /= T;
        h << ((i & CPU_BIG) ? "cpu_big_" : "")
          << ((i & CPU_SMALL) ? "cpu_small_" : "")
          << ((i & GPU) ? "gpu" : "")
          << " Average "
          << delta_temps[graph][i] << "\n";
        for(auto &kv : temp_rise)
        {
            h << kv.first << "={";

            double sm = 0;
            for(int &v : kv.second)
            {
                sm += v;
                h << v << ",";
            }
            h << "}\n";
            delta_temps2[graph][i][kv.first] = sm / kv.second.size();
            h << "Average @ " << kv.first << " " << (sm / kv.second.size()) << "\n";
        }
        h << "\n";
    }
}

void run_sched(MyStreamingHelper &h, unsigned int TL, unsigned int dt, string graph, string version)
{
    h << "***\nRunning Scheduler\n";
    h << "Images = " << images << "\n";
    h << "Inferences = " << inferences << "\n";

    // Set up flags
    run_cpu_small = static_cast<atomic_uint *>(
        mmap(NULL, sizeof(atomic_uint), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    run_cpu_big = static_cast<atomic_uint *>(
        mmap(NULL, sizeof(atomic_uint), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    run_gpu = static_cast<atomic_uint *>(
        mmap(NULL, sizeof(atomic_uint), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    val = static_cast<atomic_uint *>(
        mmap(NULL, sizeof(atomic_uint), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));

    *run_cpu_small = 0;
    *run_cpu_big   = 0;
    *run_gpu       = 0;
    *val           = 0;

    //3.1 -> 3_1 & 3.2 -> 3_2
    auto   pos          = version.find(".");
    string version_name = version;
    if(pos != string::npos)
    {
        version_name = version.replace(pos, 1, "_");
    }

    // Create Child Processes
    cout << "Sleeping until temp < 65000\n";
    while(get_temp() > 65000)
    {
        cout << "\r" << get_temp() << std::flush;
        usleep(1000000);
    }
    cout << "\n";
    usleep(5000000);
    int pid = fork();
    if(pid > 0)
    {
        auto tbegin = high_resolution_clock::now();

        string   temp_log_file_name = "temp_schedulerv" + version_name + "/" + graph + "_TL" + to_string(TL) + "_dt" + to_string(dt) + ".csv";
        ofstream file(temp_log_file_name);
        h << "Logging temp logs to " << temp_log_file_name << "\n";
        file << "time,temp\n";

        h << "Main thread running\n";
        unsigned int last_time = 0;
        while(*val < images)
        {
            auto   tnow  = high_resolution_clock::now();
            double tdiff = duration_cast<duration<double>>(tnow - tbegin).count();
            double pred_delta_temps[ALL + 1];
            int    temp                = get_temp();
            int    index               = (int)round((double)temp / 5000.0) * 5000;
            double min_pred_delta_temp = 99999999;
            for(int i = 1; i <= ALL; i++)
            {
                if(version == "3")
                {
                    pred_delta_temps[i] = delta_temps[graph][i];
                }
                else if(version == "3.1")
                {
                    pred_delta_temps[i] = delta_temps2[graph][i][index];
                }
                else if(version == "4")
                {
                    bool   cpu_small = i & CPU_SMALL;
                    bool   cpu_big   = i & CPU_BIG;
                    bool   gpu       = i & GPU;
                    double mx_time   = 0;
                    if(cpu_small)
                        mx_time = max(mx_time, time_takens[graph][CPU_SMALL]);
                    if(cpu_big)
                        mx_time = max(mx_time, time_takens[graph][CPU_BIG]);
                    if(gpu)
                        mx_time = max(mx_time, time_takens[graph][GPU]);

                    pred_delta_temps[i] = fit_temp(temp, params[graph][i], mx_time) - temp;
                }
                min_pred_delta_temp = min(min_pred_delta_temp, pred_delta_temps[i]);
            }
            if(temp + min_pred_delta_temp < TL)
            {
                int    delta_temp     = TL - temp;
                bool   min_cpu_small  = false;
                bool   min_cpu_big    = false;
                bool   min_gpu        = false;
                double min_total_time = 99999999;
                for(int i = 1; i <= ALL; i++)
                {
                    bool cpu_small = i & CPU_SMALL;
                    bool cpu_big   = i & CPU_BIG;
                    bool gpu       = i & GPU;
                    if((*run_cpu_small && !cpu_small)
                       || (*run_cpu_big && !cpu_big)
                       || (*run_gpu && !gpu))
                    {
                        continue;
                    }
                    if(pred_delta_temps[i] < delta_temp)
                    {
                        double total_time = 0;
                        if(cpu_small)
                            total_time += 1.0 / time_takens[graph][CPU_SMALL];
                        if(cpu_big)
                            total_time += 1.0 / time_takens[graph][CPU_BIG];
                        if(gpu)
                            total_time += 1.0 / time_takens[graph][GPU];
                        total_time = 1 / total_time;
                        if(total_time < min_total_time)
                        {
                            min_cpu_small  = cpu_small;
                            min_cpu_big    = cpu_big;
                            min_gpu        = gpu;
                            min_total_time = total_time;
                        }
                    }
                }
                *run_cpu_small = min_cpu_small ? 1 : 0;
                *run_cpu_big   = min_cpu_big ? 1 : 0;
                *run_gpu       = min_gpu ? 1 : 0;
            }
            else
            {
                // stop any new inferences
                *run_cpu_small = 0;
                *run_cpu_big   = 0;
                *run_gpu       = 0;
            }
            if(tdiff > last_time + 5)
            {
                h << "tdiff = " << tdiff << ", "
                  << "run_cpu_small = " << *run_cpu_small << ", "
                  << "run_cpu_big = " << *run_cpu_big << ", "
                  << "run_gpu = " << *run_gpu << ", "
                  << "temp = " << temp << ", "
                  << "min_delta_t = " << min_pred_delta_temp << ", "
                  << "val = " << *val << "\n";
                last_time += 5;
            }
            file << tdiff << ", " << temp << "\n";
            usleep(dt);
        }
        auto   tend  = high_resolution_clock::now();
        double gross = duration_cast<duration<double>>(tend - tbegin).count();
        double cost  = gross / images;
        h << cost << " per image" << endl;
        cost /= inferences;
        h << cost << " per inference" << endl;
    }
    else
    {
        for(_config const &config : configs)
        {
            int pid = fork();
            if(pid == 0)
            {
                h << config.name << ": Started process\n";

                cpu_set_t mask;
                CPU_ZERO(&mask);

                if(config.id == CPU_SMALL)
                {
                    for(int i = 0; i <= 3; i++)
                        CPU_SET(i, &mask);
                    sched_setaffinity(0, sizeof(mask), &mask);
                }
                else if(config.id == CPU_BIG)
                {
                    for(int i = 4; i <= 7; i++)
                        CPU_SET(i, &mask);
                    sched_setaffinity(0, sizeof(mask), &mask);
                }
                while(true)
                {
                    if(*val >= images)
                    {
                        break;
                    }
                    if((config.id == CPU_SMALL && *run_cpu_small) || (config.id == CPU_BIG && *run_cpu_big) || (config.id == GPU && *run_gpu))
                    {
                        //h << config.name << ": Running\n";
                        run_graph(graph, config.argc, config.argv);
                        (*val)++;
                        //h << "Done" << *val << "\n";
                    }
                    else
                    {
                        usleep(dt);
                    }
                }
                exit(0);
            }
        }
    }
}

/** Main program for ResNet50
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

    ToggleOption *helpOption = parser.add_option<ToggleOption>("help");

    ToggleOption *profileTempOption = parser.add_option<ToggleOption>("profile-temp");
    ToggleOption *profileTimeOption = parser.add_option<ToggleOption>("profile-time");
    ToggleOption *runSchedOption    = parser.add_option<ToggleOption>("run-sched");

    SimpleOption<unsigned int> *imagesOption     = parser.add_option<SimpleOption<unsigned int>>("n", 100);
    SimpleOption<unsigned int> *inferencesOption = parser.add_option<SimpleOption<unsigned int>>("i", 1);
    SimpleOption<unsigned int> *tlOption         = parser.add_option<SimpleOption<unsigned int>>("tl", 80000);
    SimpleOption<unsigned int> *dtOption         = parser.add_option<SimpleOption<unsigned int>>("dt", 10000);

    set<string> graphs   = { "resnet50", "googlenet", "mobilenet", "squeezenet", "alexnet" };
    set<string> versions = { "3", "3.1", "4" };

    SimpleOption<string> *graphOption   = parser.add_option<EnumOption<string>>("graph", graphs);
    SimpleOption<string> *versionOption = parser.add_option<EnumOption<string>>("version", versions);

    helpOption->set_help("Help");

    imagesOption->set_help("Images");
    inferencesOption->set_help("Inferences");
    tlOption->set_help("Temperature Threshold");
    dtOption->set_help("Scheduler Interval");

    profileTempOption->set_help("Profile Temperature");
    profileTimeOption->set_help("Profile Time");
    runSchedOption->set_help("Run Scheduler");

    graphOption->set_help("Graph");
    versionOption->set_help("Scheduler Version");

    parser.parse(argc, argv);

    bool help = helpOption->is_set() ? helpOption->value() : false;
    if(help)
    {
        parser.print_help(argv[0]);
        return 0;
    }

    images          = imagesOption->value();
    inferences      = inferencesOption->value();
    unsigned int TL = tlOption->value();
    unsigned int dt = dtOption->value();

    string graph   = graphOption->value();
    string version = versionOption->value();

    bool is_profile_time = profileTimeOption->is_set() && profileTimeOption->value();
    bool is_profile_temp = profileTempOption->is_set() && profileTempOption->value();
    bool is_run_sched    = runSchedOption->is_set() && runSchedOption->value();

    string suffix = "";
    if(is_profile_time)
        suffix += "_profile_time";
    if(is_profile_temp)
        suffix += "_profile_temp";
    if(is_run_sched)
        suffix += "_run_sched" + version;

    ofstream fl;
    fl.open("temp_scheduler_all/" + graph + "_TL" + to_string(TL) + "_dt" + to_string(dt) + suffix + ".log");
    MyStreamingHelper h(fl, cout);

    h << "TL = " << TL << "\n";
    h << "graph = " << graph << "\n";
    h << "version = " << version << "\n";

    if(is_profile_time)
    {
        profile_time(h, graph);
    }
    if(is_profile_temp)
    {
        profile_temp(h, TL, dt, graph);
    }
    if(is_run_sched)
    {
        run_sched(h, TL, dt, graph, version);
    }
}
