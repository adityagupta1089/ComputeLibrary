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

// maximum rise in temperatures

struct _param
{
    // T(t) = T(0) + b (1 - exp(-t / c))
    double b;
    double c;
};
static std::map<int, _param> params{
    { CPU_BIG, { 1491286.948, 312.361 } },
    { CPU_BIG | CPU_SMALL, { 15394.862, 1.425 } },
    { CPU_SMALL, { 16143.513, 6.668 } },
    { GPU, { 7523.656, 0.460 } },
    { GPU | CPU_BIG, { 7871.859, 100.244 } },
    { GPU | CPU_BIG | CPU_SMALL, { 8628.666, 0.205 } },
    { GPU | CPU_SMALL, { 7404.387, 0.375 } }
};

struct _config
{
    int    id;
    string name;
    int    argc;
    char **argv;
    double time_taken;
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
    { CPU_SMALL, "CPU Small", 3, cpu_config, 2.39182 },
    { CPU_BIG, "CPU Big", 3, cpu_config, 2.51716 },
    { GPU, "GPU", 2, gpu_config, 3.32532 }
};
std::map<int, int> configs_idx{
    { CPU_SMALL, 0 },
    { CPU_BIG, 1 },
    { GPU, 2 }
};

static int TL = 100000, dt = 10000, sdt = 5000000;

double fit_temp(double temp, _param param, double time)
{
    // T(t) = T(0) + b (1 - exp(-t / c))
    return temp + param.b * (1 - exp(-time / param.c));
}

void run_sched()
{
    cout << "***\nRunning Scheduler\n";
    cout << "Images = " << images << "\n";
    cout << "Inferences = " << inferences << "\n";

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

    // Create Child Processes
    cout << "Sleeping for 5 seconds\n";
    usleep(sdt);
    int pid = fork();
    if(pid > 0)
    {
        auto     tbegin = high_resolution_clock::now();
        ofstream file("temp_schedulerv4/resnet50_temp.csv");
        file << "time, temp\n";
        cout << "Main thread running\n";
        unsigned int last_time = 0;
        while(*val < images)
        {
            auto   tnow  = high_resolution_clock::now();
            double tdiff = duration_cast<duration<double>>(tnow - tbegin).count();
            int    temp  = get_temp();
            std::map<int, double> fit_temps;
            double min_fit_temp = 9999999999;
            for(const auto &param : params)
            {
                bool cpu_small = param.first & CPU_SMALL;
                bool cpu_big = param.first & CPU_BIG;
                bool gpu = param.first & GPU;
                double mx_time = 0;
                if (cpu_small) mx_time = max(mx_time, configs[configs_idx[CPU_SMALL]].time_taken);
                if (cpu_big) mx_time = max(mx_time, configs[configs_idx[CPU_BIG]].time_taken);
                if (gpu) mx_time = max(mx_time, configs[configs_idx[GPU]].time_taken);
                fit_temps[param.first] = fit_temp(temp, param.second, mx_time);
                min_fit_temp           = min(min_fit_temp, fit_temps[param.first]);
            }
            if(min_fit_temp < TL)
            {
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
                    if(fit_temps[i] < TL)
                    {
                        double total_time = 0;
                        if(cpu_small)
                            total_time += configs[configs_idx[CPU_SMALL]].time_taken;
                        if(cpu_big)
                            total_time += configs[configs_idx[CPU_BIG]].time_taken;
                        if(gpu)
                            total_time += configs[configs_idx[GPU]].time_taken;
                        if(total_time < min_total_time)
                        {
                            min_cpu_small = cpu_small;
                            min_cpu_big   = cpu_big;
                            min_gpu       = gpu;
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
                cout << "tdiff = " << tdiff << ", "
                     << "run_cpu_small = " << *run_cpu_small << ", "
                     << "run_cpu_big = " << *run_cpu_big << ", "
                     << "run_gpu = " << *run_gpu << ", "
                     << "temp = " << temp << ", "
                     << "val = " << *val << "\n";
                last_time += 5;
            }
            file << tdiff << ", " << temp << "\n";
            usleep(dt);
        }
        auto   tend  = high_resolution_clock::now();
        double gross = duration_cast<duration<double>>(tend - tbegin).count();
        double cost  = gross / images;
        cout << cost << " per image" << endl;
        cost /= inferences;
        cout << cost << " per inference" << endl;
    }
    else
    {
        for(_config const &config : configs)
        {
            int pid = fork();
            if(pid == 0)
            {
                cout << config.name << ": Started process\n";

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
                        //cout << config.name << ": Running\n";
                        arm_compute::utils::run_example<GraphResNet50Example>(config.argc, config.argv);
                        (*val)++;
                        //cout << "Done" << *val << "\n";
                    }
                    else
                    {
                        usleep(1000);
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

    SimpleOption<unsigned int> *imagesOption     = parser.add_option<SimpleOption<unsigned int>>("n", 100);
    SimpleOption<unsigned int> *inferencesOption = parser.add_option<SimpleOption<unsigned int>>("i", 1);

    helpOption->set_help("Help");

    imagesOption->set_help("Images");
    inferencesOption->set_help("Inferences");

    parser.parse(argc, argv);

    bool help = helpOption->is_set() ? helpOption->value() : false;
    if(help)
    {
        parser.print_help(argv[0]);
        return 0;
    }

    images     = imagesOption->value();
    inferences = inferencesOption->value();

    cout << "TL = " << TL << "\n";

    run_sched();
}
