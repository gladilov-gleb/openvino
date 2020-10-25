// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND 1e-3

// PRETTY_PARAM(Factor, float)
PRETTY_PARAM(Antialias, int)
// PRETTY_PARAM(cube_coeff, float)
// PRETTY_PARAM(batch, int)
// PRETTY_PARAM(type, int)
PRETTY_PARAM(nearestMode, int)
PRETTY_PARAM(shapeCalcMode, int)
PRETTY_PARAM(coordTransMode, int)
// PRETTY_PARAM(pads_begin, float)
// PRETTY_PARAM(pads_end, float)
// PRETTY_PARAM(sizes, int)
PRETTY_PARAM(InterpolateAxis, int)
PRETTY_PARAM(InterpolateScales, float)
PRETTY_PARAM(HwOptimization, bool)
PRETTY_PARAM(layoutPreference, vpu::LayoutPreference)
PRETTY_PARAM(CustomConfig, std::string)

enum class InterpolateMode {
    nearest     = 0,
    linear      = 1,
    linear_onnx = 2,
    cubic       = 3
};

enum class InterpolateShapeCalcMode {
    sizes  = 0,
    scales = 1
};

enum class InterpolateCoordTransMode {
    half_pixel           = 0,
    pytorch_half_pixel   = 1,
    asymmetric           = 2,
    tf_half_pixel_for_nn = 3,
    align_corners        = 4
};

enum class InterpolateNearestMode {
    round_prefer_floor = 0,
    round_prefer_ceil  = 1,
    floor              = 2,
    ceil               = 3,
    simple             = 4
};

std::string testModel = R"V0G0N(
    <net batch="1" name="ctpn" version="4">
    <layers>
    <layer id="0" name="Input0" precision="FP16" type="Input">
        <output>
        <port id="0">
            <dim>1</dim>
            <dim>2</dim>
            <dim>48</dim>
            <dim>80</dim>
        </port>
        </output>
    </layer>
    <layer id="1" name="Input1" precision="FP16" type="Input">
        <output>
        <port id="0">
            <dim>96</dim>
            <dim>160</dim>   <!-- sizes  -->
        </port>
        </output>
    </layer>
    <layer id="2" name="Input2" precision="FP16" type="Input">
        <output>
        <port id="0">
            <dim>2.0</dim>
            <dim>2.0</dim>   <!-- scales  -->
        </port>
        </output>
    </layer>
    <layer id="3" name="Input3" precision="FP16" type="Input">
        <output>
        <port id="0">
            <dim>2</dim>
            <dim>3</dim>    <!-- axes  -->
        </port>
        </output>
    </layer>
    <layer id="4" name="interpolate" precision="FP16" type="Interpolate">
        <data shape_calculation_mode="scales" pads_begin="0" pads_end="0" mode="linear"/>
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>2</dim>
                <dim>96</dim>
                <dim>80</dim>
            </port>
            <port id="1">
                <dim>2</dim>
            </port>
            <port id="2">
                <dim>2</dim>
            </port>
            <port id="3">
                <dim>2</dim>
            </port>
        </input>
        <output>
            <port id="0"  precision="FP32">
                <dim>1</dim>
                <dim>2</dim>
                <dim>24</dim>
                <dim>160</dim>
            </port>
        </output>
    </layer>
    </layers>
    <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
    <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
    <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    </net>
    )V0G0N";

typedef myriadLayerTestBaseWithParam<std::tuple<SizeVector, Antialias, nearestMode, shapeCalcMode, coordTransMode, InterpolateAxis, InterpolateScales, HwOptimization, layoutPreference, CustomConfig>>
	myriadInterpolateLayerTests_smoke;

float getOriginalCoordinate(float x_resized, float x_scale, int length_resized, int length_original, int mode) {
    switch(mode) {
        case static_cast<int>(InterpolateCoordTransMode::half_pixel) : {
            return (x_scale != 0)? ((x_resized + 0.5) / x_scale) - 0.5 : 0.0f;
            break;
        }
        case static_cast<int>(InterpolateCoordTransMode::pytorch_half_pixel) : {
            if (length_resized > 1) {
                return (x_scale != 0)? ((x_resized + 0.5) / x_scale) - 0.5 : 0.0f;
            } else {
                return 0.0f;
            }
            break;
        }
        case static_cast<int>(InterpolateCoordTransMode::asymmetric) : {
            return (x_scale != 0)? (x_resized / x_scale) : 0.0f;
            break;
        }
        case static_cast<int>(InterpolateCoordTransMode::tf_half_pixel_for_nn) : {
            return (x_scale != 0)? ((x_resized + 0.5) / x_scale) : 0.0f;
            break;
        }
        case static_cast<int>(InterpolateCoordTransMode::align_corners) : {
            if (length_resized - 1 == 0) {
                return 0.0f;
            } else {
                return x_resized * static_cast<float>(length_original - 1) / (length_resized - 1);
            }
            break;
        }
        default: {
            std::cout << "Interpolate does not support this coordinate transformation mode";
            return 0.0f;
            break;
        }
    }
}

int getNearestPixel(float originalValue, bool isDownsample, int mode) {
    switch (mode) {
        case static_cast<int>(InterpolateNearestMode::round_prefer_floor): {
            if (originalValue == (static_cast<int>(originalValue) + 0.5f)) {
                return static_cast<int>(std::floor(originalValue));
            } else {
                return static_cast<int>(std::round(originalValue));
            }
            break;
        }
        case static_cast<int>(InterpolateNearestMode::round_prefer_ceil): {
            return static_cast<int>(std::round(originalValue));
            break;
        }
        case static_cast<int>(InterpolateNearestMode::floor): {
            return static_cast<int>(std::floor(originalValue));
            break;
        }
        case static_cast<int>(InterpolateNearestMode::ceil): {
            return static_cast<int>(std::ceil(originalValue));
            break;
        }
        case static_cast<int>(InterpolateNearestMode::simple): {
            if (isDownsample) {
                return static_cast<int>(std::ceil(originalValue));
            } else {
                return static_cast<int>(originalValue);
            }
        }
        default: {
            std::cout << "Interpolate does not support this nearest round mode";
            return 0;
            break;
        }
    }
}

int changeCoord(int length, int pos) {
    return std::max(static_cast<int>(0), std::min(pos, length - 1));
}

static inline float triangleCoeff(float x) {
    return (1.0f - fabsf(x));
}

void refNearestInterpolate(const Blob::Ptr src, Blob::Ptr dst, int antialias) {
    ie_fp16 *src_data = static_cast<ie_fp16*>(src->buffer());
    ie_fp16 *output_sequences = static_cast<ie_fp16*>(dst->buffer());
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(output_sequences, nullptr);

    const auto& src_dims = src->getTensorDesc().getDims();
    const auto& dst_dims = dst->getTensorDesc().getDims();
    int OH = dst_dims[2];
    int OW = dst_dims[3];

    int C  = src_dims[4];
    int IH = src_dims[0];
    int IW = src_dims[1];
    int OD = 1, ID = 1;

    if (IH == OH && IW == OW)
    {
    	std::copy(src_data, src_data + C*IH*IW, output_sequences);
        return;
    }

    const float fy = static_cast<float>(IH) / static_cast<float>(OH);
    const float fx = static_cast<float>(IW) / static_cast<float>(OW);
    const float fz = 1;
    
    std::vector<int> ind(OD + OH + OW, 1);
    bool isDDownsample = (fz < 1) ? true : false;
    bool isHDownsample = (fy < 1) ? true : false;
    bool isWDownsample = (fx < 1) ? true : false;

    for (int oz = 0; oz < OD; oz++) {
        float iz = getOriginalCoordinate(float(oz), fz, OD, ID, 0);
        ind[oz] = getNearestPixel(iz, isDDownsample, 0);
        ind[oz] = changeCoord(ind[oz], ID);
    }
    for (int oy = 0; oy < OH; oy++) {
        float iy = getOriginalCoordinate(float(oy), fy, OH, IH, 0);
        ind[OD + oy] = getNearestPixel(iy, isHDownsample, 0);
        ind[OD + oy] = changeCoord(ind[OD + oy], IH);
    }
    for (int ox = 0; ox < OW; ox++) {
        float ix = getOriginalCoordinate(float(ox), fx, OW, IW, 0);
        ind[OD + OH + ox] = getNearestPixel(ix, isWDownsample, 0);
        ind[OD + OH + ox] = changeCoord(ind[OD + OH + ox], IW);
    }
    int *index_d = static_cast<int*>(&ind[0]);
    int *index_h = static_cast<int*>(&ind[OD]);
    int *index_w = static_cast<int*>(&ind[OD + OH]);

    for (int c = 0; c < C; c++) {
        const ie_fp16* in_ptr = src_data + IW * IH * c;
        ie_fp16* out_ptr = output_sequences + OW * OH * c;
        for (int od = 0; od < OD; od++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    out_ptr[oh * OW + ow] = in_ptr[index_h[oh] * IW + index_w[ow]];
                    // std::cout << "out_ptr[oh * OW + ow] = " << out_ptr[oh * OW + ow] << "\n";
                }
            }
        }
    }
}

TEST_P(myriadInterpolateLayerTests_smoke, Interpolate) {
    const SizeVector inputDims = std::get<0>(GetParam());
    const bool antialias = std::get<1>(GetParam());
    const int sampleNearestMode = std::get<2>(GetParam());
    const int sampleShapeCalcMode = std::get<3>(GetParam());
    const int sampleCoordTransMode = std::get<4>(GetParam());
    const int axis = std::get<5>(GetParam());
    const float scales = std::get<6>(GetParam());
    const bool hwOptimization = std::get<7>(GetParam());
    auto layoutPreference = std::get<8>(GetParam());
    const std::string customConfig = std::get<9>(GetParam());
    int sample = 0;
    printf("TEST_P\n");
    ASSERT_GT(scales, 0);

    if (customConfig.empty() && antialias) {
        GTEST_SKIP() << "Native Interpolate with antialiasing is not supported";
    }

    if (!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
    }

    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    const auto outputDims = SizeVector{inputDims[0],
                                       inputDims[1],
                                       (size_t)(inputDims[2] * scales),
                                       (size_t)(inputDims[3] * scales)};

    SetInputTensors({inputDims});
    SetOutputTensors({outputDims});

    std::map<std::string, std::string> params;
    params["antialias"] = std::to_string((int)antialias);
    params["sampleNearestMode"] = std::to_string(sampleNearestMode);
    params["sampleShapeCalcMode"] = std::to_string(sampleShapeCalcMode);
    params["sampleCoordTransMode"] = std::to_string(sampleCoordTransMode);
    params["axis"] = std::to_string(axis);
    params["scales"] = std::to_string(scales);

    // ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Interpolate").params(params),
    //                                                NetworkInitParams()
    //                                                     .useHWOpt(hwOptimization)
    //                                                     .lockLayout(true)));

    makeSingleLayerNetwork(LayerInitParams("Interpolate").params(params));
    // makeSingleLayerNetwork(LayerInitParams("Interpolate").params(params), NetworkInitParams().layoutPreference(layoutPreference));

    SetFirstInputToRange(-0.9f, 0.9f);
    printf("TEST_P before Infer\n");
    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refNearestInterpolate(_inputMap.begin()->second, _refBlob, antialias));
    printf("TEST_P before compare\n");

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<SizeVector> s_InterpolateInput = {
        {1, 1, 1, 1},
        {1, 1, 52, 52},
        {1, 1, 14, 14},
};

static std::vector<CustomConfig> s_CustomConfig = {
    {""},
#ifdef VPU_HAS_CUSTOM_KERNELS
   getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};

