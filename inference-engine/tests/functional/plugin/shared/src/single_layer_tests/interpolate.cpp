// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/interpolate.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

std::string InterpolateLayerTest::getTestCaseName(testing::TestParamInfo<InterpolateLayerTestParams> obj) {
    InterpolateSpecificParamsForTests interpolateParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes, targetShapes;
    std::string targetDevice;
    std::tie(netPrecision, inLayout, outLayout, inputShapes, targetShapes, targetDevice) = obj.param;
    std::vector<size_t> padBegin, padEnd;
    std::vector<int64_t> axes = {0, 1, 2, 3};
    std::vector<float> scales = {2, 4, 1, 1};
    bool antialias = false;
    ngraph::op::v4::Interpolate::InterpolateMode mode = ngraph::op::v4::Interpolate::InterpolateMode::nearest;
    ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode::scales;
    ngraph::op::v4::Interpolate::CoordinateTransformMode coordinateTransformMode = ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel;
    ngraph::op::v4::Interpolate::NearestMode nearestMode = ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor;
    double cubeCoef = -0.75;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "TS=" << CommonTestUtils::vec2str(targetShapes) << "_";
    result << "InterpolateMode=" << int(mode) << "_";
    result << "ShapeCalcMode=" << int(shapeCalcMode) << "_";
    result << "CoordinateTransformMode=" << int(coordinateTransformMode) << "_";
    result << "NearestMode=" << int(nearestMode) << "_";
    result << "CubeCoef=" << cubeCoef << "_";
    result << "Antialias=" << antialias << "_";
    result << "PB=" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE=" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "Axes=" << CommonTestUtils::vec2str(axes) << "_";
    result << "Scales=" << CommonTestUtils::vec2str(scales) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void InterpolateLayerTest::SetUp() {
    InterpolateSpecificParamsForTests interpolateParams;
    std::vector<size_t> inputShape, targetShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(netPrecision, inLayout, outLayout, inputShape, targetShape, targetDevice) = this->GetParam();
    std::vector<size_t> padBegin = std::vector<size_t>(1, 0), padEnd = std::vector<size_t>(1, 0);
    // std::vector<int64_t> axes = {0, 1, 2, 3};
    // std::vector<float> scales = {2, 4, 1, 1};
    std::vector<float> scales = {2, 2};
    std::vector<int64_t> axes = {2, 3};
    bool antialias = false;
    ngraph::op::v4::Interpolate::InterpolateMode mode = ngraph::op::v4::Interpolate::InterpolateMode::nearest;
    ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode::scales;
    ngraph::op::v4::Interpolate::CoordinateTransformMode coordinateTransformMode = ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel;
    ngraph::op::v4::Interpolate::NearestMode nearestMode = ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor;

    double cubeCoef = -0.75;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto sizesConst = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {targetShape.size()}, targetShape);
    auto sizesInput = std::make_shared<ngraph::opset3::Constant>(sizesConst);

    auto scales_const = ngraph::opset3::Constant(ngraph::element::Type_t::f32, {scales.size()}, scales);
    auto scalesInput = std::make_shared<ngraph::opset3::Constant>(scales_const);

    auto axesConst = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {axes.size()}, axes);
    auto axesInput = std::make_shared<ngraph::opset3::Constant>(axesConst);

    ngraph::op::v4::Interpolate::InterpolateAttrs interpolateAttributes{mode, shapeCalcMode, padBegin,
        padEnd, coordinateTransformMode, nearestMode, antialias, cubeCoef};
    auto interpolate = std::make_shared<ngraph::op::v4::Interpolate>(params[0],
                                                                     sizesInput,
                                                                     scalesInput,
                                                                     axesInput,
                                                                     interpolateAttributes);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(interpolate)};
    function = std::make_shared<ngraph::Function>(results, params, "interpolate");
}

TEST_P(InterpolateLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
