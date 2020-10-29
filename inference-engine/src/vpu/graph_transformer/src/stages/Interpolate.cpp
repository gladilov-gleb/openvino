// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <ie_common.h>
#include <ie_blob.h>
#include <vector>
#include <unordered_set>
#include <memory>
#include <set>
#include <string>
#include <ngraph/opsets/opset4.hpp>

using namespace InferenceEngine;

namespace vpu {
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

namespace {
class InterpolateStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<InterpolateStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();

        auto perm = input->desc().dimsOrder().toPermutation();
        IE_ASSERT(perm.size() <= 4);

        auto antialias = attrs().get<bool>("antialias");
        auto cube_coeff = attrs().get<float>("cube_coeff");
        auto batch = attrs().get<int>("batch");
        auto sampleType = attrs().get<int>("type");
        auto sampleNearestMode = attrs().get<int>("nearestMode");
        auto sampleShapeCalcMode = attrs().get<int>("shapeCalcMode");
        auto sampleCoordTransMode = attrs().get<int>("coordTransMode");
        auto& pads_begin = attrs().get<DimValues>("pads_begin");
        auto& pads_end = attrs().get<DimValues>("pads_end");
        auto& sizes = attrs().get<DimValues>("sizes");
        auto& axis = attrs().get<DimValues>("InterpolateAxis");
        auto& scales = attrs().get<DimValues>("InterpolateScales");

        serializer.append(static_cast<int>(antialias));
        serializer.append(static_cast<float>(cube_coeff));
        serializer.append(static_cast<int>(batch));
        serializer.append(static_cast<int>(sampleType));
        serializer.append(static_cast<int>(sampleNearestMode));
        serializer.append(static_cast<int>(sampleShapeCalcMode));
        serializer.append(static_cast<int>(sampleCoordTransMode));

        for (int i = 0; i < perm.size(); ++i) {
            serializer.append(static_cast<int>(pads_begin.get(perm[i], 0)));
            serializer.append(static_cast<int>(pads_end.get(perm[i], 0)));
            serializer.append(static_cast<int>(sizes.get(perm[i], 0)));
            serializer.append(static_cast<int>(axis.get(perm[i], 0)));
            serializer.append(static_cast<float>(scales.get(perm[i], 0)));
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

VPU_DECLARE_ENUM(ResampleType,
    Nearest  = 0,  // Currently this is only one supported
    Linear = 1,
    Cubic = 2
)

class ResampleStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ResampleStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto antialias = attrs().get<bool>("antialias");
        auto factor = attrs().get<float>("factor");
        auto sampleType = attrs().get<ResampleType>("type");

        serializer.append(static_cast<int32_t>(antialias));
        serializer.append(static_cast<float>(factor));
        serializer.append(static_cast<uint32_t>(sampleType));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};
}  // namespace

Stage StageBuilder::addInterpolateStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& output,
        const std::string& origin) {
    Stage interpolateStage = model->addNewStage<InterpolateStage>(
        name,
        StageType::Interpolate,
        layer,
        {input},
        {output});
    interpolateStage->attrs().set<std::string>("origin", origin);

    return interpolateStage;
}

void FrontEnd::parseInterpolate(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    // IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    using ngraph::opset4::Interpolate;
    const auto interpolateMode = _layer->GetParamAsString("mode");
    const auto antialias = _layer->GetParamAsBool("antialias", false);

    if (interpolateMode == "nearest") {
        auto stage = model->addNewStage<ResampleStage>(_layer->name, StageType::Resample, _layer, {inputs[0]}, outputs);

        stage->attrs().set<bool>("antialias", antialias);
        stage->attrs().set<float>("factor", -1.0f);
        stage->attrs().set<ResampleType>("type", ResampleType::Nearest);
        return;
    }

#if 0
    DimValues pads_begin;
    pads_begin.set(Dim::W, layer->pads_begin[3]);
    pads_begin.set(Dim::H, layer->pads_begin[2]);
    pads_begin.set(Dim::C, layer->pads_begin[1]);
    pads_begin.set(Dim::N, layer->pads_begin[0]);

    DimValues pads_end;
    pads_end.set(Dim::W, layer->pads_end[3]);
    pads_end.set(Dim::H, layer->pads_end[2]);
    pads_end.set(Dim::C, layer->pads_end[1]);
    pads_end.set(Dim::N, layer->pads_end[0]);

    DimValues sizes;
    sizes.set(Dim::W, layer->sizes[3]);
    sizes.set(Dim::H, layer->sizes[2]);
    sizes.set(Dim::C, layer->sizes[1]);
    sizes.set(Dim::N, layer->sizes[0]);

    DimValues scales;
    scales.set(Dim::W, layer->scales[3]);
    scales.set(Dim::H, layer->scales[2]);
    scales.set(Dim::C, layer->scales[1]);
    scales.set(Dim::N, layer->scales[0]);

    DimValues axes;
    axes.set(Dim::W, layer->axes[3]);
    axes.set(Dim::H, layer->axes[2]);
    axes.set(Dim::C, layer->axes[1]);
    axes.set(Dim::N, layer->axes[0]);

    auto stage = model->addNewStage<InterpolateStage>(_layer->name, StageType::Interpolate, _layer, inputs, outputs);

    stage->attrs().set<bool>("antialias", _layer->GetParamAsInt("antialias", 0));
    stage->attrs().set<float>("cube_coeff", _layer->GetParamAsFloat("cube_coeff", 0));
    stage->attrs().set<int>("batch", _layer->GetParamAsInt("batch", 1));
    stage->attrs().set<int>("type", _layer->GetParamAsInt("type", 0));

    stage->attrs().set<DimValues>("pads_begin", pads_begin);
    stage->attrs().set<DimValues>("pads_end", pads_end);
    stage->attrs().set<DimValues>("sizes", sizes);

    stage->attrs().set<int>("nearestMode", _layer->GetParamAsInt("nearestMode", 0));
    stage->attrs().set<int>("shapeCalcMode", _layer->GetParamAsInt("shapeCalcMode", 0));
    stage->attrs().set<int>("coordTransMode", _layer->GetParamAsInt("coordTransMode", 0));
    stage->attrs().set<DimValues>("InterpolateAxis", axes);
    stage->attrs().set<DimValues>("InterpolateScales", scales);
#endif
}

}  // namespace vpu
