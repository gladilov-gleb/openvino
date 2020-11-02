// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_broadcast.hpp"

#include "vpu/ngraph/operations/static_shape_broadcast.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include "vpu/utils/error.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset5.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeGatherElements(std::shared_ptr<ngraph::Node> target) {
    const auto broadcast = ngraph::as_type_ptr<ngraph::opset5::GatherElements>(target);
    VPU_THROW_UNLESS(broadcast,
                     "dynamicToStaticShapeBroadcast transformation is not applicable for {}, "
                     "it should be {} instead",
                     target, ngraph::opset5::GatherElements::type_info);

    auto shapeToConstant = [&broadcast](const ngraph::Output<ngraph::Node>& output,
                                        const ngraph::element::Type& elemType) -> std::shared_ptr<ngraph::opset3::Constant> {
        VPU_THROW_UNLESS(output.get_partial_shape().is_static(),
                         "DynamicToStaticShape transformation for {} of type {} expects static shape on inputs without DSR",
                         broadcast->get_friendly_name(), broadcast->get_type_info());
        return ngraph::opset3::Constant::create(elemType, {output.get_shape().size()}, output.get_shape());
    };

    const auto idxDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(broadcast->input_value(1).get_node_shared_ptr());
    const auto outShape = idxDSR ? idxDSR->input_value(1) : shapeToConstant(broadcast->input_value(1), ngraph::element::i64);
    const auto copied = target->clone_with_new_inputs(target->input_values());

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, outShape);
    outDSR->set_friendly_name(target->get_friendly_name());
    ngraph::replace_node(target, std::move(outDSR));
}

}  // namespace vpu

