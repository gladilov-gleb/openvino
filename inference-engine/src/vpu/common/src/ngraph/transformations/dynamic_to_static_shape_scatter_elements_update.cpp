// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_scatter_elements_update.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset5.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticShapeScatterElementsUpdate(std::shared_ptr<ngraph::Node> node) {
    const auto scatter = ngraph::as_type_ptr<ngraph::opset5::ScatterElementsUpdate>(node);
    VPU_THROW_UNLESS(scatter, "dynamicToStaticShapeScatterElementsUpdate transformation is not applicable for {}, it should be {} instead",
                     node, ngraph::opset5::ScatterElementsUpdate::type_info);

    const auto dataDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(scatter->input_value(0).get_node_shared_ptr());

    VPU_THROW_UNLESS(dataDSR, "dynamicToStaticShapeScatterElementsUpdate transformation for {} of type {} expects DSR as first input",
                     scatter->get_friendly_name(), scatter->get_type_info());

    const auto copied = node->clone_with_new_inputs(node->input_values());

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, dataDSR->input_value(1));
    outDSR->set_friendly_name(scatter->get_friendly_name());
    ngraph::replace_node(std::move(scatter), std::move(outDSR));
}

}  // namespace vpu
