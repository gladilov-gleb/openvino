// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset5.hpp>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeLoop : public ngraph::opset5::Loop {
public:
    static constexpr NodeTypeInfo type_info{"StaticShapeLoop", 5};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    explicit StaticShapeLoop(const Loop& loop);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor&) override;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
