// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <inference_engine.hpp>

#include "infer_request_wrap.hpp"

template<typename T>
static bool isImage(const T &blob) {
    auto descriptor = blob->getTensorDesc();
    if (descriptor.getLayout() == InferenceEngine::NCHW) {
        return descriptor.getDims()[1] == 3;
    } else if (descriptor.getLayout() == InferenceEngine::CHW) {
        return descriptor.getDims()[0] == 3 || descriptor.getDims()[2] == 3;
    }
    return false;
}

template<typename T>
static bool isImageInfo(const T &blob) {
    auto descriptor = blob->getTensorDesc();
    if (descriptor.getLayout() != InferenceEngine::NC) {
        return false;
    }
    auto channels = descriptor.getDims()[1];
    return (channels >= 2);
}

void fillBlobs(const std::vector<std::string>& inputFiles,
               const size_t& batchSize,
               const InferenceEngine::ConstInputsDataMap& info,
               std::vector<InferReqWrap::Ptr> requests);
