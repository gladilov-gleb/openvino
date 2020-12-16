// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/optional.hpp>
#include <ngraph/ngraph.hpp>
#include "ngraph/opsets/opset5.hpp"
#include "vpu/ngraph/transformations/extract_batch.hpp"

#include <queue>

namespace vpu {

NGRAPH_RTTI_DEFINITION(vpu::ExtractBatch, "ExtractBatch", 0);

ExtractBatch::ExtractBatch(std::unordered_set<ngraph::Node::type_info_t> targets) : targets(std::move(targets)) {}

namespace {

enum class SplitMode {
    Slice,
    Unchanged
};

using SplitConfiguration = vpu::Optional<std::pair<std::vector<SplitMode>, std::vector<SplitMode>>>;

SplitConfiguration matMul(const ngraph::Node* node) {
    VPU_THROW_UNLESS(node->get_input_size() == 2,  "Expecting MatMul like operation to have {} inputs, got {}", 2, node->get_input_size());
    VPU_THROW_UNLESS(node->get_output_size() == 1, "Expecting MatMul like operation to have {} outputs, got {}", 1, node->get_output_size());
    VPU_THROW_UNLESS(ngraph::as_type_ptr<ngraph::opset5::Constant>(node->input_value(1).get_node_shared_ptr()),
                     "Expecting MatMul like operation to have second input as {}, got {}", ngraph::opset5::Constant::type_info, node->input_value(1));
    const auto& dataPartialShape = node->get_input_partial_shape(0);
    VPU_THROW_UNLESS(dataPartialShape.rank().is_static(), "Dynamic rank for MatMul data input is unsupported: {}", node);

    const auto& outPartialShape = node->get_output_partial_shape(0);
    VPU_THROW_UNLESS(outPartialShape.rank().is_static(), "Dynamic rank for MatMul output is unsupported: {}", node);
    VPU_THROW_UNLESS(dataPartialShape.rank() == outPartialShape.rank(),
        "Expecting MatMul like operation to have the same rank for data and output, got data rank = {}, output rank = {}",
        dataPartialShape.rank(), outPartialShape.rank());

    const auto& dataBatch = dataPartialShape[0];
    const auto& outBatch = outPartialShape[0];
    VPU_THROW_UNLESS(dataBatch == outBatch,
        "Expecting MatMul like operation to have the same batch on data and output, got data batch = {} and output batch = {}", dataBatch, outBatch);

    // static batch is currently handled by other means
    if (dataBatch.is_static()) {
        return {};
    }

    return std::make_pair(
        std::vector<SplitMode>{SplitMode::Slice, SplitMode::Unchanged},
        std::vector<SplitMode>{SplitMode::Slice});
}

template<class T>
SplitConfiguration add(const ngraph::Node* node) {
    auto eltwise = ngraph::as_type<const T>(node);
    VPU_THROW_UNLESS(eltwise->get_input_size() == 2,  "Expecting Add operation to have {} inputs, got {}", 2, eltwise->get_input_size());
    VPU_THROW_UNLESS(eltwise->get_output_size() == 1, "Expecting Add operation to have {} outputs, got {}", 1, eltwise->get_output_size());

    const auto& lhs = eltwise->input_value(0);
    const auto& rhs = eltwise->input_value(1);
    const auto& out = eltwise->output(0);

    const auto& lhsPartialShape = lhs.get_partial_shape();
    const auto& rhsPartialShape = rhs.get_partial_shape();
    const auto& outPartialShape = out.get_partial_shape();

    const auto& broadcastSpec = eltwise->get_autob();
    auto inputPartialShape = lhsPartialShape;
    if (broadcastSpec == ngraph::op::AutoBroadcastSpec::NONE) {
        ngraph::PartialShape::merge_into(inputPartialShape, rhsPartialShape);
    } else {
        ngraph::PartialShape::broadcast_merge_into(inputPartialShape, rhsPartialShape, broadcastSpec);
    }

    const auto& inputRank = inputPartialShape.rank();
    const auto& lhsRank   = lhsPartialShape.rank();
    const auto& rhsRank   = rhsPartialShape.rank();
    const auto& outRank   = outPartialShape.rank();

    VPU_THROW_UNLESS(inputRank == outRank && inputRank.is_static(),
                     "Expecting Add operation to have the same static rank for inputs and output, got merged inputs rank = {}, output rank = {}",
                     inputRank, outRank);

    const auto& inputRankLength = inputRank.get_length();
    const auto& lhsRankLength   = lhsRank.get_length();
    const auto& rhsRankLength   = rhsRank.get_length();
    const auto& outRankLength   = outRank.get_length();

    const auto& inputsBatch = inputRankLength > 0 ? inputPartialShape[0] : 0;
    const auto& outBatch = outRankLength > 0 ? outPartialShape[0] : 0;
    VPU_THROW_UNLESS(inputsBatch == outBatch,
        "Expecting Add operation to have the same batch on both inputs and output, got input batch = {}, output batch = {}",
        inputsBatch, outBatch);


    if (inputsBatch.is_static() && inputsBatch.get_length() == 1) {
        return {};
    }

    const auto lhsSplitMode = lhsRankLength < inputRankLength || lhsPartialShape[0] != inputPartialShape[0] ? SplitMode::Unchanged : SplitMode::Slice;
    const auto rhsSplitMode = rhsRankLength < inputRankLength || rhsPartialShape[0] != inputPartialShape[0] ? SplitMode::Unchanged : SplitMode::Slice;

    return std::make_pair(
        std::vector<SplitMode>{lhsSplitMode, rhsSplitMode},
        std::vector<SplitMode>{SplitMode::Slice});
}

SplitConfiguration relu(const ngraph::Node* node) {
    VPU_THROW_UNLESS(node->get_input_size() == 1,  "Expecting ReLU operation to have {} inputs, got {}", 1, node->get_input_size());
    VPU_THROW_UNLESS(node->get_output_size() == 1, "Expecting Add operation to have {} outputs, got {}", 1, node->get_output_size());

    const auto& inp = node->input_value(0);
    const auto& out = node->output(0);

    const auto& inpPartialShape = inp.get_partial_shape();
    const auto& outPartialShape = out.get_partial_shape();

    const auto& inpRank = inpPartialShape.rank();
    const auto& outRank = outPartialShape.rank();

    VPU_THROW_UNLESS(inpRank == outRank, "Expecting ReLU operation to have the same static rank for input and output, got input rank = {}, output rank = {}",
                     inpRank, outRank);

    const auto& inpRankLength = inpRank.get_length();
    const auto& outRankLength = outRank.get_length();

    const auto& inpBatch = inpRankLength > 0 ? inpPartialShape[0] : 0;
    const auto& outBatch = outRankLength > 0 ? outPartialShape[0] : 0;
    VPU_THROW_UNLESS(inpBatch == outBatch, "Expecting ReLU operation to have the same batch on input and output, got input batch = {}, output batch = {}",
                     inpBatch, outBatch);

    if (inpBatch.is_static() && inpBatch.get_length() == 1) {
        return {};
    }

    return std::make_pair(
        std::vector<SplitMode>{SplitMode::Slice},
        std::vector<SplitMode>{SplitMode::Slice});
}

bool isConstant(const ngraph::Node* node) {
    return node->get_type_info() == ngraph::opset5::Constant::type_info;
}

bool isParameter(const ngraph::Node* node) {
    return node->get_type_info() == ngraph::opset5::Parameter::type_info;
}

bool isResult(const ngraph::Node* node) {
    return node->get_type_info() == ngraph::opset5::Result::type_info;
}

using functor = std::function<SplitConfiguration(const ngraph::Node*)>;
const std::unordered_map<ngraph::Node::type_info_t, functor> functors = {
    {ngraph::opset5::MatMul::type_info, matMul},
    {ngraph::opset5::Convolution::type_info, matMul},
    {ngraph::opset5::GroupConvolution::type_info, matMul},

    {ngraph::opset5::Add::type_info, add<ngraph::opset5::Add>},
    {ngraph::opset5::Multiply::type_info, add<ngraph::opset5::Multiply>},
    {ngraph::opset5::Minimum::type_info, add<ngraph::opset5::Minimum>},
    {ngraph::opset5::Maximum::type_info, add<ngraph::opset5::Maximum>},
    {ngraph::opset5::Relu::type_info, relu},
};

bool check(const ngraph::Node* node) {
    return functors.count(node->get_type_info()) && functors.at(node->get_type_info())(node).hasValue();
}

template<class Functor>
std::unordered_set<ngraph::Node*> getLevels(ngraph::Node* from, ngraph::Node* to, Functor&& getNext) {
    std::unordered_set<ngraph::Node*> levels;
    std::stack<ngraph::Node*> stack{{from}};

    // TODO: what if are going to visit some node twice??

    while (!stack.empty()) {
        const auto current = stack.top();
        stack.pop();

        levels.emplace(current);

        if (current == to) {
            continue;
        }

        for (const auto& node : getNext(current)) {
            stack.push(node);
        }
    }
    levels.erase(from);
    return levels;
}

template<class Functor>
std::unordered_set<ngraph::Node*> getEnds(ngraph::Node* source, const std::unordered_set<ngraph::Node*>& blackList, std::unordered_set<ngraph::Node*>& all,
                                          Functor&& getNext) {
    const auto isOk = [&blackList](ngraph::Node* node) { return check(node) && !blackList.count(node); };
    IE_ASSERT(isOk(source));
    std::unordered_set<ngraph::Node*> ends;

    {
        const auto& nextNodes = getNext(source);
        const auto exit = nextNodes.empty() || std::any_of(nextNodes.cbegin(), nextNodes.cend(), [isOk](ngraph::Node* node) { return !isOk(node); });
        if (exit) {
            return {source};
        }
    }

    for (const auto& nextNode : getNext(source)) {
        std::stack<ngraph::Node*> stack{{nextNode}};
        while (!stack.empty()) {
            const auto current = stack.top();
            stack.pop();

            if (all.count(current)) {
                continue;
            }

            all.emplace(current);

            const auto& nextNodes = getNext(current);
            const auto exit = nextNodes.empty() || std::any_of(nextNodes.cbegin(), nextNodes.cend(), [isOk](ngraph::Node* node) { return !isOk(node); });

            if (exit) {
                ends.emplace(current);
                continue;
            }

            for (const auto& next : nextNodes) {
                stack.push(next);
            }
        }
    }

    all.erase(source);
    return ends;
}

template<class NextForward, class NextBackward>
ngraph::Node* getLast(ngraph::Node* source, std::unordered_set<ngraph::Node*>& ends,
                      std::unordered_set<ngraph::Node*>& allForward, const std::unordered_set<ngraph::Node*>& allBackward,
                      NextForward&& getNextForward, NextBackward&& getNextBackward) {
    std::unordered_map<ngraph::Node*, std::size_t> allDepths{{source, 0}}, endsDepths;
    auto const less = [&allDepths](ngraph::Node* lhs, ngraph::Node* rhs) {
        VPU_THROW_UNLESS(allDepths.count(lhs), "There is no {} in all depth", lhs);
        VPU_THROW_UNLESS(allDepths.count(rhs), "There is no {} in all depth", rhs);
        return allDepths.at(lhs) < allDepths.at(rhs);
    };

    auto const equal = [&allDepths](ngraph::Node* lhs, ngraph::Node* rhs) {
        VPU_THROW_UNLESS(allDepths.count(lhs), "There is no {} in all depth", lhs);
        VPU_THROW_UNLESS(allDepths.count(rhs), "There is no {} in all depth", rhs);
        return allDepths.at(lhs) == allDepths.at(rhs);
    };

    if (ends.size() == 1 && ends.count(source)) {
        allForward.clear();
        return source;
    }

    std::queue<ngraph::Node*> queue;
    for (const auto& nextNode : getNextForward(source)) {
        queue.push(nextNode);
    }

    while (!queue.empty()) {
        const auto current = queue.front();
        queue.pop();

        if (allDepths.count(current)) {
            continue;
        }

        auto entries = getNextBackward(current);
        for (auto it = entries.begin(); it != entries.end();) {
            if (allBackward.count(*it)) {
                it = entries.erase(it);
            } else {
                it++;
            }
        }
        if (std::any_of(entries.cbegin(), entries.cend(), [&allDepths](ngraph::Node* entry) { return !allDepths.count(entry); })) {
            continue;
        }

        const auto depth = allDepths.at(*std::max_element(entries.cbegin(), entries.cend(), less)) + 1;
        allDepths[current] = depth;

        if (ends.count(current)) {
            endsDepths[current] = depth;
            continue;
        }

        for (const auto& nextNode : getNextForward(current)) {
            queue.push(nextNode);
        }
    }

    VPU_THROW_UNLESS(endsDepths.size() == ends.size(), "endsDepths and ends have different sizes: {} vs {}", endsDepths.size(), ends.size());

    const auto minDepthArg = std::min_element(ends.cbegin(), ends.cend(), less);
    while (!std::all_of(ends.cbegin(), ends.cend(), [equal, minDepthArg](ngraph::Node* end) { return equal(end, *minDepthArg); })) {
        std::unordered_map<ngraph::Node*, ngraph::Node*> updates;
        for (const auto& end : ends) {
            auto current = end;
            while (!equal(current, *minDepthArg)) {
                const auto& nextNodes = getNextBackward(current);
                current = *std::max_element(nextNodes.cbegin(), nextNodes.cend(), less);
            }

            updates[end] = current;
        }

        for (const auto& update : updates) {
            ends.erase(update.first);
            ends.emplace(update.second);
        }
    }

    auto result = ends;
    while (result.size() != 1) {
        std::unordered_map<ngraph::Node*, ngraph::Node*> updates;
        for (const auto& end : result) {
            const auto& nextNodes = getNextBackward(end);
            const auto next = *std::max_element(nextNodes.cbegin(), nextNodes.cend(), less);

            updates[end] = next;
        }

        for (const auto& update : updates) {
            result.erase(update.first);
            result.emplace(update.second);
        }
    }

    allForward.clear();
    allForward = getLevels(source, *result.begin(), getNextForward);
    return *result.begin();
}

template<class Functor>
std::shared_ptr<ngraph::opset5::Loop> makeLoop(ngraph::Node* root, ngraph::Node* leaf, Functor&& getNextTop) {
    ngraph::ParameterVector parameters;
    ngraph::ResultVector results;
    std::unordered_map<std::shared_ptr<ngraph::opset5::Parameter>, ngraph::Output<ngraph::Node>> slicedInputs, invariantInputs;
    std::set<ngraph::Output<ngraph::Node>> concatenatedResults;
    std::set<ngraph::Output<ngraph::Node>> iterValueResults;

    std::map<ngraph::Output<ngraph::Node>, ngraph::Output<ngraph::Node>> nodes;
    const auto getInput = [&nodes, &parameters, &slicedInputs, &invariantInputs](const ngraph::Output<ngraph::Node>& currentInput) {
        if (nodes.count(currentInput)) {
            return nodes.at(currentInput);
        } else {
            const auto& currentInputNode = currentInput.get_node();
            VPU_THROW_UNLESS(isConstant(currentInputNode) || isParameter(currentInputNode),
                "Encountered intermediate node {} which is not cloned yet", currentInputNode);

            // assume if constant has several consumers all of them requires either Slice or Unchanged
            const auto& targetInputs = currentInput.get_target_inputs();
            const auto& targetInput = targetInputs.begin();
            const auto& node = targetInput->get_node();
            const auto& index = targetInput->get_index();
            const auto splitInputConfiguration = functors.at(node->get_type_info())(node).get().first;

            // TODO: Assert all of them have the same value
            if (splitInputConfiguration[index] == SplitMode::Slice) {
                auto partialShape = currentInput.get_partial_shape();
                partialShape[0] = 1;
                auto parameter = std::make_shared<ngraph::opset5::Parameter>(currentInput.get_element_type(), partialShape);
                parameters.emplace_back(parameter);
                slicedInputs[parameter] = currentInput;

                nodes[currentInput] = parameter;
                return static_cast<ngraph::Output<ngraph::Node>>(parameter);
            } else {
                auto argument = currentInput;
                if (isParameter(currentInputNode)) {
                    auto parameter = std::make_shared<ngraph::opset5::Parameter>(currentInput.get_element_type(), currentInput.get_partial_shape());
                    parameters.emplace_back(parameter);
                    invariantInputs[parameter] = currentInput;

                    argument = parameter;
                }

                nodes[currentInput] = argument;
                return argument;
            }
        }
    };

    const auto splitInputConfiguration = functors.at(root->get_type_info())(root).get().first;
    for (std::size_t i = 0; i < root->get_input_size(); ++i) {
        const auto& input = root->input_value(i);
        ngraph::Output<ngraph::Node> argument;
        if (splitInputConfiguration[i] == SplitMode::Slice) {
            auto partialShape = input.get_partial_shape();
            partialShape[0] = 1;

            auto parameter = std::make_shared<ngraph::opset5::Parameter>(input.get_element_type(), partialShape);
            parameters.emplace_back(parameter);
            slicedInputs[parameter] = input;

            argument = parameter;
        } else if (!isConstant(input.get_node())) {
            auto parameter = std::make_shared<ngraph::opset5::Parameter>(input.get_element_type(), input.get_partial_shape());
            parameters.emplace_back(parameter);
            invariantInputs[parameter] = input;

            argument = parameter;
        } else {
            argument = input;
        }

        nodes[input] = argument;
    }

    std::queue<ngraph::Node*> queue{{root}};
    const auto rootSize = getNextTop(root).size();
    VPU_THROW_UNLESS(rootSize > 0, "Encountered {} as loop root, but root without inputs is not supported for loop", root);
    std::unordered_map<ngraph::Node*, std::size_t> visited{{root, rootSize - 1}};
    while (!queue.empty()) {
        const auto current = queue.front();
        queue.pop();

        visited[current]++;
        const auto size = getNextTop(current).size();
        VPU_THROW_UNLESS(visited[current] <= size, "Encountered loop at {}", current);

        if (visited[current] < size) {
            VPU_THROW_UNLESS(!queue.empty(), "Node {} should be visited only after all predecessors, but it is not available through all of them", current);
            continue;
        }

        if (current == leaf) {
            continue;
        }

        std::vector<ngraph::Output<ngraph::Node>> newInputs;
        newInputs.reserve(current->get_input_size());
        const auto& currentInputs = current->input_values();
        std::transform(currentInputs.cbegin(), currentInputs.cend(), std::back_inserter(newInputs), getInput);

        auto bodyNode = current->copy_with_new_inputs(newInputs);
        bodyNode->set_friendly_name(current->get_friendly_name());
        VPU_THROW_UNLESS(bodyNode->get_output_size() == current->get_output_size(),
            "Encountered mismatch in output count between original node {} and copy without batch {}", current, bodyNode);

        for (std::size_t i = 0; i < current->get_output_size(); ++i) {
            const auto& currentOutput = current->output(i);
            const auto& bodyOutput = bodyNode->output(i);
            const auto& currentOutputNode = currentOutput.get_node();
            if (isResult(currentOutputNode)) {
                const auto splitOutputConfiguration = functors.at(current->get_type_info())(current).get().second;
                if (splitOutputConfiguration[i] == SplitMode::Slice) {
                    concatenatedResults.emplace(bodyOutput);
                } else {
                    iterValueResults.emplace(bodyOutput);
                }
                results.emplace_back(std::make_shared<ngraph::opset5::Result>(bodyOutput));
            } else {
                nodes[currentOutput] = bodyOutput;
                for (const auto& consumer : current->get_output_target_inputs(i)) {
                    queue.push(consumer.get_node());
                }
            }
        }
    }

    std::vector<ngraph::Output<ngraph::Node>> newInputs;
    newInputs.reserve(leaf->get_input_size());
    const auto& currentInputs = leaf->input_values();
    std::transform(currentInputs.cbegin(), currentInputs.cend(), std::back_inserter(newInputs), getInput);

    const auto& bodyNode = leaf->copy_with_new_inputs(newInputs);
    VPU_THROW_UNLESS(bodyNode->get_output_size() == leaf->get_output_size(),
        "Encountered mismatch in output count between original node {} and copy without batch {}", leaf, bodyNode);
    bodyNode->set_friendly_name(leaf->get_friendly_name());

    const auto splitOutputConfiguration = functors.at(leaf->get_type_info())(leaf).get().second;
    for (std::size_t i = 0; i < bodyNode->get_output_size(); ++i) {
        const auto& output = bodyNode->output(i);
        auto result = std::make_shared<ngraph::opset5::Result>(output);
        if (splitOutputConfiguration[i] == SplitMode::Slice) {
            concatenatedResults.emplace(output);
        } else {
            iterValueResults.emplace(output);
        }
        results.emplace_back(result);
    }

    VPU_THROW_UNLESS(!slicedInputs.empty(), "Failed to find sliced inputs for loop in extract batch");
    const auto& slicedInput = slicedInputs.begin()->second;
    const auto shapeOf = std::make_shared<ngraph::opset5::ShapeOf>(slicedInput);
    const auto constant = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{}, 0);
    // TODO: check all other sliced inputs have the same batch?
    const auto batchSize = std::make_shared<ngraph::opset5::Gather>(shapeOf, constant, constant);

    const auto constantTrue = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{}, true);
    auto loop = std::make_shared<ngraph::opset5::Loop>(
        batchSize,
        constantTrue);

    results.emplace_back(std::make_shared<ngraph::opset5::Result>(constantTrue));
    auto body = std::make_shared<ngraph::Function>(results, parameters, "body");
    loop->set_function(body);
    for (const auto& entry : slicedInputs) {
        loop->set_sliced_input(entry.first, entry.second, 0, 1, 1, -1, 0);
    }

    for (const auto& entry : invariantInputs) {
        loop->set_invariant_input(entry.first, entry.second);
    }

    for (const auto& entry : iterValueResults) {
        loop->get_iter_value(entry, -1);
    }

    for (const auto& entry : concatenatedResults) {
        loop->get_concatenated_slices(entry, 0, 1, 1, -1, 0);
    }

    loop->set_special_body_ports({-1, static_cast<std::int64_t>(results.size()) - 1});
    loop->validate_and_infer_types();
    return loop;
}

}  // namespace

bool ExtractBatch::run_on_function(std::shared_ptr<ngraph::Function> functionPointer) {
    auto& function = *functionPointer;
    bool changed = false;

    std::unordered_set<ngraph::Node*> sources;
    for (const auto& operation : function.get_ordered_ops()) {
        if (targets.count(operation->get_type_info())) {
            sources.emplace(operation.get());
        }
    }

    auto getNextTop = [](const ngraph::Node* node) {
        std::unordered_set<ngraph::Node*> nextNodes;
        for (std::size_t i = 0; i < node->get_input_size(); ++i) {
            const auto next = node->get_input_source_output(i).get_node();
            if (isConstant(next) || isParameter(next)) {
                continue;
            }
            nextNodes.emplace(next);
        }
        return nextNodes;
    };

    auto getNextBottom = [](const ngraph::Node* node) {
        std::unordered_set<ngraph::Node*> nextNodes;
        for (std::size_t i = 0; i < node->get_output_size(); ++i) {
            const auto consumers = node->get_output_target_inputs(i);
            for (const auto consumer : consumers) {
                const auto next = consumer.get_node();
                if (isResult(next)) {
                    continue;
                }
                nextNodes.insert(next);
            }
        }
        return nextNodes;
    };

    for (auto currentSource = sources.begin(); currentSource != sources.end(); currentSource = sources.erase(currentSource)) {
        const auto& source = *currentSource;

        VPU_THROW_UNLESS(functors.count(source->get_type_info()),
            "{} was requested as target operation type for batch extraction, but functor for this type is not provided.", source->get_type_info());

        if (!functors.at(source->get_type_info())(source).hasValue()) {
            continue;
        }

        std::unordered_set<ngraph::Node*> withExternalConnectionTop, tops;
        std::unordered_set<ngraph::Node*> withExternalConnectionBottom, bottoms;

        auto topEnds    = getEnds(source, withExternalConnectionTop, tops, getNextTop);
        auto bottomEnds = getEnds(source, withExternalConnectionBottom, bottoms, getNextBottom);

        auto const hasExternalConnection = [&tops, &bottoms, source](const std::unordered_set<ngraph::Node*>& nextNodes) {
            return std::any_of(nextNodes.cbegin(), nextNodes.cend(), [&tops, &bottoms, source](ngraph::Node* next) {
                return !tops.count(next) && !bottoms.count(next) && next != source;
            });
        };

        bool hasExternalConnectionsFromTop = false;
        for (const auto& node : tops) {
            if (hasExternalConnection(getNextBottom(node))) {
                hasExternalConnectionsFromTop = true;
                withExternalConnectionTop.insert(node);
            }
        }

        if (hasExternalConnectionsFromTop) {
            tops.clear();
            topEnds = getEnds(source, withExternalConnectionTop, tops, getNextTop);
        }

        bool hasExternalConnectionsFromBottom = false;
        for (const auto& node : bottoms) {
            if (hasExternalConnection(getNextTop(node))) {
                hasExternalConnectionsFromBottom = true;
                withExternalConnectionBottom.insert(node);
            }
        }

        if (hasExternalConnectionsFromBottom) {
            bottoms.clear();
            bottomEnds = getEnds(source, withExternalConnectionBottom, bottoms, getNextBottom);
        }

        ngraph::Node* top = nullptr;
        ngraph::Node* bottom = nullptr;
        do {
            top = getLast(source, topEnds, tops, bottoms, getNextTop, getNextBottom);
            bottom = getLast(source, bottomEnds, bottoms, tops, getNextBottom, getNextTop);

            hasExternalConnectionsFromTop = false;
            for (const auto& node : tops) {
                if (hasExternalConnection(getNextBottom(node))) {
                    hasExternalConnectionsFromTop = true;
                    withExternalConnectionTop.insert(node);
                }
            }

            if (hasExternalConnectionsFromTop) {
                tops.clear();
                topEnds = getEnds(source, withExternalConnectionTop, tops, getNextTop);
            }

            hasExternalConnectionsFromBottom = false;
            for (const auto& node : bottoms) {
                if (hasExternalConnection(getNextTop(node))) {
                    hasExternalConnectionsFromBottom = true;
                    withExternalConnectionBottom.insert(node);
                }
            }

            if (hasExternalConnectionsFromBottom) {
                bottoms.clear();
                bottomEnds = getEnds(source, withExternalConnectionBottom, bottoms, getNextBottom);
            }
        } while (hasExternalConnectionsFromTop || hasExternalConnectionsFromBottom);

        for (const auto& node : tops) {
            if (sources.count(node)) {
                sources.erase(node);
            }
        }

        for (const auto& node : bottoms) {
            if (sources.count(node)) {
                sources.erase(node);
            }
        }

        auto loop = makeLoop(top, bottom, getNextTop);

        // TODO: do we need to check if loop is going to be output operation and handle its name accordingly?

        auto bottomNode = bottom->shared_from_this();
        ngraph::copy_runtime_info(bottomNode, loop);
        ngraph::replace_node(bottomNode, loop);
        function.validate_nodes_and_infer_types();
        changed = true;
    }

    return changed;
}

}  // namespace vpu
