// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <map>
#include <optional>
#include <stack>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include "common/assert.h"
#include "common/common_types.h"
#include "video_core/shader/node_helper.h"
#include "video_core/shader/shader_ir.h"

namespace VideoCommon::Shader {

namespace {

template <OperationCode code>
const OperationNode* GetOperation(const Node& node) {
    const auto operation = std::get_if<OperationNode>(&*node);
    if (operation && operation->GetCode() == code) {
        return operation;
    }
    return {};
}

u32 GetFirstImmediate(const OperationNode& op) {
    return std::get<ImmediateNode>(*op[0]).GetValue();
}

bool IsUnconditionalBranch(const Node& node) {
    const auto op = std::get_if<OperationNode>(&*node);
    if (!op) {
        return false;
    }
    return op->GetCode() == OperationCode::Branch || op->GetCode() == OperationCode::PopFlowStack ||
           op->GetCode() == OperationCode::Exit;
}

std::stack<u32>& GetStack(FlowStack& stack, const OperationNode& pop) {
    // TODO(Rodrigo): Implement multiple stacks
    return stack.ssy;
}

} // Anonymous namespace

void ShaderIR::FlowStackRemover() {
    const auto first_label = basic_blocks.cbegin()->first;
    std::map<u32, NodeBlock> new_basic_blocks;
    std::set<u32> visited_blocks;
    if (FlowStackRemoverBasicBlock(new_basic_blocks, visited_blocks, first_label)) {
        basic_blocks = std::move(new_basic_blocks);
    }
}

bool ShaderIR::FlowStackRemoverBasicBlock(std::map<u32, NodeBlock>& new_basic_blocks,
                                          std::set<u32>& visited_blocks, u32 label,
                                          FlowStack stack) {
    if (visited_blocks.find(label) != visited_blocks.end()) {
        // This block has been previously visited
        return true;
    }
    visited_blocks.emplace(label);

    const NodeBlock& old_block = basic_blocks.at(label);
    auto result = FlowStackRemoverNodes(new_basic_blocks, visited_blocks, old_block, stack);
    if (!result) {
        return false;
    }

    new_basic_blocks.emplace(label, std::move(result->first));
    if (IsUnconditionalBranch(old_block.back())) {
        return true;
    }
    const auto it = basic_blocks.upper_bound(label);
    ASSERT_OR_EXECUTE(it != basic_blocks.end(), return false;);
    return FlowStackRemoverBasicBlock(new_basic_blocks, visited_blocks, it->first, result->second);
}

std::optional<std::pair<NodeBlock, FlowStack>> ShaderIR::FlowStackRemoverNodes(
    std::map<u32, NodeBlock>& new_basic_blocks, std::set<u32>& visited_blocks,
    const NodeBlock& old_block, FlowStack stack) {
    NodeBlock new_code;
    for (const auto node : old_block) {
        if (const auto branch = GetOperation<OperationCode::Branch>(node)) {
            if (!FlowStackRemoverBasicBlock(new_basic_blocks, visited_blocks,
                                            GetFirstImmediate(*branch), stack)) {
                return {};
            }
            new_code.push_back(node);

        } else if (const auto push = GetOperation<OperationCode::PushFlowStack>(node)) {
            const auto target = GetFirstImmediate(*push);
            stack.ssy.push(target);
            new_code.push_back(Comment(fmt::format("Push 0x{:x}, optimized out", target)));

        } else if (const auto pop = GetOperation<OperationCode::PopFlowStack>(node)) {
            auto& current_stack = GetStack(stack, *pop);
            const auto target = current_stack.top();
            current_stack.pop();
            if (!FlowStackRemoverBasicBlock(new_basic_blocks, visited_blocks, target, stack)) {
                return {};
            }
            new_code.push_back(Operation(OperationCode::Branch, Immediate(target)));

        } else if (const auto conditional = std::get_if<ConditionalNode>(&*node)) {
            auto result = FlowStackRemoverNodes(new_basic_blocks, visited_blocks,
                                                conditional->GetCode(), stack);
            if (!result) {
                return {};
            }
            new_code.push_back(Conditional(conditional->GetCondition(), std::move(result->first)));

        } else {
            new_code.push_back(node);
        }
    }
    return std::make_pair(new_code, stack);
}

} // namespace VideoCommon::Shader
