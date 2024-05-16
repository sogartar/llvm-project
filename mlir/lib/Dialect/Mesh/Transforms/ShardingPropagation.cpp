//===- ShardingPropagation.cpp ------------------------------------- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include <vector>

namespace mlir {
namespace mesh {
#define GEN_PASS_DEF_SHARDINGPROPAGATION
#include "mlir/Dialect/Mesh/Transforms/Passes.h.inc"
} // namespace mesh
} // namespace mlir

#define DEBUG_TYPE "sharding-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// This method retrieves all potential sharding attributes, prioritizing
// specific shardings. For example, mustShardings = [shard0, None] and
// optionalShardings = [None, shard1], the result will be [[shard0, shard1],
// [shard0, None]]
static SmallVector<SmallVector<MeshShardingAttr>>
getOrderedPossibleShardingAttrs(ArrayRef<MeshShardingAttr> mustShardings,
                                ArrayRef<MeshShardingAttr> optionalShardings) {
  SmallVector<SmallVector<MeshShardingAttr>> allShardingAttrs;
  SmallVector<MeshShardingAttr> curShardingAttrs;

  std::function<void(size_t)> dfsCreateShardingAttrs = [&](size_t i) {
    if (i == mustShardings.size()) {
      allShardingAttrs.push_back(
          SmallVector<MeshShardingAttr>(curShardingAttrs));
      return;
    }

    if (mustShardings[i]) {
      curShardingAttrs.push_back(mustShardings[i]);
      dfsCreateShardingAttrs(i + 1);
      curShardingAttrs.pop_back();
      return;
    }

    if (optionalShardings[i]) {
      curShardingAttrs.push_back(optionalShardings[i]);
      dfsCreateShardingAttrs(i + 1);
      curShardingAttrs.pop_back();
      curShardingAttrs.push_back(nullptr);
      dfsCreateShardingAttrs(i + 1);
      curShardingAttrs.pop_back();
      return;
    }

    curShardingAttrs.push_back(nullptr);
    dfsCreateShardingAttrs(i + 1);
    curShardingAttrs.pop_back();
  };

  dfsCreateShardingAttrs(0);
  return allShardingAttrs;
}

// From all the sharding options return the one that is most compatible with
// the sharding annotations of operands and results of the operation.
// The order of preference is form highest to lowest:
// 1. No resharding is required (all existing annotations are compatible).
// 2. No resharding for operands/results that have annotation specifically
//   targeting this operation. This means
//   * operands that are the result of `mesh.shard` ops marked with
//     `annotate_for_users`.
//   * results that are annotated with `mesh.shard` ops without
//     `annotate_for_users`.
// 3. All other cases. Resharding is required for operands/results with
//   annotation targeting explicitly this operation.
// size_t preferredShardingOption(Operation *op, const SmallVector<ShardingOption>& shardingOptions) {

// }

template <bool onlyForExplicitAnnotationsForThisOp>
bool needsResharding(Operation *op, const SmallVector<MeshShardingAttr>& operandAndResultShardings) {
  size_t operandsCount = op->getOperands().size();
  auto operandShardings = llvm::make_range(operandAndResultShardings.begin(), operandAndResultShardings.begin() + operandsCount);
  auto resultShardings = llvm::make_range(operandAndResultShardings.begin() + operandsCount, operandAndResultShardings.end());

  for ( auto [operand, sharding] : llvm::zip_equal(op->getOperands(), operandShardings)) {
    ShardOp shardOp = llvm::dyn_cast<ShardOp>(operand.getDefiningOp());
    if (!shardOp) {
      continue;
    }
    if (onlyForExplicitAnnotationsForThisOp && !shardOp.getAnnotateForUsers()) {
      continue;
    }
    if (shardOp.getShardAttr() != sharding) {
      return true;
    }
  }

  for ( auto [result, sharding] : llvm::zip_equal(op->getResults(), resultShardings)) {
    for (auto user : result.getUsers()) {
        ShardOp shardOp = llvm::dyn_cast<ShardOp>(user);
        if (!shardOp) {
          continue;
        }
        if (onlyForExplicitAnnotationsForThisOp && shardOp.getAnnotateForUsers()) {
          continue;
        }
        if (shardOp.getShardAttr() != sharding) {
          return true;
        }
    }
  }

  return false;
}

// For each operation that implements the ShardingInterface, infer the sharding
// option of the operation from its operands and/or results using the
// `getShardingOption` method. If the inferred sharding option is not empty, add
// a `mesh.shard` operation for all remaining operands and results that do not
// have sharding annotations.
static LogicalResult visitOp(Operation *op, OpBuilder &builder) {
  if (op->hasTrait<OpTrait::IsTerminator>() || llvm::isa<mesh::ShardOp>(op))
    return success();

  ShardingInterface shardingOp = llvm::dyn_cast<ShardingInterface>(op);
  if (!shardingOp) {
    op->emitOpError() << "sharding interface is not implemented.";
    return failure();
  }

  // collect MeshShardingAttr from results
  SmallVector<MeshShardingAttr> allowConflictsResultShardings;
  allowConflictsResultShardings.resize(op->getNumResults());
  SmallVector<MeshShardingAttr> resultMustShardings;
  resultMustShardings.resize(op->getNumResults());
  for (OpResult result : op->getResults()) {
    FailureOr<std::pair<bool, MeshShardingAttr>> maybeShardAttr =
        getMeshShardingAttr(result);
    if (failed(maybeShardAttr))
      continue;
    if (!maybeShardAttr->first)
      resultMustShardings[result.getResultNumber()] = maybeShardAttr->second;
    else
      allowConflictsResultShardings[result.getResultNumber()] =
          maybeShardAttr->second;
  }

  // collect MeshShardingAttr from operands
  SmallVector<MeshShardingAttr> allowConflictsOperandShardings;
  allowConflictsOperandShardings.resize(op->getNumOperands());
  SmallVector<MeshShardingAttr> operandMustShardings;
  operandMustShardings.resize(op->getNumOperands());
  for (OpOperand &opOperand : op->getOpOperands()) {
    FailureOr<std::pair<bool, MeshShardingAttr>> maybeShardAttr =
        getMeshShardingAttr(opOperand);
    if (failed(maybeShardAttr))
      continue;

    if (maybeShardAttr->first)
      operandMustShardings[opOperand.getOperandNumber()] =
          maybeShardAttr->second;
    else
      allowConflictsOperandShardings[opOperand.getOperandNumber()] =
          maybeShardAttr->second;
  }

  // try to get the sharding option
  SmallVector<SmallVector<MeshShardingAttr>> possibleOperandShardingAttrs =
      getOrderedPossibleShardingAttrs(operandMustShardings,
                                      allowConflictsOperandShardings);
  SmallVector<SmallVector<MeshShardingAttr>> possibleResultShardingAttrs =
      getOrderedPossibleShardingAttrs(resultMustShardings,
                                      allowConflictsResultShardings);
  FailureOr<ShardingOption> finalShardingOption = failure();
  for (ArrayRef<MeshShardingAttr> resultShardings :
       possibleResultShardingAttrs) {
    if (succeeded(finalShardingOption))
      break;
    for (ArrayRef<MeshShardingAttr> operandShardings :
         possibleOperandShardingAttrs) {
      FailureOr<ShardingOption> shardingOption =
          shardingOp.getShardingOption(operandShardings, resultShardings);
      if (succeeded(shardingOption)) {
        finalShardingOption = shardingOption;
        break;
      }
    }
  }

  if (failed(finalShardingOption)) {
    op->emitOpError() << "fail to get sharding option.";
    return failure();
  }
  // sharding info is empty, return immediately
  if (finalShardingOption->empty)
    return success();

  if (failed(
          shardingOp.addShardingAnnotations(builder, *finalShardingOption))) {
    op->emitOpError() << "fail to set sharding annotations.";
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShardingPropagation
//===----------------------------------------------------------------------===//
struct ShardingPropagation
    : public mesh::impl::ShardingPropagationBase<ShardingPropagation> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Region &region = funcOp.getFunctionBody();
    OpBuilder builder(ctx);
    if (!region.hasOneBlock()) {
      funcOp.emitOpError() << "only one block is supported!";
      signalPassFailure();
    }
    Block &block = region.front();

    LLVM_DEBUG(
        DBGS() << "print all the ops' iterator types and indexing maps in the "
                  "block.\n";
        for (Operation &op
             : block.getOperations()) {
          if (auto shardingOp = llvm::dyn_cast<ShardingInterface>(&op))
            shardingOp.printLoopTypesAndIndexingMaps(llvm::dbgs());
        });

    // 1. propagate in reversed order
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block)))
      if (failed(visitOp(&op, builder)))
        return signalPassFailure();

    LLVM_DEBUG(DBGS() << "After reversed order propagation:\n"
                      << funcOp << "\n");
    LLVM_DEBUG(assert(succeeded(mlir::verify(funcOp))));

    // 2. propagate in original order
    for (Operation &op : llvm::make_early_inc_range(block))
      if (failed(visitOp(&op, builder)))
        return signalPassFailure();
  }
};
