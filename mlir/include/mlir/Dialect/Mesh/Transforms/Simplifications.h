//===- Simplifications.h - Mesh Simplifications -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_TRANSFORMS_SIMPLIFICATIONS_H
#define MLIR_DIALECT_MESH_TRANSFORMS_SIMPLIFICATIONS_H

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/HomomorphismSimplification.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>

namespace mlir {
namespace mesh {

template <typename AlgebraicOp>
void populateAllReduceHomomorphismSimplificationPatterns(
    RewritePatternSet &patterns, Partial reduction) {
  auto getHomomorphismOpOperand = [](Operation *op) {
    auto allReduceOp = llvm::cast<AllReduceOp>(op);
    return &allReduceOp.getInputMutable();
  };
  auto getHomomorphismOpResult = [](Operation *op) {
    auto allReduceOp = llvm::cast<AllReduceOp>(op);
    return allReduceOp->getResult(0);
  };
  auto getAlgebraicOpOperands = [](Operation *op,
                                   SmallVector<OpOperand *> &operands) {
    auto algebraicOp = llvm::cast<AlgebraicOp>(op);
    std::transform(algebraicOp->getOpOperands().begin(),
                   algebraicOp->getOpOperands().end(),
                   std::back_inserter(operands),
                   [](OpOperand &operand) { return &operand; });
  };
  auto getAlgebraicOpResult = [](Operation *op) {
    auto algebraicOp = llvm::cast<AlgebraicOp>(op);
    return algebraicOp->getResult(0);
  };
  auto isHomomorphismOp = [reduction](Operation *op, Value opResult) {
    auto allReduceOp = llvm::dyn_cast<AllReduceOp>(op);
    return allReduceOp && opResult == allReduceOp.getResult() &&
           allReduceOp.getInput().getType().getElementType() ==
               allReduceOp.getResult().getType().getElementType() &&
           allReduceOp.getReduction() == reduction;
  };
  auto isAlgebraicOp = [](Operation *op) {
    return static_cast<bool>(llvm::dyn_cast<AlgebraicOp>(op));
  };

  using ConcreteHomomorphismSimplification = HomomorphismSimplification<
      std::decay_t<decltype(getHomomorphismOpOperand)>,
      std::decay_t<decltype(getHomomorphismOpResult)>,
      std::decay_t<decltype(getAlgebraicOpOperands)>,
      std::decay_t<decltype(getAlgebraicOpResult)>,
      std::decay_t<decltype(isHomomorphismOp)>,
      std::decay_t<decltype(isAlgebraicOp)>>;
  patterns.add(std::make_unique<ConcreteHomomorphismSimplification>(
      std::move(getHomomorphismOpOperand), std::move(getHomomorphismOpResult),
      std::move(getAlgebraicOpOperands), std::move(getAlgebraicOpResult),
      std::move(isHomomorphismOp), std::move(isAlgebraicOp),
      AlgebraicOp::getOperationName(), 1, patterns.getContext()));
}

void populateSimplificationPatterns(RewritePatternSet &patterns);

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_TRANSFORMS_SIMPLIFICATIONS_H
