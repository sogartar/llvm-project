//===- RegionUtils.h - Region-related transformation utilities --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SIMPLIFY_HOMOMORPHISM_H_
#define MLIR_TRANSFORMS_SIMPLIFY_HOMOMORPHISM_H_

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <iterator>
#include <type_traits>
#include <utility>

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

// If `f` is a homomorphism with respect to the algebraic structure induced by
// function `g`, transforms `g(f(x1), f(x2) ..., f(xn))` into
// `f(g(x1, x2, ..., xn))`.
template <typename GetHomomorphismOpOperandFn,
          typename GetHomomorphismOpResultFn, typename GetAlgebraicOpOperandsFn,
          typename GetAlgebraicOpResultFn, typename IsHomomorphismOpFn,
          typename IsAlgebraicOpFn>
struct HomomorphismSimplification : RewritePattern {
  template <typename GetHomomorphismOpOperandFnArg,
            typename GetHomomorphismOpResultFnArg,
            typename GetAlgebraicOpOperandsFnArg,
            typename GetAlgebraicOpResultFnArg, typename IsHomomorphismOpFnArg,
            typename IsAlgebraicOpFnArg, typename... RewritePatternArgs>
  HomomorphismSimplification(
      GetHomomorphismOpOperandFnArg &&getHomomorphismOpOperand,
      GetHomomorphismOpResultFnArg &&getHomomorphismOpResult,
      GetAlgebraicOpOperandsFnArg &&getAlgebraicOpOperands,
      GetAlgebraicOpResultFnArg &&getAlgebraicOpResult,
      IsHomomorphismOpFnArg &&isHomomorphismOp,
      IsAlgebraicOpFnArg &&isAlgebraicOp, RewritePatternArgs &&...args)
      : RewritePattern(std::forward<RewritePatternArgs>(args)...),
        getHomomorphismOpOperand(std::forward<GetHomomorphismOpOperandFnArg>(
            getHomomorphismOpOperand)),
        getHomomorphismOpResult(std::forward<GetHomomorphismOpResultFnArg>(
            getHomomorphismOpResult)),
        getAlgebraicOpOperands(
            std::forward<GetAlgebraicOpOperandsFnArg>(getAlgebraicOpOperands)),
        getAlgebraicOpResult(
            std::forward<GetAlgebraicOpResultFnArg>(getAlgebraicOpResult)),
        isHomomorphismOp(std::forward<IsHomomorphismOpFnArg>(isHomomorphismOp)),
        isAlgebraicOp(std::forward<IsAlgebraicOpFnArg>(isAlgebraicOp)) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (failed(matchOp(op, algebraicOpOperands))) {
      return failure();
    }
    return rewriteOp(op, algebraicOpOperands, rewriter);
  }

private:
  LogicalResult matchOp(Operation *algebraicOp,
                        SmallVector<OpOperand *> &algebraicOpOperands) const {
    if (!isAlgebraicOp(algebraicOp)) {
      return failure();
    }
    algebraicOpOperands.clear();
    getAlgebraicOpOperands(algebraicOp, algebraicOpOperands);
    for (auto operand : algebraicOpOperands) {
      Operation *homomorphismOp = operand->get().getDefiningOp();
      if (!homomorphismOp ||
          !isHomomorphismOp(homomorphismOp, operand->get())) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult rewriteOp(Operation *algebraicOp,
                          const SmallVector<OpOperand *> &algebraicOpOperands,
                          PatternRewriter &rewriter) const {
    irMapping.clear();
    for (auto operand : algebraicOpOperands) {
      Operation *homomorphismOp = operand->get().getDefiningOp();
      irMapping.map(operand->get(),
                    getHomomorphismOpOperand(homomorphismOp)->get());
    }
    Operation *newAlgebraicOp = rewriter.clone(*algebraicOp, irMapping);

    irMapping.clear();
    assert(!algebraicOpOperands.empty());
    Operation *firstHomomorphismOp =
        algebraicOpOperands[0]->get().getDefiningOp();
    irMapping.map(getHomomorphismOpOperand(firstHomomorphismOp)->get(),
                  getAlgebraicOpResult(newAlgebraicOp));
    Operation *newHomomorphismOp =
        rewriter.clone(*firstHomomorphismOp, irMapping);
    rewriter.replaceAllUsesWith(getAlgebraicOpResult(algebraicOp),
                                getHomomorphismOpResult(newHomomorphismOp));
    return success();
  }

  GetHomomorphismOpOperandFn getHomomorphismOpOperand;
  GetHomomorphismOpResultFn getHomomorphismOpResult;
  GetAlgebraicOpOperandsFn getAlgebraicOpOperands;
  GetAlgebraicOpResultFn getAlgebraicOpResult;
  IsHomomorphismOpFn isHomomorphismOp;
  IsAlgebraicOpFn isAlgebraicOp;
  mutable SmallVector<OpOperand *> algebraicOpOperands;
  mutable IRMapping irMapping;
};

template <typename GetHomomorphismOpOperandFn,
          typename GetHomomorphismOpResultFn, typename GetAlgebraicOpOperandsFn,
          typename GetAlgebraicOpResultFn, typename IsHomomorphismOpFn,
          typename IsAlgebraicOpFn, typename... RewritePatternArgs>
auto makeHomomorphismSimplification(
    GetHomomorphismOpOperandFn &&getHomomorphismOpOperand,
    GetHomomorphismOpResultFn &&getHomomorphismOpResult,
    GetAlgebraicOpOperandsFn &&getAlgebraicOpOperands,
    GetAlgebraicOpResultFn &&getAlgebraicOpResult,
    IsHomomorphismOpFn &&isHomomorphismOp, IsAlgebraicOpFn &&isAlgebraicOp,
    RewritePatternArgs &&...args) {
  return HomomorphismSimplification<std::decay_t<GetHomomorphismOpOperandFn>,
                                    std::decay_t<GetHomomorphismOpResultFn>,
                                    std::decay_t<GetAlgebraicOpOperandsFn>,
                                    std::decay_t<GetAlgebraicOpResultFn>,
                                    std::decay_t<IsHomomorphismOpFn>,
                                    std::decay_t<IsAlgebraicOpFn>>(
      std::forward<GetHomomorphismOpOperandFn>(getHomomorphismOpOperand),
      std::forward<GetHomomorphismOpResultFn>(getHomomorphismOpResult),
      std::forward<GetAlgebraicOpOperandsFn>(getAlgebraicOpOperands),
      std::forward<GetAlgebraicOpResultFn>(getAlgebraicOpResult),
      std::forward<IsHomomorphismOpFn>(isHomomorphismOp),
      std::forward<IsAlgebraicOpFn>(isAlgebraicOp));
}

} // namespace mlir

#endif // MLIR_TRANSFORMS_SIMPLIFY_HOMOMORPHISM_H_
