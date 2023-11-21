//===- EndomorphismSimplification.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SIMPLIFY_ENDOMORPHISM_H_
#define MLIR_TRANSFORMS_SIMPLIFY_ENDOMORPHISM_H_

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>

#include "mlir/Support/LogicalResult.h"

namespace mlir {

// If `f` is an endomorphism with respect to the algebraic structure induced by
// function `g`, transforms `g(f(x1), f(x2) ..., f(xn))` into
// `f(g(x1, x2, ..., xn))`.
// `g` is the algebraic operation and `f` is the endomorphism.
//
// Functors:
// ---------
// `GetEndomorphismOpOperandFn`: `(Operation*) -> OpOperand*`
// Returns the operand relevant to the endomorphism.
// There may be other operands that are not relevant.
//
// `GetEndomorphismOpResultFn`: `(Operation*) -> OpResult`
// Returns the result relevant to the endomorphism.
//
// `GetAlgebraicOpOperandsFn`: `(Operation*, SmallVector<OpOperand*>&) -> void`
// Populates into the vector the operands relevant to the endomorphism.
//
// `GetAlgebraicOpResultFn`: `(Operation*) -> OpResult`
//  Return the result relevant to the endomorphism.
//
// `IsEndomorphismOpFn`: `(Operation*, std::optional<Operation*>) -> bool`
// Check if the operation is an endomorphism of the required type.
// Additionally if the optional is present checks if the operations are
// compatible endomorphisms.
//
// `IsAlgebraicOpFn`: `(Operation*) -> bool`
// Check if the operation is an operation of the algebraic structure.
template <typename GetEndomorphismOpOperandFn,
          typename GetEndomorphismOpResultFn, typename GetAlgebraicOpOperandsFn,
          typename GetAlgebraicOpResultFn, typename IsEndomorphismOpFn,
          typename IsAlgebraicOpFn>
struct EndomorphismSimplification : RewritePattern {
  template <typename GetEndomorphismOpOperandFnArg,
            typename GetEndomorphismOpResultFnArg,
            typename GetAlgebraicOpOperandsFnArg,
            typename GetAlgebraicOpResultFnArg, typename IsEndomorphismOpFnArg,
            typename IsAlgebraicOpFnArg, typename... RewritePatternArgs>
  EndomorphismSimplification(
      GetEndomorphismOpOperandFnArg &&getEndomorphismOpOperand,
      GetEndomorphismOpResultFnArg &&getEndomorphismOpResult,
      GetAlgebraicOpOperandsFnArg &&getAlgebraicOpOperands,
      GetAlgebraicOpResultFnArg &&getAlgebraicOpResult,
      IsEndomorphismOpFnArg &&isEndomorphismOp,
      IsAlgebraicOpFnArg &&isAlgebraicOp, RewritePatternArgs &&...args)
      : RewritePattern(std::forward<RewritePatternArgs>(args)...),
        getEndomorphismOpOperand(std::forward<GetEndomorphismOpOperandFnArg>(
            getEndomorphismOpOperand)),
        getEndomorphismOpResult(std::forward<GetEndomorphismOpResultFnArg>(
            getEndomorphismOpResult)),
        getAlgebraicOpOperands(
            std::forward<GetAlgebraicOpOperandsFnArg>(getAlgebraicOpOperands)),
        getAlgebraicOpResult(
            std::forward<GetAlgebraicOpResultFnArg>(getAlgebraicOpResult)),
        isEndomorphismOp(std::forward<IsEndomorphismOpFnArg>(isEndomorphismOp)),
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
    if (algebraicOpOperands.empty()) {
      return failure();
    }

    Operation *firstEndomorphismOp =
        algebraicOpOperands.front()->get().getDefiningOp();
    if (!firstEndomorphismOp ||
        !isEndomorphismOp(firstEndomorphismOp, std::nullopt)) {
      return failure();
    }
    OpResult firstEndomorphismOpResult =
        getEndomorphismOpResult(firstEndomorphismOp);
    if (firstEndomorphismOpResult != algebraicOpOperands.front()->get()) {
      return failure();
    }

    for (auto operand : algebraicOpOperands) {
      Operation *endomorphismOp = operand->get().getDefiningOp();
      if (!endomorphismOp ||
          !isEndomorphismOp(endomorphismOp, firstEndomorphismOp)) {
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
      Operation *endomorphismOp = operand->get().getDefiningOp();
      irMapping.map(operand->get(),
                    getEndomorphismOpOperand(endomorphismOp)->get());
    }
    Operation *newAlgebraicOp = rewriter.clone(*algebraicOp, irMapping);

    irMapping.clear();
    assert(!algebraicOpOperands.empty());
    Operation *firstEndomorphismOp =
        algebraicOpOperands[0]->get().getDefiningOp();
    irMapping.map(getEndomorphismOpOperand(firstEndomorphismOp)->get(),
                  getAlgebraicOpResult(newAlgebraicOp));
    Operation *newEndomorphismOp =
        rewriter.clone(*firstEndomorphismOp, irMapping);
    rewriter.replaceAllUsesWith(getAlgebraicOpResult(algebraicOp),
                                getEndomorphismOpResult(newEndomorphismOp));
    return success();
  }

  GetEndomorphismOpOperandFn getEndomorphismOpOperand;
  GetEndomorphismOpResultFn getEndomorphismOpResult;
  GetAlgebraicOpOperandsFn getAlgebraicOpOperands;
  GetAlgebraicOpResultFn getAlgebraicOpResult;
  IsEndomorphismOpFn isEndomorphismOp;
  IsAlgebraicOpFn isAlgebraicOp;
  mutable SmallVector<OpOperand *> algebraicOpOperands;
  mutable IRMapping irMapping;
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_SIMPLIFY_ENDOMORPHISM_H_
