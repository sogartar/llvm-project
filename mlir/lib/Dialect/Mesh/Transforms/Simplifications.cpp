//===- Patterns.cpp - Mesh Patterns -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Transforms/Simplifications.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace mesh {

void populateSimplificationPatterns(RewritePatternSet &patterns) {
  populateAllReduceHomomorphismSimplificationPatterns<arith::AddFOp>(
      patterns, Partial::Sum);
}

} // namespace mesh
} // namespace mlir
