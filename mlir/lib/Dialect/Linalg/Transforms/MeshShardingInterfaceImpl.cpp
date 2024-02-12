//===- MeshShardingInterfaceImpl.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/MeshShardingInterfaceImpl.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iterator>
#include <optional>

namespace mlir::linalg {

static mesh::ReductionKind reductionKind(Operation *op) {
  return llvm::TypeSwitch<Operation *, mesh::ReductionKind>(op)
      // Floating-point operations.
      .Case([](arith::AddFOp op) { return mesh::ReductionKind::Sum; })
      .Case([](arith::MulFOp op) { return mesh::ReductionKind::Product; })
      .Case([](arith::MaximumFOp op) { return mesh::ReductionKind::Max; })
      .Case([](arith::MinimumFOp op) { return mesh::ReductionKind::Min; })
      // Integer operations.
      .Case([](arith::AddIOp op) { return mesh::ReductionKind::Sum; })
      .Case([](arith::OrIOp op) { return mesh::ReductionKind::BitwiseOr; })
      .Case([](arith::XOrIOp op) { return mesh::ReductionKind::BitwiseXor; })
      .Case([](arith::AndIOp op) { return mesh::ReductionKind::Sum; })
      .Case([](arith::MaxUIOp op) { return mesh::ReductionKind::Max; })
      .Case([](arith::MinUIOp op) { return mesh::ReductionKind::Min; })
      .Case([](arith::MaxSIOp op) { return mesh::ReductionKind::Max; })
      .Case([](arith::MinSIOp op) { return mesh::ReductionKind::Min; })
      .Case([](arith::MulIOp op) { return mesh::ReductionKind::Product; })
      .Default([](Operation *op) { return mesh::ReductionKind::Generic; });
}

static mesh::ReductionKind reductionKindOfLinalgOp(LinalgOp op) {
  SmallVector<Operation *, 4> combinerOps;
  Value reducedValue = matchReduction(op.getRegionOutputArgs(), 0, combinerOps);
  if (!reducedValue || combinerOps.size() != 1) {
    return mesh::ReductionKind::Generic;
  }
  Operation *reductionOp = combinerOps[0];
  return reductionKind(reductionOp);
}

namespace {

// ShardingInterface for ops that implement LinalgStructuredInterface.
// The supported ops are only those where the indexing maps are projected
// permutations.
template <typename Op>
struct StructuredOpShardingInterface
    : mesh::ShardingInterface::ExternalModel<StructuredOpShardingInterface<Op>,
                                             Op> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    LinalgOp linalgOp = llvm::cast<LinalgOp>(op);
    return linalgOp.getIteratorTypesArray();
  }

  SmallVector<mesh::ReductionKind>
  getReductionLoopIteratorKinds(Operation *op) const {
    LinalgOp linalgOp = llvm::cast<LinalgOp>(op);
    SmallVector<utils::IteratorType> iterTypes =
        linalgOp.getIteratorTypesArray();
    size_t reductionIterCount =
        llvm::count_if(iterTypes, [](utils::IteratorType i) {
          return i == utils::IteratorType::reduction;
        });
    if (reductionIterCount == 0) {
      return {};
    }

    return SmallVector<mesh::ReductionKind>(reductionIterCount,
                                            reductionKindOfLinalgOp(linalgOp));
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    LinalgOp linalgOp = llvm::cast<LinalgOp>(op);
    return linalgOp.getIndexingMapsArray();
  }

  FailureOr<mesh::ShardingOption> getShardingOption(Operation *op) const {
    
  }
};

} // namespace

template <typename OpType>
static void registerOne(MLIRContext *ctx) {
  OpType::template attachInterface<StructuredOpShardingInterface<OpType>>(*ctx);
  OpType::template attachInterface<StructuredOpShardingInterface<OpType>>(*ctx);
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerAll(MLIRContext *ctx) {
  (registerOne<OpTypes>(ctx), ...);
}

void registerShardingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LinalgDialect *dialect) {
    registerOne<linalg::GenericOp>(ctx);
    registerAll<
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >(ctx);
  });
}

} // namespace mlir::linalg
