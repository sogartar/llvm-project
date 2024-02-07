//===- Transforms.cpp ---------------------------------------------- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Transforms/Transforms.h"
#include "TransformsDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>
#include <numeric>

namespace mlir::mesh {

namespace {

/// Lower `mesh.process_multi_index` into expression using
/// `mesh.process_linear_index` and `mesh.mesh_shape`.
struct ProcessMultiIndexOpLowering
    : OpRewritePatternWithSymbolTableCollection<ProcessMultiIndexOp> {
  using OpRewritePatternWithSymbolTableCollection::
      OpRewritePatternWithSymbolTableCollection;

  LogicalResult matchAndRewrite(ProcessMultiIndexOp op,
                                PatternRewriter &rewriter) const override {
    MeshOp mesh = getMesh(op, symbolTableCollection);
    if (!mesh) {
      return failure();
    }

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value linearIndex = builder.create<ProcessLinearIndexOp>(mesh);
    ValueRange meshShape = builder.create<MeshShapeOp>(mesh).getResults();
    llvm::errs() << "meshShape.size() = " << meshShape.size() << "\n";
    SmallVector<Value> completeMultiIndex =
        builder.create<affine::AffineDelinearizeIndexOp>(linearIndex, meshShape)
            .getMultiIndex();
    SmallVector<Value> multiIndex;
    ArrayRef<MeshAxis> opMeshAxes = op.getAxes();
    SmallVector<MeshAxis> opAxesIota;
    if (opMeshAxes.empty()) {
      opAxesIota.resize(mesh.getRank());
      std::iota(opAxesIota.begin(), opAxesIota.end(), 0);
      opMeshAxes = opAxesIota;
    }
    llvm::transform(opMeshAxes, std::back_inserter(multiIndex),
                    [&completeMultiIndex](MeshAxis meshAxis) {
                      return completeMultiIndex[meshAxis];
                    });
    rewriter.replaceAllUsesWith(op.getResults(), multiIndex);
    return success();
  }
};

struct AllScatterOpLowering
    : OpRewritePatternWithSymbolTableCollection<AllScatterOp> {
  using OpRewritePatternWithSymbolTableCollection::
      OpRewritePatternWithSymbolTableCollection;

  LogicalResult matchAndRewrite(AllScatterOp op,
                                PatternRewriter &rewriter) const override {
    MeshOp mesh = getMesh(op, symbolTableCollection);
    if (!mesh) {
      return failure();
    }

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());

    Value zero = builder.create<arith::ConstantOp>(builder.getIndexAttr(0));

    Operation::result_range processInGroupMultiIndex =
        builder.create<ProcessMultiIndexOp>(mesh.getSymName(), op.getMeshAxes())
            .getResults();

    Operation::result_range processGroupShape =
        builder.create<MeshShapeOp>(mesh.getSymName(), op.getMeshAxes())
            .getResult();
    Value processGroupSize =
        createCollectiveProcessGroupSize(mesh, op.getMeshAxes(), builder);

    int64_t scatterAxis = op.getScatterAxis().getSExtValue();
    Value operandScatterAxisSize =
        builder.create<tensor::DimOp>(op.getOperand(), scatterAxis);
    Value operandScatterAxisSizeModProcessGroupSize =
        builder.create<arith::RemUIOp>(operandScatterAxisSize,
                                       processGroupSize);
    Value isTargetShapeExactlyDivisible = builder.create<arith::CmpIOp>(
        arith::CmpIPredicate::eq, operandScatterAxisSizeModProcessGroupSize,
        zero);
    builder.create<cf::AssertOp>(isTargetShapeExactlyDivisible,
                                 "Scattering a tensor with axis size that is "
                                 "not exactly divisible by the "
                                 "mesh process group size is not supported.");
    Value resultScatterAxisSize = builder.create<arith::DivUIOp>(
        operandScatterAxisSize, processGroupSize);
    OpFoldResult processInGroupLinearIndex = affine::linearIndexFromShape(
        llvm::to_vector_of<OpFoldResult>(processInGroupMultiIndex),
        llvm::to_vector_of<OpFoldResult>(processGroupShape), builder);

    // extract slice
    RankedTensorType operandType =
        op.getOperand().getType().cast<RankedTensorType>();
    SmallVector<OpFoldResult> sizes;
    for (int64_t i = 0; i < operandType.getRank(); ++i) {
      if (i == scatterAxis) {
        sizes.emplace_back(resultScatterAxisSize);
      } else {
        Value dimSize = builder.create<tensor::DimOp>(op.getOperand(), i);
        sizes.emplace_back(dimSize);
      }
    }
    SmallVector<OpFoldResult> offsets(
        operandType.getRank(), getAsIndexOpFoldResult(builder.getContext(), 0));
    offsets[scatterAxis] =
        ArithBuilder(builder, builder.getLoc())
            .mul(getValueOrCreateConstantIndexOp(builder, builder.getLoc(),
                                                 processInGroupLinearIndex),
                 resultScatterAxisSize);
    SmallVector<OpFoldResult> strides(
        operandType.getRank(), getAsIndexOpFoldResult(builder.getContext(), 1));
    Value slice = builder.create<tensor::ExtractSliceOp>(
        op.getOperand(), offsets, sizes, strides);
    Value newResult =
        builder.create<tensor::CastOp>(op.getResult().getType(), slice);
    rewriter.replaceAllUsesWith(op.getResult(), newResult);

    return success();
  }
};

} // namespace

void processMultiIndexOpLoweringPopulatePatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection) {
  patterns.add<ProcessMultiIndexOpLowering>(symbolTableCollection,
                                            patterns.getContext());
}

void processMultiIndexOpLoweringRegisterDialects(DialectRegistry &registry) {
  registry.insert<affine::AffineDialect, mesh::MeshDialect>();
}

void allScatterOpLoweringPopulatePatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection) {
  patterns.add<AllScatterOpLowering>(symbolTableCollection,
                                     patterns.getContext());
}

void allScatterOpLoweringRegisterDialects(DialectRegistry &registry) {
  registry.insert<affine::AffineDialect, arith::ArithDialect,
                  cf::ControlFlowDialect, mesh::MeshDialect,
                  tensor::TensorDialect>();
}

void populateAllOpLoweringPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection) {
  processMultiIndexOpLoweringPopulatePatterns(patterns, symbolTableCollection);
  allScatterOpLoweringPopulatePatterns(patterns, symbolTableCollection);
}

void registerAllOpLoweringDialects(DialectRegistry &registry) {
  processMultiIndexOpLoweringRegisterDialects(registry);
  allScatterOpLoweringRegisterDialects(registry);
}

TypedValue<IndexType>
createCollectiveProcessGroupSize(MeshOp mesh, ArrayRef<MeshAxis> axes,
                                 ImplicitLocOpBuilder &builder) {
  Operation::result_range meshShape =
      builder.create<mesh::MeshShapeOp>(mesh, axes).getResults();
  return arith::createProduct(builder, builder.getLoc(),
                              llvm::to_vector_of<Value>(meshShape),
                              builder.getIndexType())
      .cast<TypedValue<IndexType>>();
}

} // namespace mlir::mesh
