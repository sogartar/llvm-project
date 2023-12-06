//===- Simplifications.h - Mesh Simplifications -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_TRANSFORMS_SPMDIZATION_H
#define MLIR_DIALECT_MESH_TRANSFORMS_SPMDIZATION_H

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace mesh {

ShapedType shardShapedType(ShapedType shape, ClusterOp mesh,
                           MeshShardingAttr sharding);

TypedValue<ShapedType> reshard(OpBuilder &builder, ClusterOp mesh,
                               ShardOp source, ShardOp target,
                               TypedValue<ShapedType> sourceShardValue);

void reshardingRegisterDependentDialects(DialectRegistry &registry);

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_TRANSFORMS_SPMDIZATION_H
