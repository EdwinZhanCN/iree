// RUN: iree-compile --compile-to=hal --iree-hal-target-device=metal \
// RUN:   --iree-metal-target-platform=ios --iree-metal-compile-to-metallib=false \
// RUN:   %s | FileCheck %s

// A row reduction on the default Apple target lowers to subgroup arithmetic
// and shuffles. On iOS SPIRV-Cross emits SIMD-group functions for those only
// when told to; otherwise it uses quad-group functions and rejects the
// reduction.

// CHECK:      hal.executable.binary public @metal_msl_fb <
// CHECK-SAME:   format = "metal-msl-fb"
func.func @row_sum(%input: tensor<32x1024xf32>) -> tensor<32xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
  %sum = linalg.reduce ins(%input : tensor<32x1024xf32>) outs(%fill : tensor<32xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
      %add = arith.addf %in, %acc : f32
      linalg.yield %add : f32
    }
  return %sum : tensor<32xf32>
}
