#include <memory>
#include <stack>
#include <iostream>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

class Traverser {
public:
  void visitLoadOp(triton::LoadOp op) {
    op.dump();
  }
  void visitDotOp(triton::DotOp op) {
    op.dump();
    visit(op.getOperand(0).getDefiningOp());
    visit(op.getOperand(1).getDefiningOp());
  }
  void visit(Operation *op) {
    if (auto new_op = dyn_cast<triton::DotOp>(op)) {
      visitDotOp(new_op);
    } else if (auto new_op = dyn_cast<triton::LoadOp>(op)) {
      visitLoadOp(new_op);
    }
  }
};

class WarpSpecializationAnalysisPass
    : public TritonAnalyzeWarpSpecializationBase<WarpSpecializationAnalysisPass> {

public:

  void runOnOperation() override {
    std::cout << "WarpSpecializationAnalysisPass\n";

    ModuleOp m = getOperation();
    m.dump();
    auto t = Traverser();
    for (auto func : m.getOps<triton::FuncOp>()) {
      // find all scf.for ops
      for (auto forOp : func.getOps<scf::ForOp>()) {
        forOp.walk([&](Operation *op) {
          t.visit(op);
        });
      }
    }
  }
};

std::unique_ptr<Pass> triton::createWarpSpecializationAnalysisPass() {
  return std::make_unique<WarpSpecializationAnalysisPass>();
}
