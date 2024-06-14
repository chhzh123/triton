#include <memory>
#include <stack>
#include <iostream>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

std::string getSsaId(Value value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  value.printAsOperand(os, OpPrintingFlags().assumeVerified());
  return str;
}

class Traverser {
public:
  void visitProgramIdOp(triton::GetProgramIdOp op) {
    std::cout << "pid";
  }
  void visitConstantOp(arith::ConstantOp op) {
    if (mlir::isa<mlir::IntegerType>(op.getType())) {
      std::cout << std::to_string(mlir::cast<IntegerAttr>(op.getValue()).getInt());
    }
  }
  void visitSplat(triton::SplatOp op) {
    if (mlir::isa<BlockArgument>(op.getOperand())) {
      std::cout << "Splat(" << getSsaId(op.getOperand()) << ")";
    } else {
      std::cout << "Splat(";
      visit(op.getOperand());
      std::cout << ")";
    }
  }
  void visitMakeRange(triton::MakeRangeOp op) {
    std::cout << "MakeRange(" << op.getStart() << ", " << op.getEnd() << ")";
  }
  void visitRemSI(arith::RemSIOp op) {
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " % ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitDivSI(arith::DivSIOp op) {
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " / ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitExpandDims(triton::ExpandDimsOp op) {
    std::cout << "ExpandDims(";
    visit(op.getOperand());
    std::cout << ")";
  }
  void visitMul(arith::MulIOp op) {
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " * ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitBroadcast(triton::BroadcastOp op) {
    std::cout << "Broadcast(";
    visit(op.getOperand());
    std::cout << ")";
  }
  void visitAdd(arith::AddIOp op) {
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " + ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitSub(arith::SubIOp op) {
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " - ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitMinSI(arith::MinSIOp op) {
    std::cout << "min(";
    visit(op.getOperand(0));
    std::cout << ", ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitAddptr(triton::AddPtrOp op) {
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " + ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitLoadOp(triton::LoadOp op) {
    if (mlir::isa<BlockArgument>(op.getOperand(0))) {
      std::cout << "*(";
      // get the block of operand 0 and then get the defining scf.for operation for this block
      auto blockArg = dyn_cast<BlockArgument>(op.getOperand(0));
      auto block = blockArg.getOwner();
      auto forOp = block->getParentOp();
      // blockArg.dump();
      // std::cout << "argNum: " << blockArg.getArgNumber() << ", ";
      // visit(dyn_cast<scf::ForOp>(forOp).getInitArgs()[blockArg.getArgNumber() - 2]);
      // print out all the for op operands
      std::cout << "\nfor(";
      for (auto arg : forOp->getOperands()) {
        arg.dump();
      }
      std::cout << ") \n";
      visit(forOp->getOperand(blockArg.getArgNumber() + 1));
      std::cout << ")";
    }
  }
  void visitDotOp(triton::DotOp op) {
    std::cout << "DotOp: ";
    visit(op.getOperand(0));
    std::cout << " x ";
    visit(op.getOperand(1));
    std::cout << "\n";
  }
  void visit(Value value) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      std::cout << getSsaId(blockArg);
    } else {
      auto op = value.getDefiningOp();
      if (auto new_op = dyn_cast<triton::GetProgramIdOp>(op)) {
        visitProgramIdOp(new_op);
      } else if (auto new_op = dyn_cast<arith::ConstantOp>(op)) {
        visitConstantOp(new_op);
      } else if (auto new_op = dyn_cast<triton::DotOp>(op)) {
        visitDotOp(new_op);
      } else if (auto new_op = dyn_cast<triton::LoadOp>(op)) {
        visitLoadOp(new_op);
      } else if (auto new_op = dyn_cast<triton::AddPtrOp>(op)) {
        visitAddptr(new_op);
      } else if (auto new_op = dyn_cast<triton::SplatOp>(op)) {
        visitSplat(new_op);
      } else if (auto new_op = dyn_cast<triton::BroadcastOp>(op)) {
        visitBroadcast(new_op);
      } else if (auto new_op = dyn_cast<triton::ExpandDimsOp>(op)) {
        visitExpandDims(new_op);
      } else if (auto new_op = dyn_cast<arith::AddIOp>(op)) {
        visitAdd(new_op);
      } else if (auto new_op = dyn_cast<arith::SubIOp>(op)) {
        visitSub(new_op);
      } else if (auto new_op = dyn_cast<arith::MulIOp>(op)) {
        visitMul(new_op);
      } else if (auto new_op = dyn_cast<arith::DivSIOp>(op)) {
        visitDivSI(new_op);
      } else if (auto new_op = dyn_cast<arith::RemSIOp>(op)) {
        visitRemSI(new_op);
      } else if (auto new_op = dyn_cast<arith::MinSIOp>(op)) {
        visitMinSI(new_op);
      } else if (auto new_op = dyn_cast<triton::MakeRangeOp>(op)) {
        visitMakeRange(new_op);
      }
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
      // debug
      // GraphDumper().dumpToFile(func, "func.dot");
      // find all scf.for ops
      for (auto forOp : func.getOps<scf::ForOp>()) {
        forOp.walk([&](triton::DotOp op) {
          t.visit(op);
        });
      }
    }
  }
};

std::unique_ptr<Pass> triton::createWarpSpecializationAnalysisPass() {
  return std::make_unique<WarpSpecializationAnalysisPass>();
}
