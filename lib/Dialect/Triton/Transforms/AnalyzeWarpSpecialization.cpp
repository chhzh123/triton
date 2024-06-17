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
  Traverser(scf::ForOp op) {
    this->topForOp = op;
  }
  void visitProgramIdOp(triton::GetProgramIdOp op) {
    std::cout << "pid";
  }
  void visitConstantOp(arith::ConstantOp op) {
    if (mlir::isa<mlir::IntegerType>(op.getType())) {
      std::cout << std::to_string(mlir::cast<IntegerAttr>(op.getValue()).getInt());
    } else if (mlir::isa<mlir::FloatType>(op.getType())) {
      std::cout << std::to_string(mlir::cast<FloatAttr>(op.getValue()).getValueAsDouble());
    // tensor type
    } else if (mlir::isa<mlir::RankedTensorType>(op.getType())) {
      auto tensorType = mlir::cast<mlir::RankedTensorType>(op.getType());
      auto tensorValue = mlir::cast<mlir::DenseElementsAttr>(op.getValue());
      // std::cout << "[";
      for (auto it : tensorValue.getValues<IntegerAttr>()) {
        std::cout << it.getInt();// << ", ";
        break;
      }
      // std::cout << "]";
    }
  }
  void visitSplat(triton::SplatOp op) {
    // std::cout << "Splat(";
    visit(op.getOperand());
    // std::cout << ")";
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
    // std::cout << "ExpandDims(";
    visit(op.getOperand());
    // std::cout << ")";
  }
  void visitMul(arith::MulIOp op) {
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " * ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitBroadcast(triton::BroadcastOp op) {
    // std::cout << "Broadcast(";
    visit(op.getOperand());
    // std::cout << ")";
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
    if (this->is_iv)
      std::cout << getSsaId(this->topForOp.getInductionVar()) << " * ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitLoadOp(triton::LoadOp op) {
    if (mlir::isa<BlockArgument>(op.getOperand(0))) {
      std::cout << "*(";
      // get the block of operand 0 and then get the defining scf.for operation for this block
      auto blockArg = dyn_cast<BlockArgument>(op.getOperand(0));
      auto block = blockArg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());
      visit(blockArg);
      // get initial value of the corresponding block argument (long expressions)
      // visit(forOp.getInitArgs()[blockArg.getArgNumber() - 1]); // arg0 is the loop variable
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
  void visitYieldOp(scf::YieldOp op) {
    // std::cout << "YieldOp: ";
    is_iv = true;
    for (auto operand : op.getOperands()) {
      std::cout << getSsaId(operand) << " = ";
      visit(operand);
      std::cout << "\n";
    }
    is_iv = false;
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
      } else if (auto new_op = dyn_cast<scf::YieldOp>(op)) {
        visitYieldOp(new_op);
      }
    }
  }
private:
  scf::ForOp topForOp;
  bool is_iv = false;
};

class WarpSpecializationAnalysisPass
    : public TritonAnalyzeWarpSpecializationBase<WarpSpecializationAnalysisPass> {

public:

  void runOnOperation() override {
    std::cout << "WarpSpecializationAnalysisPass\n";

    ModuleOp m = getOperation();
    m.dump();
    for (auto func : m.getOps<triton::FuncOp>()) {
      // debug
      // GraphDumper().dumpToFile(func, "func.dot");
      // find all scf.for ops
      for (auto forOp : func.getOps<scf::ForOp>()) {
        auto t = Traverser(forOp);
        forOp.walk([&](scf::YieldOp op) {
          t.visitYieldOp(op);
        });
      }
    }
  }
};

std::unique_ptr<Pass> triton::createWarpSpecializationAnalysisPass() {
  return std::make_unique<WarpSpecializationAnalysisPass>();
}
