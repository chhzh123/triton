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

std::string getCmpPredicate(arith::CmpIOp op) {
  switch (op.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return "==";
  case arith::CmpIPredicate::ne:
    return "!=";
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return "<";
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return "<=";
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return ">";
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return ">=";
  }
  assert(false && "unsupported compare type");
  return "";
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
    appendNode(op.getResult(), "Constant");
    if (mlir::isa<mlir::IntegerType>(op.getType())) {
      std::cout << std::to_string(mlir::cast<IntegerAttr>(op.getValue()).getInt());
    } else if (mlir::isa<mlir::FloatType>(op.getType())) {
      std::cout << std::to_string(mlir::cast<FloatAttr>(op.getValue()).getValueAsDouble());
    // tensor type
    } else if (mlir::isa<mlir::RankedTensorType>(op.getType())) {
      auto tensorType = mlir::cast<mlir::RankedTensorType>(op.getType());
      auto tensorValue = mlir::cast<mlir::DenseElementsAttr>(op.getValue());
      // std::cout << "[";
      // if values are integers
      if (mlir::isa<IntegerType>(tensorType.getElementType())) {
        for (auto it : tensorValue.getValues<IntegerAttr>()) {
          std::cout << it.getInt();// << ", ";
          break;
        }
      }
      // if values are floats
      else if (mlir::isa<FloatType>(tensorType.getElementType())) {
        for (auto it : tensorValue.getValues<FloatAttr>()) {
          std::cout << it.getValueAsDouble();// << ", ";
          break;
        }
      }
      // std::cout << "]";
    }
  }
  void visitSplat(triton::SplatOp op) {
    appendNode(op.getResult(), "Splat");
    appendEdge(op.getOperand(), op.getResult());
    // std::cout << "Splat(";
    visit(op.getOperand());
    // std::cout << ")";
  }
  void visitMakeRange(triton::MakeRangeOp op) {
    appendNode(op.getResult(), "MakeRange");
    std::cout << "MakeRange(" << op.getStart() << ", " << op.getEnd() << ")";
  }
  void visitRemSI(arith::RemSIOp op) {
    appendNode(op.getResult(), "RemSI");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " % ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitDivSI(arith::DivSIOp op) {
    appendNode(op.getResult(), "DivSI");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " / ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitDivF(arith::DivFOp op) {
    appendNode(op.getResult(), "DivF");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " / ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitExpandDims(triton::ExpandDimsOp op) {
    appendNode(op.getResult(), "ExpandDims");
    appendEdge(op.getOperand(), op.getResult());
    // std::cout << "ExpandDims(";
    visit(op.getOperand());
    // std::cout << ")";
  }
  void visitBroadcast(triton::BroadcastOp op) {
    appendNode(op.getResult(), "Broadcast");
    appendEdge(op.getOperand(), op.getResult());
    // std::cout << "Broadcast(";
    visit(op.getOperand());
    // std::cout << ")";
  }
  void visitAddI(arith::AddIOp op) {
    appendNode(op.getResult(), "AddI");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " + ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitAddF(arith::AddFOp op) {
    appendNode(op.getResult(), "AddF");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " + ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitSubI(arith::SubIOp op) {
    appendNode(op.getResult(), "SubI");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " - ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitSubF(arith::SubFOp op) {
    appendNode(op.getResult(), "SubF");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " - ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitMulI(arith::MulIOp op) {
    appendNode(op.getResult(), "MulI");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " * ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitMulF(arith::MulFOp op) {
    appendNode(op.getResult(), "MulF");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " * ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitMinSI(arith::MinSIOp op) {
    appendNode(op.getResult(), "MinSI");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "min(";
    visit(op.getOperand(0));
    std::cout << ", ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitExtSI(arith::ExtSIOp op) {
    appendNode(op.getResult(), "ExtSI");
    appendEdge(op.getOperand(), op.getResult());
    // std::cout << "ExtSI(";
    visit(op.getOperand());
    // std::cout << ")";
  }
  void visitTruncFOp(arith::TruncFOp op) {
    appendNode(op.getResult(), "TruncF");
    appendEdge(op.getOperand(), op.getResult());
    // std::cout << "TruncF(";
    visit(op.getOperand());
    // std::cout << ")";
  }
  void visitExpOp(math::ExpOp op) {
    appendNode(op.getResult(), "Exp");
    appendEdge(op.getOperand(), op.getResult());
    std::cout << "exp(";
    visit(op.getOperand());
    std::cout << ")";
  }
  void visitExp2Op(math::Exp2Op op) {
    appendNode(op.getResult(), "Exp2");
    appendEdge(op.getOperand(), op.getResult());
    std::cout << "exp2(";
    visit(op.getOperand());
    std::cout << ")";
  }
  void visitCmpIOp(arith::CmpIOp op) {
    appendNode(op.getResult(), "CmpI");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " " << getCmpPredicate(op) << " ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitMaxNumF(arith::MaxNumFOp op) {
    appendNode(op.getResult(), "MaxNumF");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "max(";
    visit(op.getOperand(0));
    std::cout << ", ";
    visit(op.getOperand(1));
    std::cout << ")";
  }
  void visitSelectOp(arith::SelectOp op) {
    appendNode(op.getResult(), "Select");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "Select(";
    visit(op.getOperand(0));
    std::cout << ", ";
    visit(op.getOperand(1));
    std::cout << ", ";
    visit(op.getOperand(2));
    std::cout << ")";
  }
  void visitAddptr(triton::AddPtrOp op) {
    is_iter = true;
    appendNode(op.getResult(), "AddPtr");
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    std::cout << "(";
    visit(op.getOperand(0));
    std::cout << " + ";
    // if (this->is_iv)
    //   std::cout << getSsaId(this->topForOp.getInductionVar()) << " * ";
    visit(op.getOperand(1));
    std::cout << ")";
    is_iter = false;
  }
  void visitLoadOp(triton::LoadOp op) {
    appendNode(op.getResult(), "Load", true);
    is_iter = true;
    appendEdge(op.getOperand(0), op.getResult());
    if (mlir::isa<BlockArgument>(op.getOperand(0))) {
      appendNode(op.getOperand(0), "BlockArg");
      std::cout << "*(";
      // get the block of operand 0 and then get the defining scf.for operation for this block
      auto blockArg = dyn_cast<BlockArgument>(op.getOperand(0));
      auto block = blockArg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());
      visit(blockArg);
      // get initial value of the corresponding block argument (long expressions)
      // visit(forOp.getInitArgs()[blockArg.getArgNumber() - 1]); // arg0 is the loop variable
      std::cout << ")";
    } else {
      std::cout << "*(";
      visit(op.getOperand(0));
      std::cout << ")";
    }
    is_iter = false;
  }
  void visitStoreOp(triton::StoreOp op) {
    appendNode(op.getOperand(0), "Store", true);
    appendEdge(op.getOperand(1), op.getOperand(0), true);
    is_iter = true;
    std::cout << "*(";
    visit(op.getOperand(0));
    std::cout << ") = ";
    is_iter = false;
    visit(op.getOperand(1));
  }
  void visitDotOp(triton::DotOp op) {
    appendNode(op.getResult(), "Dot", true);
    appendEdge(op.getOperand(0), op.getResult());
    appendEdge(op.getOperand(1), op.getResult());
    visit(op.getOperand(0));
    std::cout << " x ";
    visit(op.getOperand(1));
  }
  void visitReduceOp(triton::ReduceOp op) {
    appendNode(*(op.getResult().begin()), "Reduce", true);
    appendEdge(op.getOperand(0), *(op.getResult().begin()));
    std::cout << "Reduce(";
    visit(op.getOperand(0));
    std::cout << ")";
  }
  void visitYieldOp(scf::YieldOp op) {
    for (auto operand : op.getOperands()) {
      std::cout << getSsaId(operand) << " = ";
      visit(operand);
      std::cout << "\n";
    }
  }
  void visit(Operation* op) {
      if (this->visited.count(op) > 0) {
        return;
      }
      visited.insert(op);
      if (auto new_op = dyn_cast<triton::GetProgramIdOp>(op)) {
        visitProgramIdOp(new_op);
      } else if (auto new_op = dyn_cast<arith::ConstantOp>(op)) {
        visitConstantOp(new_op);
      } else if (auto new_op = dyn_cast<triton::DotOp>(op)) {
        visitDotOp(new_op);
      } else if (auto new_op = dyn_cast<math::ExpOp>(op)) {
        visitExpOp(new_op);
      } else if (auto new_op = dyn_cast<math::Exp2Op>(op)) {
        visitExp2Op(new_op);
      } else if (auto new_op = dyn_cast<triton::LoadOp>(op)) {
        visitLoadOp(new_op);
      } else if (auto new_op = dyn_cast<triton::StoreOp>(op)) {
        visitStoreOp(new_op);
      } else if (auto new_op = dyn_cast<triton::AddPtrOp>(op)) {
        visitAddptr(new_op);
      } else if (auto new_op = dyn_cast<triton::SplatOp>(op)) {
        visitSplat(new_op);
      } else if (auto new_op = dyn_cast<triton::BroadcastOp>(op)) {
        visitBroadcast(new_op);
      } else if (auto new_op = dyn_cast<triton::ExpandDimsOp>(op)) {
        visitExpandDims(new_op);
      } else if (auto new_op = dyn_cast<arith::AddIOp>(op)) {
        visitAddI(new_op);
      } else if (auto new_op = dyn_cast<arith::AddFOp>(op)) {
        visitAddF(new_op);
      } else if (auto new_op = dyn_cast<arith::SubIOp>(op)) {
        visitSubI(new_op);
      } else if (auto new_op = dyn_cast<arith::SubFOp>(op)) {
        visitSubF(new_op);
      } else if (auto new_op = dyn_cast<arith::MulIOp>(op)) {
        visitMulI(new_op);
      } else if (auto new_op = dyn_cast<arith::MulFOp>(op)) {
        visitMulF(new_op);
      } else if (auto new_op = dyn_cast<arith::DivSIOp>(op)) {
        visitDivSI(new_op);
      } else if (auto new_op = dyn_cast<arith::DivFOp>(op)) {
        visitDivF(new_op);
      } else if (auto new_op = dyn_cast<arith::RemSIOp>(op)) {
        visitRemSI(new_op);
      } else if (auto new_op = dyn_cast<arith::ExtSIOp>(op)) {
        visitExtSI(new_op);
      } else if (auto new_op = dyn_cast<arith::TruncFOp>(op)) {
        visitTruncFOp(new_op);
      } else if (auto new_op = dyn_cast<arith::MinSIOp>(op)) {
        visitMinSI(new_op);
      } else if (auto new_op = dyn_cast<arith::CmpIOp>(op)) {
        visitCmpIOp(new_op);
      } else if (auto new_op = dyn_cast<arith::MaxNumFOp>(op)) {
        visitMaxNumF(new_op);
      } else if (auto new_op = dyn_cast<arith::SelectOp>(op)) {
        visitSelectOp(new_op);
      } else if (auto new_op = dyn_cast<triton::MakeRangeOp>(op)) {
        visitMakeRange(new_op);
      } else if (auto new_op = dyn_cast<scf::YieldOp>(op)) {
        visitYieldOp(new_op);
      } else if (auto new_op = dyn_cast<triton::ReduceOp>(op)) {
        visitReduceOp(new_op);
      } else {
        std::cout << "Unknown op: " << op->getName().getStringRef().str() << "\n";
      }
  }
  void visit(Value value) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      std::cout << getSsaId(blockArg);
    } else {
      auto op = value.getDefiningOp();
      // check if the operation has been visited
      visit(op);
    }
  }
  void appendNode(Value op, std::string name, bool is_tile_statement = false) {
    // if (is_iter && (name != "AddPtr" && name != "BlockArg"))
    //   return;
    std::string id = getSsaId(op);
    if (name == "Store")
      id += "-S";
    nodestr += "  \"" + id + "\" [label = \"" + id + " " + name + "\"";
    if (is_tile_statement)
      nodestr += ", shape = \"box\"";
    else
      nodestr += ", shape = \"ellipse\"";
    if (name == "Load")
      nodestr += ", color = \"blue\"";
    nodestr += "];\n";
  }
  void appendEdge(Value src, Value dest, bool is_store = false) {
    std::string dest_id = getSsaId(dest);
    if (is_store)
      dest_id += "-S";
    edgestr += "  \"" + getSsaId(src) + "\" -> \"" + dest_id + "\"";
    if (this->is_iter)
      edgestr += "[color = \"orange\"]\n";
    else
      edgestr += "\n";
  }
  std::string getDag() {
    return "digraph G {\n" + nodestr + "\n" + edgestr + "}\n";
  }
private:
  scf::ForOp topForOp;
  bool is_iter = false;
  DenseSet<Operation *> visited;
  std::string nodestr = "";
  std::string edgestr = "";
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
      for (auto forOp : func.getOps<scf::ForOp>()) {
        auto t = Traverser(forOp);
        for (auto op = forOp.getBody()->getOperations().rbegin(); op != forOp.getBody()->getOperations().rend(); ++op) {
          if (llvm::isa<scf::YieldOp, triton::StoreOp>(*op)) {
            t.visit(&(*op));
          }
        }
        std::cout << t.getDag();
      }
    }
  }
};

std::unique_ptr<Pass> triton::createWarpSpecializationAnalysisPass() {
  return std::make_unique<WarpSpecializationAnalysisPass>();
}
