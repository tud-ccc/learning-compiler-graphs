// Copyright (c) 2019 TU Dresden
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic push

#include "llvm/IR/Instructions.h"

#pragma GCC diagnostic pop

#include "funcinfo.hh"

#include <sstream>

using namespace llvm;
using namespace std;

/**
 * Get a unique Name for an LLVM value.
 *
 * This function should always be used instead of the values getName()
 * function. If the object has no name yet, a new unique name is generated
 * based on the default name.
 */
std::string FunctionInfoPass::getUniqueName(const Value &v) {
  if (v.hasName())
    return v.getName();

  auto iter = valueNames.find(&v);
  if (iter != valueNames.end())
    return iter->second;

  stringstream ss;
  if (isa<Argument>(v))
    ss << "arg";
  else if (isa<BasicBlock>(v))
    ss << "bb";
  else if (isa<Function>(v))
    ss << "func";
  else
    ss << "v";

  ss << valueCounter++;

  valueNames[&v] = ss.str();
  return ss.str();
}

unsigned FunctionInfoPass::getUniqueID(const Instruction &inst) {
  auto iter = instructionIDs.find(&inst);
  if (iter != instructionIDs.end())
    return iter->second;
  // else
  unsigned id = instructionCounter++;
  instructionIDs[&inst] = id;
  return id;
}

unsigned FunctionInfoPass::getUniqueID(const MemoryAccess &acc) {
  auto iter = memoryAccessIDs.find(&acc);
  if (iter != memoryAccessIDs.end())
    return iter->second;
  // else
  unsigned id = memoryAccessCounter++;
  memoryAccessIDs[&acc] = id;
  return id;
}

void FunctionInfoPass::init() {
  valueCounter = 0;
  valueNames.clear();
  memoryAccessCounter = 0;
  memoryAccessIDs.clear();
  instructionCounter = 0;
  instructionIDs.clear();
}

ArgInfoPtr FunctionInfoPass::getArgInfo(const Argument &arg) {
  ArgInfoPtr info(new ArgInfo());

  info->name = getUniqueName(arg);

  // collect the type
  string typeName;
  raw_string_ostream rso(typeName);
  arg.getType()->print(rso);
  info->type = rso.str();

  return info;
}

InstructionInfoPtr
FunctionInfoPass::getInstructionInfo(const Instruction &inst) {
  InstructionInfoPtr info(new InstructionInfo());

  string typeName;
  raw_string_ostream rso(typeName);
  inst.getType()->print(rso);

  info->id = getUniqueID(inst);
  info->opcode = inst.getOpcodeName();
  info->type = rso.str();
  info->basicBlock = getUniqueName(*inst.getParent());

  // collect data dependencies
  for (auto &use : inst.operands()) {
    if (isa<Instruction>(use.get())) {
      auto &opInst = *cast<Instruction>(use.get());
      info->operands.push_back(getUniqueID(opInst));
    }
  }

  // collect called function (if this instruction is a call)
  if (isa<CallInst>(inst)) {
    auto &call = cast<CallInst>(inst);
    Function* calledFunction = call.getCalledFunction();
    if(calledFunction != nullptr) {
      info->calls = getUniqueName(*calledFunction);
    }
  }

  // load or store?
  info->isLoadOrStore = false;
  if (isa<LoadInst>(inst))
    info->isLoadOrStore=true;
  if (isa<StoreInst>(inst))
    info->isLoadOrStore=true;

  return info;
}

BasicBlockInfoPtr FunctionInfoPass::getBasicBlockInfo(const BasicBlock &bb) {
  BasicBlockInfoPtr info(new BasicBlockInfo());

  info->name = getUniqueName(bb);

  // collect entry and terminator instruction
  auto &entry = bb.front();
  auto term = bb.getTerminator();
  info->entryInst = getUniqueID(entry);
  info->termInst = getUniqueID(*term);

  // collect all successors
  for (auto *succ : term->successors()) {
    info->successors.push_back(getUniqueName(*succ));
  }

  return info;
}

MemoryAccessInfoPtr FunctionInfoPass::getMemoryAccessInfo(MemoryAccess &acc) {
  MemoryAccessInfoPtr info(new MemoryAccessInfo());

  info->id = getUniqueID(acc);
  info->block = getUniqueName(*acc.getBlock());

  if (isa<MemoryUseOrDef>(acc)) {
    if (isa<MemoryUse>(acc))
      info->type = "use";
    else
      info->type = "def";

    auto inst = cast<MemoryUseOrDef>(acc).getMemoryInst();
    if (inst != nullptr) {
      info->inst = getUniqueID(*inst);
    } else {
      info->inst = -1;
      assert(info->type == "def");
      info->type = "live on entry";
    }

    auto dep = cast<MemoryUseOrDef>(acc).getDefiningAccess();
    if (dep != nullptr) {
      info->dependencies.push_back(getUniqueID(*dep));
    }
  } else {
    info->type = "phi";
    info->inst = -1;
    auto &phi = cast<MemoryPhi>(acc);
    for (unsigned i = 0; i < phi.getNumIncomingValues(); i++) {
      auto dep = phi.getIncomingValue(i);
      info->dependencies.push_back(getUniqueID(*dep));
    }
  }

  return info;
}

bool FunctionInfoPass::runOnFunction(llvm::Function &func) {
  init(); // wipe all data from the previous run

  // create a new info object and invalidate the old one
  info = FunctionInfoPtr(new FunctionInfo());

  info->name = getUniqueName(func);
  info->isVarArg = func.isVarArg();
  info->numArgs = func.arg_size();
  info->entryBlock = getUniqueName(func.getEntryBlock());

  string rtypeName;
  raw_string_ostream rso(rtypeName);
  func.getReturnType()->print(rso);
  info->returnType = rso.str();

  // collect all instructions and basic blocks
  for (auto &bb : func.getBasicBlockList()) {
    for (auto &inst : bb) {
      info->instructions.push_back(getInstructionInfo(inst));
    }
    info->basicBlocks.push_back(getBasicBlockInfo(bb));
  }

  // collect all arguments
  for (auto &arg : func.args()) {
    info->args.push_back(getArgInfo(arg));
  }

  // dump app memory accesses
  auto &mssaPass = getAnalysis<MemorySSAWrapperPass>();
  auto &mssa = mssaPass.getMSSA();
  for (auto &bb : func.getBasicBlockList()) {
    // live on entry
    auto entry = mssa.getLiveOnEntryDef();
    info->memoryAccesses.push_back(getMemoryAccessInfo(*entry));

    // memory phis
    auto phi = mssa.getMemoryAccess(&bb);
    if (phi != nullptr) {
      info->memoryAccesses.push_back(getMemoryAccessInfo(*phi));
    }

    // memory use or defs
    for (auto &inst : bb) {
      auto access = mssa.getMemoryAccess(&inst);
      if (access != nullptr) {
        info->memoryAccesses.push_back(getMemoryAccessInfo(*access));
      }
    }
  }

  // indicate that nothing was changed
  return false;
}

void FunctionInfoPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<MemorySSAWrapperPass>();
  au.setPreservesAll();
}

char FunctionInfoPass::ID = 0;

static RegisterPass<FunctionInfoPass> X("funcinfo", "Function Info Extractor",
                                        true /* Only looks at CFG */,
                                        true /* Analysis Pass */);
