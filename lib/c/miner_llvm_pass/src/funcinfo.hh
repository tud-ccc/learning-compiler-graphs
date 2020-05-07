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

#pragma once

#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include <unordered_map>

// Forward Declarations
struct ArgInfo;
struct BasicBlockInfo;
struct FunctionInfo;
struct InstructionInfo;
struct MemoryAccessInfo;

using ArgInfoPtr = std::unique_ptr<ArgInfo>;
using BasicBlockInfoPtr = std::unique_ptr<BasicBlockInfo>;
using FunctionInfoPtr = std::unique_ptr<FunctionInfo>;
using InstructionInfoPtr = std::unique_ptr<InstructionInfo>;
using MemoryAccessInfoPtr = std::unique_ptr<MemoryAccessInfo>;

struct ArgInfo {
  std::string name;
  std::string type;
};

struct BasicBlockInfo {
  std::string name;
  unsigned entryInst;
  unsigned termInst;
  std::vector<std::string> successors;
};

struct FunctionInfo {
  std::string name;
  unsigned numArgs;
  bool isVarArg;
  std::string returnType;
  std::string entryBlock;
  std::vector<ArgInfoPtr> args;
  std::vector<BasicBlockInfoPtr> basicBlocks;
  std::vector<InstructionInfoPtr> instructions;
  std::vector<MemoryAccessInfoPtr> memoryAccesses;
};

struct InstructionInfo {
  unsigned id;
  std::string type;
  std::string opcode;
  std::string basicBlock;
  std::string calls;
  bool isLoadOrStore;
  std::vector<unsigned> operands;
};

struct MemoryAccessInfo {
  unsigned id;
  std::string type;
  int inst;
  std::string block;
  std::vector<unsigned> dependencies;
};

class FunctionInfoPass : public llvm::FunctionPass {
private:
  FunctionInfoPtr info;

public:
  static char ID;

  FunctionInfoPass() : llvm::FunctionPass(ID), info(nullptr) {}

  bool runOnFunction(llvm::Function &func) override;

  void getAnalysisUsage(llvm::AnalysisUsage &au) const override;

  const FunctionInfoPtr &getInfo() const { return info; }
  FunctionInfoPtr &getInfo() { return info; }

private:
  // keep track of ids and names
  unsigned instructionCounter;
  std::unordered_map<const llvm::Instruction *, unsigned> instructionIDs;
  unsigned memoryAccessCounter;
  std::unordered_map<const llvm::MemoryAccess *, unsigned> memoryAccessIDs;
  unsigned valueCounter;
  std::unordered_map<const llvm::Value *, std::string> valueNames;

  /// clear all data structures
  void init();

  // helper functions
  unsigned getUniqueID(const llvm::Instruction &inst);
  unsigned getUniqueID(const llvm::MemoryAccess &acc);
  std::string getUniqueName(const llvm::Value &v);

  ArgInfoPtr getArgInfo(const llvm::Argument &arg);
  BasicBlockInfoPtr getBasicBlockInfo(const llvm::BasicBlock &bb);
  InstructionInfoPtr getInstructionInfo(const llvm::Instruction &inst);
  MemoryAccessInfoPtr getMemoryAccessInfo(llvm::MemoryAccess &acc);
};
