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

#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <map>

#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Core/CheckerRegistry.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Analysis/Analyses/LiveVariables.h"

#include "clang/Basic/LangOptions.h"
#include "clang/AST/AST.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Analysis/CFG.h"
#include "llvm/Support/raw_ostream.h"

#include "json.hh"

using namespace clang;
using namespace clang::driver;
using namespace clang::ento;
using namespace clang::tooling;
using namespace nlohmann;

static llvm::cl::OptionCategory ToolingSampleCategory("Tooling Sample");
static llvm::cl::opt<std::string> outFileName("o", llvm::cl::desc("Filename to write the graph as json"),
                                              llvm::cl::cat(ToolingSampleCategory));

struct StmtInfo {
    Stmt* clangStmt = nullptr;

    // Data
    std::string name;
    std::string operatorStr;
    std::string valueStr;
};
using StmtInfoPtr = std::shared_ptr<StmtInfo>;

struct DeclInfo {
    ValueDecl* clangDecl = nullptr;

    // Data
    int scalarType;
    std::string functionNameStr;

};
using DeclInfoPtr = std::shared_ptr<DeclInfo>;

struct NodeContainer {
    int nodeId = -1;
    bool isRoot = false;

    StmtInfoPtr stmtInfo = nullptr;
    DeclInfoPtr declInfo = nullptr;

    std::vector<std::shared_ptr<NodeContainer>> livenessRelations;
    std::vector<std::shared_ptr<NodeContainer>> astRelations;
};
using NodeContainerPtr = std::shared_ptr<NodeContainer>;

struct FunctionContainer {
    std::string _name;
    int _scalarReturnType;

    std::vector<NodeContainerPtr> _functionArguments;
    NodeContainerPtr _bodyRootStmt;
};
using FunctionContainerPtr = std::shared_ptr<FunctionContainer>;


class ClangCodeGraph {
public:
    static ClangCodeGraph &getInstance() {
        static ClangCodeGraph instance;
        return instance;
    }

    ClangCodeGraph(ClangCodeGraph const &) = delete;

    void operator=(ClangCodeGraph const &) = delete;

public:
    NodeContainerPtr GetNodeContainerByClangStmt(const Stmt *clangStmt) {
        std::map<const Stmt*, NodeContainerPtr>::iterator it;
        it = _allStmts.find(clangStmt);

        if(it != _allStmts.end()) {
            return it->second;
        }

        std::cerr << "No NodeContainer has been found for clangStmt: " << clangStmt->getStmtClassName() << std::endl;
        return nullptr;
    }

    NodeContainerPtr GetNodeContainerByClangDecl(const ValueDecl *clangDecl) {
        std::map<const ValueDecl*, NodeContainerPtr>::iterator it;
        it = _allDecls.find(clangDecl);

        if(it != _allDecls.end()) {
            return it->second;
        }

        std::cerr << "No NodeContainer has been found for clangDecl: " << clangDecl->getDeclName().getAsString() << std::endl;
        return nullptr;
    }

    json ToJson() {
        AssignNodeIds();

        json jFunctions;

        // Functions
        for (std::vector<FunctionContainerPtr>::iterator it = _functionContainers.begin();
             it != _functionContainers.end(); ++it) {

            std::string name = (*it)->_name;
            int scalarReturnType = (*it)->_scalarReturnType;
            std::vector<NodeContainerPtr> functionArguments = (*it)->_functionArguments;
            NodeContainerPtr bodyRootStmt = (*it)->_bodyRootStmt;

            json jFunction;

            // 0. Basics
            jFunction["name"] = name;
            jFunction["type"] = scalarReturnType;

            // 1. Arguments
            json jArguments;
            for (std::vector<NodeContainerPtr>::iterator it = functionArguments.begin();
                 it != functionArguments.end(); ++it) {

                NodeContainerPtr currentContainer = (*it);

                json jArgument;
                jArgument["name"] = "FunctionArgument";

                // Specific statement information
                // Declarations
                if (currentContainer->declInfo->clangDecl) {
                    jArgument["type"] = currentContainer->declInfo->scalarType;
                }

                // Liveness relations
                json jLivenessRelations;
                if (currentContainer->livenessRelations.empty() == false) {
                    for (std::vector<NodeContainerPtr>::iterator it = currentContainer->livenessRelations.begin();
                         it != currentContainer->livenessRelations.end(); ++it) {
                        jLivenessRelations.push_back((*it)->nodeId);
                    }
                    jArgument["liveness_relations"] = jLivenessRelations;
                }

                jArguments.push_back(jArgument);
            }

            // 2. Body
            json jBody;
            std::stack<NodeContainerPtr> stmtStack;
            stmtStack.push(bodyRootStmt);

            while (stmtStack.empty() == false) {
                // Get and remove top element
                NodeContainerPtr currentContainer = stmtStack.top();
                stmtStack.pop();

                // Build JSON
                json jNode;

                jNode["name"] = currentContainer->stmtInfo->name;
                jNode["is_root"] = currentContainer->isRoot;

                // Specific declaration information
                // Declarations
                if (currentContainer->declInfo) {
                    // Type
                    jNode["type"] = currentContainer->declInfo->scalarType;

                    // CalleeName
                    if (!currentContainer->declInfo->functionNameStr.empty()) {
                        jNode["function_name"] = currentContainer->declInfo->functionNameStr;
                    }
                }

                // Specific statement information
                // IntegerLiteral
                if (!currentContainer->stmtInfo->valueStr.empty()) {
                    jNode["value"] = currentContainer->stmtInfo->valueStr;
                }

                // Unary operator
                // Binary operator
                if (!currentContainer->stmtInfo->operatorStr.empty()) {
                    jNode["operator"] = currentContainer->stmtInfo->operatorStr;
                }

                // AST relations
                json jAstRelations;
                if (currentContainer->astRelations.empty() == false) {
                    for (std::vector<NodeContainerPtr>::iterator it = currentContainer->astRelations.begin();
                         it != currentContainer->astRelations.end(); ++it) {
                        jAstRelations.push_back((*it)->nodeId);

                        // Also add to DFS traversal stack
                        stmtStack.push(*it);
                    }
                    jNode["ast_relations"] = jAstRelations;
                }

                // Liveness relations
                json jLivenessRelations;
                if (currentContainer->livenessRelations.empty() == false) {
                    for (std::vector<NodeContainerPtr>::iterator it = currentContainer->livenessRelations.begin();
                         it != currentContainer->livenessRelations.end(); ++it) {
                        jLivenessRelations.push_back((*it)->nodeId);
                    }
                    jNode["liveness_relations"] = jLivenessRelations;
                }

                jBody.push_back(jNode);
            }

            jFunction["arguments"] = jArguments;
            jFunction["body"] = jBody;

            jFunctions.push_back(jFunction);
        }

        json jRoot;
        jRoot["num_functions"] = _numFunctions;
        jRoot["functions"] = jFunctions;

        return jRoot;
    }

    void addStmt(NodeContainerPtr sInfo) {
        _allStmts.insert(std::pair<const Stmt*, NodeContainerPtr>(sInfo->stmtInfo->clangStmt, sInfo));
    }

    void addDecl(NodeContainerPtr sInfo) {
        _allDecls.insert(std::pair<const ValueDecl*, NodeContainerPtr>(sInfo->declInfo->clangDecl, sInfo));
    }

private:
    void AssignNodeIds() {
        for (std::vector<FunctionContainerPtr>::iterator it = _functionContainers.begin();
             it != _functionContainers.end(); ++it) {

            NodeContainerPtr bodyRootStmt = (*it)->_bodyRootStmt;

            int currentNodeId = 0;

            std::stack<NodeContainerPtr> stmtStack;
            stmtStack.push(bodyRootStmt);

            while (stmtStack.empty() == false) {
                // Get and remove top element
                NodeContainerPtr currentContainer = stmtStack.top();
                stmtStack.pop();

                currentContainer->nodeId = currentNodeId;
                currentNodeId++;

                // Add element's children to stack
                for (std::vector<NodeContainerPtr>::iterator it = currentContainer->astRelations.begin();
                     it != currentContainer->astRelations.end(); ++it) {
                    stmtStack.push(*it);
                }
            }
        }
    }

public:
    std::vector<FunctionContainerPtr> _functionContainers;

    std::map<const Stmt*, NodeContainerPtr> _allStmts;
    std::map<const ValueDecl*, NodeContainerPtr> _allDecls;

    int _numFunctions = 0;

private:
    ClangCodeGraph() {}
};


// AST
class CustomASTVisitor : public RecursiveASTVisitor<CustomASTVisitor> {
public:
    CustomASTVisitor(ASTContext &context) : _context(context) {}

    bool VisitStmt(Stmt *s) {

        return true;
    }

    bool VisitFunctionDecl(FunctionDecl *f) {
        // Create function container
        FunctionContainerPtr fnInfo(new FunctionContainer);

        // Only function definitions (with bodies), not declarations.
        if (f->hasBody() && f->getDeclName().isIdentifier()) {
            std::string functionName = f->getName().data();
            if(functionName == "__sputc") {
              return true;
            }

            if(!_context.getSourceManager().isInMainFile(f->getBeginLoc())) {
              return true;
            }

            // Increment number of functions
            ClangCodeGraph::getInstance()._numFunctions++;

            // Extract function name
            fnInfo->_name = f->getNameInfo().getName().getAsString();

            // Extract return type
            if(f->getReturnType()->isScalarType()) {
                fnInfo->_scalarReturnType = f->getReturnType()->getScalarTypeKind();
            } else if(f->getReturnType()->isVoidType()){
                fnInfo->_scalarReturnType = -1;
            } else {
                fnInfo->_scalarReturnType = -2;
            }

            // 1. Extract arguments
            for (ParmVarDecl *arg : f->parameters()) {
                NodeContainerPtr sInfo(new NodeContainer);
                sInfo->declInfo.reset(new DeclInfo);
                sInfo->declInfo->clangDecl = dyn_cast<ValueDecl>(arg);
                if(sInfo->declInfo->clangDecl->getType()->isScalarType()) {
                    sInfo->declInfo->scalarType = sInfo->declInfo->clangDecl->getType()->getScalarTypeKind();
                } else {
                    sInfo->declInfo->scalarType = -2;
                }

                fnInfo->_functionArguments.push_back(sInfo);
                ClangCodeGraph::getInstance().addDecl(sInfo);
            }

            // 2. Extract body
            Stmt *funcBody = f->getBody();

            // Do DFS on Statements
            std::stack<NodeContainerPtr> stmtStack;

            // Create Statement object and add to CodeGraph
            NodeContainerPtr sInfo(new NodeContainer);

            sInfo->isRoot = true;

            sInfo->stmtInfo.reset(new StmtInfo);
            sInfo->stmtInfo->clangStmt = funcBody;
            sInfo->stmtInfo->name = sInfo->stmtInfo->clangStmt->getStmtClassName();

            fnInfo->_bodyRootStmt = sInfo;
            ClangCodeGraph::getInstance().addStmt(sInfo);

            stmtStack.push(sInfo);

            while (stmtStack.empty() == false) {
                // Get and remove top element
                NodeContainerPtr currentContainer = stmtStack.top();
                stmtStack.pop();

                // Add element's children to stack
                for (StmtIterator it = currentContainer->stmtInfo->clangStmt->child_begin();
                     it != currentContainer->stmtInfo->clangStmt->child_end(); ++it) {
                    if (*it) { // Catch Literals which are children but a nullptr
//                        (*it)->dump();

                        // Decls
                        if (const auto *ds = dyn_cast<DeclStmt>(*it)) {
                            for (auto it_ds = ds->decl_begin(); it_ds != ds->decl_end(); ++it_ds) {
                                if(auto *nd = dyn_cast<ValueDecl>(*it_ds)) {

                                    if (!ClangCodeGraph::getInstance().GetNodeContainerByClangDecl(nd)) {
                                        NodeContainerPtr sInfo(new NodeContainer);
                                        sInfo->stmtInfo.reset(new StmtInfo);
                                        sInfo->stmtInfo->clangStmt = (*it);
                                        sInfo->stmtInfo->name = sInfo->stmtInfo->clangStmt->getStmtClassName();

                                        sInfo->declInfo.reset(new DeclInfo);
                                        sInfo->declInfo->clangDecl = nd;
                                        if (nd->getType()->isScalarType()) {
                                            sInfo->declInfo->scalarType = nd->getType()->getScalarTypeKind();
                                        } else {
                                            sInfo->declInfo->scalarType = -1;
                                        }

                                        stmtStack.push(sInfo);

                                        currentContainer->astRelations.push_back(sInfo);
                                        ClangCodeGraph::getInstance().addDecl(sInfo);
                                        ClangCodeGraph::getInstance().addStmt(sInfo);
                                    }
                                }
                            }


                        // Others
                        } else {
                            NodeContainerPtr sInfo(new NodeContainer);
                            sInfo->stmtInfo.reset(new StmtInfo);
                            sInfo->stmtInfo->clangStmt = (*it);
                            sInfo->stmtInfo->name = sInfo->stmtInfo->clangStmt->getStmtClassName();

                            // IntegerLiteral
                            if (const IntegerLiteral *ds = dyn_cast<IntegerLiteral>(sInfo->stmtInfo->clangStmt)) {
                                sInfo->stmtInfo->valueStr = std::to_string((int) ds->getValue().roundToDouble());
                            }

                            // Unary operator
                            if (const UnaryOperator *ds = dyn_cast<UnaryOperator>(sInfo->stmtInfo->clangStmt)) {
                                sInfo->stmtInfo->operatorStr = ds->getOpcodeStr(ds->getOpcode());
                            }

                            // Binary operator
                            if (const BinaryOperator *ds = dyn_cast<BinaryOperator>(sInfo->stmtInfo->clangStmt)) {
                                sInfo->stmtInfo->operatorStr = ds->getOpcodeStr();
                            }

                            // DeclRefExpr
                            if (const DeclRefExpr *ds = dyn_cast<DeclRefExpr>(sInfo->stmtInfo->clangStmt)) {
                                if (!ClangCodeGraph::getInstance().GetNodeContainerByClangDecl(ds->getDecl())) {
                                    sInfo->declInfo.reset(new DeclInfo);
                                    sInfo->declInfo->clangDecl = (ValueDecl *) ds->getDecl();
                                    if (ds->getType()->isScalarType()) {
                                        sInfo->declInfo->scalarType = ds->getType()->getScalarTypeKind();
                                    }
                                    sInfo->declInfo->functionNameStr = ds->getDecl()->getNameAsString();
                                }
                            }

                            stmtStack.push(sInfo);
                            currentContainer->astRelations.push_back(sInfo);
                            ClangCodeGraph::getInstance().addStmt(sInfo);
                        }
                    }
                }
            }

            //CFG
            std::unique_ptr<CFG> sourceCFG = CFG::buildCFG(f, funcBody, &_context, CFG::BuildOptions());

//            sourceCFG->print(llvm::errs(), LangOptions(), true);

            ClangCodeGraph::getInstance()._functionContainers.push_back(fnInfo);
        }

        return true;
    }

private:
    ASTContext &_context;
};

class MyASTConsumer : public ASTConsumer {
public:
    MyASTConsumer(ASTContext &context) : _visitor(context) {}

    bool HandleTopLevelDecl(DeclGroupRef DR) override {
//        llvm::errs() << "MyASTConsumer::HandleTopLevelDecl" << "\n";

        for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
            // Traverse the declaration using our AST visitor.
            _visitor.TraverseDecl(*b);
            //(*b)->dump();
        }
        return true;
    }

private:
    CustomASTVisitor _visitor;
};

// LiveVariables
class LiveVariablesObserver : public LiveVariables::Observer {
public:
    void observeStmt(const Stmt *S,
                     const CFGBlock *currentBlock,
                     const LiveVariables::LivenessValues &V) override {

        // Get NodeContainer structure from CodeGraph
        NodeContainerPtr toStmt = ClangCodeGraph::getInstance().GetNodeContainerByClangStmt(S);
        if(toStmt) {
            // In case it is a DeclRefExpr, add it to the NodeContainer in the CodeGraph structure
            if (const DeclRefExpr *dre = dyn_cast<DeclRefExpr>(S)) {
                const ValueDecl *d = dre->getDecl();

                NodeContainerPtr fromStmt = ClangCodeGraph::getInstance().GetNodeContainerByClangDecl(d);
                if (fromStmt) {
                    fromStmt->livenessRelations.push_back(toStmt);
                }
            }
        }
    }
};

class LiveVariablesChecker : public Checker<check::ASTCodeBody> {
public:
    void checkASTCodeBody(const Decl *D, AnalysisManager &Mgr,
                          BugReporter &BR) const {
        llvm::errs() << "LiveVariablesChecker::checkASTCodeBody" << "\n";

        if (LiveVariables *L = Mgr.getAnalysis<LiveVariables>(D)) {
//            L->dumpBlockLiveness(Mgr.getSourceManager());

            LiveVariablesObserver obs;
            L->runOnAllBlocks(obs);
        }
    }
};

class CustomFrontendAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        llvm::errs() << "CustomFrontendAction::CreateASTConsumer" << "\n";

        // Create consumers
        std::vector<std::unique_ptr<ASTConsumer>> consumers;

        // LiveVariables consumer
        std::unique_ptr<AnalysisASTConsumer> analysis_consumer = CreateAnalysisConsumer(CI);
        CI.getAnalyzerOpts()->CheckersControlList = {{"custom.LiveVariablesChecker", true}};
        analysis_consumer->AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
            Registry.addChecker<LiveVariablesChecker>("custom.LiveVariablesChecker", "LiveVariablesChecker");
        });
        consumers.push_back(std::move(analysis_consumer));

        // AST consumer
        consumers.push_back(llvm::make_unique<MyASTConsumer>(CI.getASTContext()));

        return llvm::make_unique<MultiplexConsumer>(std::move(consumers));
    }
};

std::unique_ptr<FrontendActionFactory> CustomFrontendActionFactory() {
    class SimpleFrontendActionFactory : public FrontendActionFactory {
    public:
        FrontendAction *create() override {
            return new CustomFrontendAction();
        }
    };

    return std::unique_ptr<FrontendActionFactory>(
            new SimpleFrontendActionFactory());
}

int main(int argc, const char **argv) {

    // Run Clang tool for AST extraction and Clang checker for LiveVariables extraction
    CommonOptionsParser op(argc, argv, ToolingSampleCategory);
    ClangTool Tool(op.getCompilations(), op.getSourcePathList());

    if (Tool.run(CustomFrontendActionFactory().get()) != 0) {
        return 1;
    }

    // Dump codegraph as JSON to console
    llvm::outs() << ClangCodeGraph::getInstance().ToJson().dump(4);

    // Optinal: Dump codegraph as JSON to file
    if(!outFileName.getValue().empty()) {
        std::ofstream jsonOutFile(outFileName);
        jsonOutFile << ClangCodeGraph::getInstance().ToJson().dump(4);
        jsonOutFile.close();
    }

    return 0;
}