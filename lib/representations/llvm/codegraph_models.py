# Copyright (c) 2019 TU Dresden
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""LLVM IR graph representation."""

import pydot

import utils


# Entity classes
class CodeGraph(object):
    """Represents a code graph."""

    def __init__(self):
        self.functions = []

        self.all_instructions = {}
        self.all_basic_blocks = {}

    def visit(self, visitor: object) -> None:
        """Traverses graph."""
        visitor.visit(self)
        for function in self.functions:
            function.visit(visitor)

        visitor.visit_end(self)


class Function(object):
    """Represents a function."""

    def __init__(self, name: str):
        self.name = name
        self.basic_blocks = []
        self.entry_instr = None

    def visit(self, visitor: object) -> None:
        """Traverses graph."""
        visitor.visit(self)
        for basic_block in self.basic_blocks:
            basic_block.visit(visitor)

        visitor.visit_end(self)


class BasicBlock(object):
    """Represents a basic block."""

    def __init__(self, name: str):
        self.name = name
        self.instructions = []

    def visit(self, visitor: object) -> None:
        """Traverses graph."""
        visitor.visit(self)
        for instruction in self.instructions:
            instruction.visit(visitor)

        visitor.visit_end(self)


class Instruction(object):
    """Represents an instruction."""

    def __init__(self, opcode):
        self.opcode = opcode
        self.edges = []

    def visit(self, visitor: object) -> None:
        """Traverses graph."""
        visitor.visit(self)
        for cfg_edge in self.edges:
            cfg_edge.visit(visitor)
        visitor.visit_end(self)


class Edge(object):
    """Represents an edge."""

    def __init__(self, edge_type: int, src: object, dest: object):
        self.edge_type = edge_type
        self.src = src
        self.dest = dest

    def visit(self, visitor: object) -> None:
        """Traverses graph."""
        visitor.visit(self)
        visitor.visit_end(self)


# Visitor classes
class VisitorBase(object):
    """Base class for LLVM IR graph visitors."""

    def __init__(self):
        self.node_types = {}
        self.edge_types = {}

    def visit(self, obj: object) -> None:
        """Hook method called on beginning of graph traversal."""
        pass

    def visit_end(self, obj: object) -> None:
        """Hook method called on end of graph traversal."""
        pass

    def _get_id_for_node_type(self, name: str) -> int:
        return self._get_unique_id(self.node_types, name) + 1

    def _get_id_for_edge_type(self, name: str) -> int:
        if name == 'cfg':
            return 1
        if name == 'dataflow':
            return 3
        if name == 'call':
            return 5
        if name == 'memaccess':
            return 7

        raise Exception()

    def _get_id_for_reverse_edge_type(self, name: str) -> int:
        if name == 'cfg':
            return 2
        if name == 'dataflow':
            return 4
        if name == 'call':
            return 6
        if name == 'memaccess':
            return 8

        raise Exception()

    def _get_unique_id(self, mapping: dict, name: str) -> int:
        id_new = len(mapping)
        if name not in mapping:
            mapping[name] = {
                'id': id_new,
                'count': 0
            }

        mapping[name]['count'] += 1
        return mapping[name]['id']


class StatisticsVisitor(VisitorBase):
    """Gathers various statistics of a LLVM IR graph."""

    def __init__(self):
        super(StatisticsVisitor, self).__init__()

        self.num_codegraphs = 0
        self.current_num_nodes = 0
        self.current_num_functions = 0
        self.num_nodes = []

    def visit(self, obj: object) -> None:
        if isinstance(obj, CodeGraph):
            self.num_codegraphs += 1

            self.current_num_nodes = 0
            self.current_num_functions = 0

        if isinstance(obj, Instruction):
            self._get_id_for_node_type(obj.opcode)
            self.current_num_nodes += 1

        if isinstance(obj, Edge):
            self._get_id_for_edge_type(obj.edge_type)

    def visit_end(self, obj: object):
        if isinstance(obj, CodeGraph):
            if self.current_num_nodes != 0:
                self.num_nodes.append(self.current_num_nodes)

    def get_current_num_nodes(self) -> int:
        """Returns current number of nodes."""
        return self.current_num_nodes

    def get_summary(self) -> dict:
        """Returns a summary."""
        # Edge types are always the same and have a fixed order
        edge_types = ['', 'cfg', 'cfg', 'dataflow', 'dataflow', 'call', 'call', 'memaccess',
                      'memaccess', '', '']

        return {
            'node_types': self.node_types,
            'num_node_types': len(self.node_types),
            'edge_types': edge_types,
            'num_edge_types': len(self.edge_types),
            'num_nodes': utils.min_max_avg(self.num_nodes),
            'num_codegraphs': self.num_codegraphs
        }


class NodeInfoExtractionVisitor(VisitorBase):
    """Extracts node infos of a LLVM IR graph."""

    def __init__(self, node_types_of_all_graphs):
        super(NodeInfoExtractionVisitor, self).__init__()

        self.__node_types = {}
        self.__node_types_of_all_graphs = node_types_of_all_graphs

    def visit(self, obj: object) -> None:
        if isinstance(obj, Instruction):
            if obj.node_id not in self.__node_types:
                self.__node_types[obj.node_id] = self.__node_types_of_all_graphs[obj.opcode]['id']

    def get_node_types(self):
        """Returns a list of node types."""
        ret = []
        for idx in range(0, max(self.__node_types.keys()) + 1):
            if idx not in self.__node_types:
                raise Exception()

            if len(ret) > idx:
                raise Exception()

            ret.append(self.__node_types[idx])

        return ret


class EdgeExtractionVisitor(VisitorBase):
    """Extracts the edges of a LLVM IR graph."""

    def __init__(self, edge_types: dict = None):
        super(EdgeExtractionVisitor, self).__init__()

        self.edges = []
        self.edge_types = edge_types

    def visit(self, obj: object) -> None:
        if isinstance(obj, Edge):
            if obj.edge_type == 'dataflow':
                src = obj.dest.node_id
                dest = obj.src.node_id
            else:
                src = obj.src.node_id
                dest = obj.dest.node_id

            edge_info = (src,
                         self.edge_types[
                             obj.edge_type] if self.edge_types else self._get_id_for_edge_type(obj.edge_type),
                         dest)

            if edge_info not in self.edges:
                self.edges.append(edge_info)


class NodeIdCreateVisitor(VisitorBase):
    """Annotates a LLVM IR graph's nodes with incremental ids."""

    def __init__(self):
        super(NodeIdCreateVisitor, self).__init__()

        self.current_node_id = 0

    def visit(self, obj: object) -> None:
        if isinstance(obj, Instruction):
            obj.node_id = self.current_node_id
            self.current_node_id += 1


class DotGraphVisitor(VisitorBase):
    """Creates a graph visualization of a LLVM IR graph with the graphviz dot tool."""

    def __init__(self, debug: int = False):
        super(DotGraphVisitor, self).__init__()

        self.debug = debug

        self.fns = []
        self.bbs = []
        self.instrs = []

        self.dot = pydot.Dot(graph_type="digraph", compound="true")

    def visit(self, obj: object) -> None:
        if isinstance(obj, Function):
            fn_name = 'fn_' + str(len(self.fns))

            cluster = pydot.Cluster(fn_name, label=fn_name)
            self.dot.add_subgraph(cluster)

            self.fns.append(cluster)

        if isinstance(obj, BasicBlock):
            last_fn = self.fns[-1]
            bb_name = 'bb_' + str(len(self.bbs))

            cluster = pydot.Cluster(bb_name, label=bb_name)
            last_fn.add_subgraph(cluster)

            self.bbs.append(cluster)

        if isinstance(obj, Instruction):
            last_bb = self.bbs[-1]
            instr_name = 'inst_' + str(obj.node_id)

            instr_opname = obj.opcode

            if instr_opname == 'load' or instr_opname == 'store':
                color = "green"
            else:
                color = "black"

            if self.debug:
                node = pydot.Node(instr_name, label=instr_opname,
                                  xlabel=str(obj.node_id) + ':' + str(round(obj.prop, 2)),
                                  color=color)
            else:
                node = pydot.Node(instr_name, label=instr_opname, color=color)

            last_bb.add_node(node)

            self.instrs.append(node)

        if isinstance(obj, Edge):
            from_name = 'inst_' + str(obj.src.node_id)
            to_name = 'inst_' + str(obj.dest.node_id)

            if obj.edge_type == 'dataflow':
                color = "blue"
            elif obj.edge_type == 'call':
                color = "red"
            elif obj.edge_type == 'memaccess':
                color = "green"
            else:
                color = "black"

            if self.debug:
                self.dot.add_edge(
                    pydot.Edge(from_name, to_name, xlabel=str(round(obj.prop, 2)), color=color))
            else:
                self.dot.add_edge(pydot.Edge(from_name, to_name, color=color))

    def save_to(self, filename: str, filetype: str) -> None:
        """
        Writes dot graph to file.

        Args:
            filename: A string of the filename and path.
            filetype: A string of the filename.
        """
        try:
            self.dot.write_raw('/tmp/graph.dot')
            (graph,) = pydot.graph_from_dot_file('/tmp/graph.dot')
            if filetype == 'png':
                graph.write_png(filename)
            elif filetype == 'pdf':
                graph.write_pdf(filename)
        except:
            print('Exception in DotGraphVisitor.')


# Functions
def __get_edge_dests_by_type(edges, edge_type):
    """
    Filters a given list of edges by a given type.

    Args:
        edges: A list of edges.
        edge_type: A edge type id that the edges are filtered by.

    Returns:
        A list of filtered edges.
    """
    edges_filtered = []
    for edge in edges:
        if edge.edge_type == edge_type:
            edges_filtered.append(edge.dest)

    return edges_filtered


def get_node_types(graphs):
    """
    Gathers the used node types of a given list of LLVM IR graphs.

    Args:
        graphs: A list of LLVM IR graphs.

    Returns:
        A dict of node types.
    """
    stats_vstr = StatisticsVisitor()

    if type(graphs) == list:
        for graph in graphs:
            graph.visit(stats_vstr)
    elif type(graphs) == dict:
        for _, graph in graphs.items():
            graph.visit(stats_vstr)

    summary = stats_vstr.get_summary()
    node_types = summary['node_types']

    return node_types


def graph_to_export_format(graph, node_types):
    """
    Transforms a given LLVM IR graph into a numerical representation.

    Args:
        graph: A LLVM IR graph object.
        node_types: A dict, which maps node types to node type ids.

    Returns:
        A dict consisting of a list of node type ids and a list of edges.
    """
    # Create node ids
    node_id_vstr = NodeIdCreateVisitor()
    graph.visit(node_id_vstr)

    # Extract node infos
    ni_vstr = NodeInfoExtractionVisitor(node_types)
    graph.visit(ni_vstr)
    nodes = ni_vstr.get_node_types()

    # Extract edges
    ee_vstr = EdgeExtractionVisitor(edge_types={'cfg': 0, 'dataflow': 1, 'memaccess': 2, 'call': 3})
    graph.visit(ee_vstr)
    edges = ee_vstr.edges

    graph_export = {
        utils.T.NODES: nodes,
        utils.T.EDGES: edges
    }

    return graph_export


def codegraphs_create_from_miner_output(jRoot: dict) -> object:
    """
    Creates LLVM IR graphs by parsing the output of the LLVM pass.

    Args:
        jRoot: A dict containing the parsed JSON output of the LLVM pass.

    Returns:
        A list of LLVM IR graphs. One graph per function.
    """
    cgs = []
    for fn_name, fn_obj in jRoot['functions'].items():
        cg = CodeGraph()

        fn = Function(fn_name)
        cg.functions.append(fn)

        entry_instr_id = fn_obj['basic blocks'][fn_obj['entry block']]['entry instruction']

        # Create BBs and instructions
        for bb_name, bb_obj in fn_obj['basic blocks'].items():
            bb = BasicBlock(bb_name)
            fn.basic_blocks.append(bb)
            cg.all_basic_blocks[bb_name] = bb

            # Create all instructions
            for instr_obj in fn_obj['instructions']:
                if instr_obj['basic block'] == bb_name:
                    instr_id = instr_obj['id']

                    instr = Instruction(instr_obj['opcode'])
                    bb.instructions.append(instr)
                    cg.all_instructions[instr_id] = instr

                    # If neccessary, assign entry instruction
                    if instr_id == entry_instr_id:
                        fn.entry_instr = instr

                    # If neccessary, create call target node and edge
                    if instr_obj['calls'] != '':
                        instr_calltarget = Instruction(instr_obj['calls'])
                        bb.instructions.append(instr_calltarget)

                        edge = Edge('call', instr, instr_calltarget)
                        instr.edges.append(edge)

        # Create CFG edges between instructions
        instrs_by_bb = {}
        for instr_obj in fn_obj["instructions"]:
            bb_name = instr_obj['basic block']
            instr_id = instr_obj['id']

            if bb_name not in instrs_by_bb:
                instrs_by_bb[bb_name] = []
            instrs_by_bb[bb_name].append(instr_id)
        for bb_name, bb_obj in fn_obj['basic blocks'].items():
            instrs = instrs_by_bb[bb_name]

            for i in range(0, len(instrs) - 1):
                if i <= len(cg.all_instructions):
                    instr_from = cg.all_instructions[instrs[i]]
                    instr_to = cg.all_instructions[instrs[i + 1]]

                    edge = Edge('cfg', instr_from, instr_to)
                    instr_from.edges.append(edge)

        # Create CFG edges between BBs
        for bb_name, bb_obj in fn_obj['basic blocks'].items():
            for bb_succ_name in bb_obj["successors"]:
                instr_from_id = instrs_by_bb[bb_name][-1]
                instr_from = cg.all_instructions[instr_from_id]
                instr_to_id = instrs_by_bb[bb_succ_name][0]
                instr_to = cg.all_instructions[instr_to_id]

                edge = Edge('cfg', instr_from, instr_to)
                instr_from.edges.append(edge)

        # Create dataflow edges
        for bb_name, bb_obj in fn_obj['basic blocks'].items():
            for instr_obj in fn_obj['instructions']:
                if instr_obj['basic block'] == bb_name:
                    instr_to_id = instr_obj['id']
                    instr_to = cg.all_instructions[instr_to_id]

                    for operand_id in instr_obj['operands']:
                        instr_from_id = operand_id
                        instr_from = cg.all_instructions[instr_from_id]

                        edge = Edge('dataflow', instr_from, instr_to)
                        instr_from.edges.append(edge)

        # Create memory access edges
        for mem_access_obj in fn_obj['memory accesses']:
            instr_to_id = mem_access_obj['inst']
            if instr_to_id != -1:
                instr_to = cg.all_instructions[instr_to_id]

                for dep_idx in mem_access_obj['dependencies']:
                    instr_from_id = fn_obj['memory accesses'][dep_idx]['inst']
                    if instr_from_id != -1:
                        instr_from = cg.all_instructions[instr_from_id]

                        edge = Edge('memaccess', instr_from, instr_to)
                        instr_from.edges.append(edge)
        cgs.append(cg)

    return cgs


def make_dot_graph(graph: object, debug: bool = False):
    """
    Creates a graph visualization of a LLVM IR graph with the graphviz dot tool.

    Args:
        graph: A LLVM IR graph.
        debug: Debug mode (adds verbosity).

    Returns:
        A DotGraphVisitor object holding the state of the dot graph.
    """
    # Create node ids
    nic_vstr = NodeIdCreateVisitor()
    graph.visit(nic_vstr)

    # Create dot graph
    dg_vstr = DotGraphVisitor(debug)
    graph.visit(dg_vstr)

    return dg_vstr


def save_dot_graph(graph: object, filename: str, filetype: str, debug: bool = False):
    """
    Creates a graph visualization of a LLVM IR graph with the graphviz dot tool and writes it to
    disk.

    Args:
        graph: A LLVM IR graph.
        filename: A string with the filename.
        filetype: A string with the filetype, either PDF or PNG.
        debug: Debug mode (adds verbosity).
    """
    dg_vstr = make_dot_graph(graph, debug)

    dg_vstr.save_to(filename, filetype)
