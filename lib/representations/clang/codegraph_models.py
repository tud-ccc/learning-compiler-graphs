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

"""AST graph representation."""

import pydot

import utils


# Helper functions
def get_id_for_edge_type(name: str) -> int:
    """Returns a edge type id, given a edge type as name."""
    if name == 'AST':
        return 1
    if name == 'LIVE':
        return 3

    raise Exception()


def get_id_for_reverse_edge_type(name: str) -> int:
    """Returns a reverse edge type id, given a edge type as name."""
    if name == 'AST':
        return 2
    if name == 'LIVE':
        return 4

    raise Exception()


def is_forward_edge_type(edge_type_id) -> bool:
    """Returns whether edge is forward or backward edge."""
    if edge_type_id % 2 == 1:
        return True

    return False


def get_edge_name_by_edge_type(edge_type_id) -> str:
    """Returns a edge type string, given a edge type id."""
    if edge_type_id == 1 or edge_type_id == 2:
        return 'AST'
    if edge_type_id == 3 or edge_type_id == 4:
        return 'LIVE'

    raise Exception()


# Entity classes
class CodeGraph(object):
    """Represents a code graph."""

    def __init__(self):
        self.functions = []

        self.has_complex_types = False

    def accept(self, visitor: object, sorting_function=None) -> None:
        """Traverses graph."""
        visitor.visit(self)
        for function in self.functions:
            function.accept(visitor, sorting_function)
        visitor.visit_end(self)


class Function(object):
    """Represents a function."""

    def __init__(self):
        self.name = 'Function'
        self.specifics = {}

        self.all_statements = []
        self.edges = []

        self.function_node_type = -1
        self.node_id = -1

        self.is_first = True

    def accept(self, visitor: object, sorting_function=None) -> None:
        """Traverses graph."""
        if sorting_function:
            edges = sorting_function(self.edges)
        else:
            edges = self.edges

        visitor.visit(self)

        for edge in edges:
            if edge.type == 'AST':
                edge.accept(visitor, sorting_function)

        visitor.visit_end(self)

    def get_root_statement(self):
        """Returns root statement."""
        for edge in self.edges:
            if edge.dest.name == 'CompoundStmt':
                return edge.dest

    def get_arguments(self):
        """Returns arguments"""
        arguments = []

        for edge in self.edges:
            if edge.dest.name == 'FunctionArgument':
                arguments.append(edge.dest)

        return arguments


class Statement(object):
    """Represents a statement."""

    def __init__(self, name):
        self.name = name
        self.specifics = {}

        self.edges = []

        self.node_id = 1337

    def accept(self, visitor: object, sorting_function=None) -> None:
        """Traverses graph."""
        if sorting_function:
            edges = sorting_function(self.edges)
        else:
            edges = self.edges

        visitor.visit(self)

        for edge_idx, edge in enumerate(edges):
            edge.accept(visitor, sorting_function)

            if edge_idx < len(self.edges) - 1:
                visitor.visit_intermediate(self, edge)

        visitor.visit_end(self)


class Edge(object):
    """Represents an edge."""

    def __init__(self, type: int, src: object, dest: object):
        self.type = type
        self.src = src
        self.dest = dest

    def accept(self, visitor: object, sorting_function=None) -> None:
        """Traverses graph."""
        visitor.visit(self)

        if self.type == 'AST' \
                or (self.type == 'LIVE' and self.dest.name == 'IntegerLiteral'):
            self.dest.accept(visitor, sorting_function)

        visitor.visit_end(self)


# Visitor classes
class VisitorBase(object):
    """Base class for all AST graph visitors."""

    def visit(self, obj: object) -> None:
        """Hook method called on beginning of graph traversal."""
        pass

    def visit_intermediate(self, obj: object, edge: object) -> None:
        """Hook method called on mid of graph traversal."""
        pass

    def visit_end(self, obj: object) -> None:
        """Hook method called on end of graph traversal."""
        pass


class StatisticsVisitor(VisitorBase):
    """Gathers various statistics of a AST graph."""

    def __init__(self):
        super(StatisticsVisitor, self).__init__()

        self.num_nodes = 0

    def visit(self, obj: object) -> None:
        if isinstance(obj, Statement) or isinstance(obj, Function):
            self.num_nodes += 1


class DFSNodeIdCreateVisitor(VisitorBase):
    """Annotates a AST graph's nodes with incremental ids."""

    def __init__(self):
        super(DFSNodeIdCreateVisitor, self).__init__()

        self.current_node_id = 0

    def visit(self, obj: object) -> None:
        if isinstance(obj, Statement) or isinstance(obj, Function):
            obj.node_id = self.current_node_id
            self.current_node_id += 1


class EdgeIdxCreateVisitor(VisitorBase):
    """Annotates a AST graph's edges with incremental ids."""

    def __init__(self):
        super(EdgeIdxCreateVisitor, self).__init__()

    def visit(self, obj: object) -> None:
        if isinstance(obj, Statement) or isinstance(obj, Function):
            edge_idxs = {}

            for edge in obj.edges:
                if 'type' not in edge_idxs:
                    edge_idxs['type'] = 0
                else:
                    edge_idxs['type'] += 1

                edge.idx = edge_idxs['type']


class NodeRankCreateVisitor(VisitorBase):
    """Annotates a AST graph's nodes with their rank, i.e. their distance to the root node."""

    def __init__(self):
        super(NodeRankCreateVisitor, self).__init__()

    def visit(self, obj: object) -> None:
        if isinstance(obj, Function):
            obj.rank = 0

            for edge in obj.edges:
                if edge.type == 'AST':
                    statement = edge.dest
                    statement.rank = 1

        if isinstance(obj, Statement):
            for edge in obj.edges:
                if edge.type == 'AST' or (
                        edge.type == 'LIVE' and edge.dest.name == 'IntegerLiteral'):
                    statement = edge.dest
                    statement.rank = obj.rank + 1


class RankNeighborsCreateVisitor(VisitorBase):
    """Creates a dict of AST graph's nodes ranks, i.e. their distance to the root node."""

    def __init__(self):
        super(RankNeighborsCreateVisitor, self).__init__()

        self.last_node_of_rank = {}
        self.first_node_of_rank = {}

    def visit(self, obj: object) -> None:
        if isinstance(obj, Statement) or isinstance(obj, Function):
            if obj.rank not in self.last_node_of_rank:
                self.last_node_of_rank[obj.rank] = obj
                self.first_node_of_rank[obj.rank] = obj
            else:
                self.last_node_of_rank[obj.rank].rank_next = obj
                self.last_node_of_rank[obj.rank] = obj

            # Statements might have the rank_next attribute from a previous iteration of the
            # graph transformer. Therefore: delete if attribute exists
            if hasattr(obj, 'rank_next'):
                del obj.rank_next


class IntegerLiteralLiveEdgeCreateVisitor(VisitorBase):
    """Adds live edges between integer literals and their use in a AST graph."""

    def __init__(self, debug: int = False):
        super(IntegerLiteralLiveEdgeCreateVisitor, self).__init__()
        self.debug = debug

    def visit(self, obj: object) -> None:
        if isinstance(obj, Statement):
            for edge in obj.edges:
                if edge.type == 'AST':
                    if edge.dest.name == 'IntegerLiteral':
                        new_edge = Edge('LIVE', obj, edge.dest)
                        obj.edges.append(new_edge)

                        obj.edges.remove(edge)


class FunctionNameResolvingVisitor(VisitorBase):
    """Resolves function names. In the live analysis, function names are subject to liveness too.
    This visitor dissolves this and makes function names own entities."""

    def __init__(self, debug: int = False):
        super(FunctionNameResolvingVisitor, self).__init__()
        self.debug = debug

    def visit(self, obj: object) -> None:
        if isinstance(obj, Statement):
            for edge in obj.edges:
                if edge.type == 'LIVE' \
                        and obj.name == 'DeclRefExpr' \
                        and edge.dest.name == 'DeclRefExpr' \
                        and 'function_name' in edge.dest.specifics:
                    obj.specifics['function_name'] = edge.dest.specifics['function_name']
                    obj.edges.remove(edge)


class EliminationVisitor(VisitorBase):
    """Eliminates nodes and maintains their relationships by merging."""

    def __init__(self, to_eliminate, debug: int = False):
        super(EliminationVisitor, self).__init__()
        self.debug = debug

        self.to_eliminate = to_eliminate

    def visit(self, obj: object) -> None:
        if isinstance(obj, Statement):
            for edge in obj.edges:
                if edge.type == 'AST' and edge.dest.name in self.to_eliminate:
                    if 'function_name' in edge.dest.specifics:
                        obj.specifics['function_name'] = edge.dest.specifics['function_name']

                    obj.edges.remove(edge)
                    for child_edge in edge.dest.edges:
                        new_edge = Edge(child_edge.type, obj, child_edge.dest)
                        obj.edges.append(new_edge)


class DotGraphVisitor(VisitorBase):
    """Creates a graph visualization of a AST graph with the graphviz dot tool."""

    def __init__(self, debug: int = False):
        super(DotGraphVisitor, self).__init__()
        self.debug = debug

        self.rank_subraphs = {}

        self.dot = pydot.Dot(graph_type="digraph", rankdir="TB")

    def visit(self, obj: object) -> None:
        if isinstance(obj, Function):
            # Create function node
            function_display_name = self._build_node_name(obj)
            node = pydot.Node('node_0', label=function_display_name, color="black", shape='box')

            subgraph = pydot.Subgraph(rank='same')
            subgraph.add_node(node)
            self.dot.add_subgraph(subgraph)

            # Add edges
            for edge in obj.edges:
                from_name = 'node_' + str(edge.src.node_id)
                to_name = 'node_' + str(edge.dest.node_id)

                if edge.type == 'LIVE':
                    color = "blue"
                else:
                    color = "black"

                self.dot.add_edge(pydot.Edge(from_name, to_name, color=color))

        if isinstance(obj, Statement):
            # Get or create subgraph
            if obj.rank not in self.rank_subraphs:
                self.rank_subraphs[obj.rank] = pydot.Subgraph(rank='same')
                self.dot.add_subgraph(self.rank_subraphs[obj.rank])
            subgraph = self.rank_subraphs[obj.rank]

            # Create node
            node_name = 'node_' + str(obj.node_id)
            node_display_name = self._build_node_name(obj)
            node = pydot.Node(node_name, label=node_display_name, color="black", shape='box')

            subgraph.add_node(node)

            # Add invisible edges to enforce node order
            if hasattr(obj, 'rank_next'):
                from_name = 'node_' + str(obj.node_id)
                to_name = 'node_' + str(obj.rank_next.node_id)
                self.dot.add_edge(
                    pydot.Edge(from_name, to_name, color='green' if self.debug else 'invis'))

            # Add edges
            for edge in obj.edges:
                from_name = 'node_' + str(edge.src.node_id)
                to_name = 'node_' + str(edge.dest.node_id)

                if edge.type == 'LIVE':
                    color = "blue"
                else:
                    color = "black"

                if self.debug:
                    self.dot.add_edge(
                        pydot.Edge(from_name, to_name, color=color, xlabel=str(edge.idx)))
                else:
                    self.dot.add_edge(pydot.Edge(from_name, to_name, color=color))

    def _build_node_name(self, obj: object):
        ret = str(obj.name)

        if hasattr(obj, 'specifics'):
            ret += '\l' + str(obj.specifics)

        if self.debug:
            if hasattr(obj, 'node_type_id'):
                ret = 'node_type_id: ' + str(obj.node_type_id) + '\l' + ret

            if hasattr(obj, 'node_id'):
                ret = 'node_id: ' + str(obj.node_id) + '\l' + ret

            if hasattr(obj, 'step_idx'):
                ret = 'step_idx: ' + str(obj.step_idx) + '\l' + ret

            if hasattr(obj, 'p_pick'):
                ret = 'p_pick: ' + '%.2f' % obj.p_pick + '\l' + ret

        return ret

    def save_to(self, filename: str, filetype: str) -> None:
        """
        Writes dot graph to file.

        Args:
            filename: A string of the filename and path.
            filetype: A string of the filename.
        """
        self.dot.write_raw('/tmp/graph.dot')
        (graph,) = pydot.graph_from_dot_file('/tmp/graph.dot')

        if filetype == 'png':
            graph.write_png(filename)
        elif filetype == 'pdf':
            graph.write_pdf(filename)


class NodeTypeIdCreateVisitor(VisitorBase):
    """Assigns unique ids to nodes according to their type."""

    def __init__(self, with_functionnames: bool = True, with_callnames: bool = True):
        super(NodeTypeIdCreateVisitor, self).__init__()

        self.with_functionnames = with_functionnames
        self.with_callnames = with_callnames
        self.node_type_ids_by_statements = {}

    def visit(self, obj: object) -> None:
        if isinstance(obj, Statement) or isinstance(obj, Function):
            classname = obj.__class__.__name__

            if classname == 'Function' and self.with_functionnames == False:
                key_hashed = hash((classname, utils.freeze_dict(obj.specifics)))
            elif obj.name == 'CallExpr' and self.with_callnames == False:
                key_hashed = hash((obj.name))
            elif obj.name == 'IntegerLiteral':
                key_hashed = hash((obj.name))
            else:
                specifics = obj.specifics
                if 'function_name' in specifics:
                    del specifics['function_name']
                key_hashed = hash((obj.name, utils.freeze_dict(specifics)))

            # Add to map
            if key_hashed in self.node_type_ids_by_statements:
                self.node_type_ids_by_statements[key_hashed]['count'] += 1
            else:
                self.node_type_ids_by_statements[key_hashed] = {
                    # id is incremented by 1 because type 0 is reserved as terminator type
                    'id': len(self.node_type_ids_by_statements) + 1,
                    'classname': classname,
                    'name': obj.name,
                    'specifics': obj.specifics,
                    'count': 1
                }

            # Assign node id
            obj.node_type_id = self.node_type_ids_by_statements[key_hashed]['id']


class EdgeExtractionVisitor(VisitorBase):
    """Extracts the edges of a AST graph."""

    def __init__(self, edge_types: dict = None):
        super(EdgeExtractionVisitor, self).__init__()

        self.edges = []
        self.edge_types = edge_types

    def visit(self, obj: object) -> None:
        if isinstance(obj, Edge):
            if obj.type == 'LIVE':
                src = obj.dest.node_id
                dest = obj.src.node_id
            else:
                src = obj.src.node_id
                dest = obj.dest.node_id

            edge_info = (src,
                         self.edge_types[obj.type] if self.edge_types else get_id_for_edge_type(
                             obj.type),
                         dest)

            if edge_info not in self.edges:
                self.edges.append(edge_info)


class NodeInfoExtractionVisitor(VisitorBase):
    """Extracts node infos of a AST graph."""

    def __init__(self):
        super(NodeInfoExtractionVisitor, self).__init__()

        self.__node_types = {}
        self.__node_values = {}

    def visit(self, obj: object) -> None:
        if isinstance(obj, Statement) or isinstance(obj, Function):
            if obj.node_id not in self.__node_types:
                self.__node_types[obj.node_id] = obj.node_type_id

            if 'value' in obj.specifics:
                if obj.node_id not in self.__node_values:
                    self.__node_values[obj.node_id] = int(obj.specifics['value'])

    def node_types(self):
        """Returns node types."""
        ret = []
        for idx in range(0, max(self.__node_types.keys()) + 1):
            if idx not in self.__node_types:
                raise Exception()

            if len(ret) > idx:
                raise Exception()

            ret.append(self.__node_types[idx])

        return ret

    def node_values(self):
        """Returns node values."""
        ret = []
        for i in range(0, max(list(self.__node_types.keys()))):
            if i in self.__node_values:
                ret.append(self.__node_values[i])
            else:
                ret.append(0)

        return ret


class StmtNameQueryVisitor(VisitorBase):
    """Queries the names of all statements of a AST graph."""

    def __init__(self):
        super(StmtNameQueryVisitor, self).__init__()

        self.statement_names = set()

    def visit(self, obj: object) -> None:
        if isinstance(obj, Statement):
            statement_name = obj.name

            self.statement_names.add(statement_name)


# Functions
def transform_graph(graph: object) -> object:
    """
    Transforms graph. This includes elimination of AST nodes of certain types.

    Args:
        graph: An input AST graph.

    Returns:
        The transformed AST graph.
    """
    # Add live edges to IntegerLiteral nodes
    ill_vstr = IntegerLiteralLiveEdgeCreateVisitor()
    for i in range(0, 10):
        graph.accept(ill_vstr)

    fnres_vstr = FunctionNameResolvingVisitor()
    graph.accept(fnres_vstr)

    # Eliminate nodes
    ele_vstr = EliminationVisitor(['ImplicitCastExpr', 'DeclRefExpr'])

    nic_vstr = DFSNodeIdCreateVisitor()
    graph.accept(nic_vstr)
    num_nodes = nic_vstr.current_node_id

    previous_num_nodes = -1

    # Do until fixpoint is reached
    while num_nodes != previous_num_nodes:
        nic_vstr = DFSNodeIdCreateVisitor()
        graph.accept(nic_vstr)
        previous_num_nodes = nic_vstr.current_node_id

        graph.accept(ele_vstr)

        nic_vstr = DFSNodeIdCreateVisitor()
        graph.accept(nic_vstr)
        num_nodes = nic_vstr.current_node_id

    return graph


def make_dot_graph(graph: object, debug: bool = False):
    """
    Creates a graph visualization of a AST graph with the graphviz dot tool.

    Args:
        graph: A AST graph.
        debug: Debug mode (adds verbosity).

    Returns:
        A DotGraphVisitor object holding the state of the dot graph.
    """
    # Assign node ids
    assign_node_ids_in_bfs_order(graph)

    # Assign edge idx
    eic_vstr = EdgeIdxCreateVisitor()
    graph.accept(eic_vstr)

    # Create dot Graph
    rnk_vstr = NodeRankCreateVisitor()
    graph.accept(rnk_vstr)

    rnkn_vstr = RankNeighborsCreateVisitor()
    graph.accept(rnkn_vstr)

    dg_vstr = DotGraphVisitor(debug)
    graph.accept(dg_vstr)

    return dg_vstr


def save_dot_graph(graph: object, filename: str, filetype: str, debug: bool = False):
    """
    Creates a graph visualization of a AST graph with the graphviz dot tool and writes it to disk.

    Args:
        graph: A AST graph.
        filename: A string with the filename.
        filetype: A string with the filetype, either PDF or PNG.
        debug: Debug mode (adds verbosity).
    """
    dg_vstr = make_dot_graph(graph, debug)

    dg_vstr.save_to(filename, filetype)


def assign_node_ids_in_bfs_order(graph: object):
    """
    Annotates AST graph with node ids that are created in BFS traversal order.

    Args:
        graph: A AST graph.
    """
    # Rank nodes according to their distance to the root node
    rnk_vstr = NodeRankCreateVisitor()
    graph.accept(rnk_vstr)

    rnkn_vstr = RankNeighborsCreateVisitor()
    graph.accept(rnkn_vstr)

    # Create a list of nodes to process in this order
    nodes = []
    for rank, first_node in rnkn_vstr.first_node_of_rank.items():
        current_node = first_node

        while True:
            nodes.append(current_node)

            if not hasattr(current_node, 'rank_next'):
                break
            current_node = current_node.rank_next

    for idx, node in enumerate(nodes):
        node.node_id = idx


def get_node_types(graphs, with_functionnames, with_callnames):
    """
    Returns a list of node types of a AST graph.

    Args:
        graphs: A list of AST graphs.
        with_functionnames: Boolean indicating whether function names should be own node types.
        with_callnames: Boolean indicating whether call names should be own node types.

    Returns:
        A list of integer node types.
    """

    nic_vstr = NodeTypeIdCreateVisitor(with_functionnames=with_functionnames,
                                       with_callnames=with_callnames)

    if type(graphs) == list:
        for graph in graphs:
            graph.accept(nic_vstr)
    elif type(graphs) == dict:
        for _, graph in graphs.items():
            graph.accept(nic_vstr)

    node_types = nic_vstr.node_type_ids_by_statements

    return node_types


def graph_to_export_format(graph):
    """
    Transforms a given AST graph into a numerical representation.

    Args:
        graph: A AST graph object.

    Returns:
        A dict consisting of a list of node type ids and a list of edges.
    """
    # Extract node infos
    ni_vstr = NodeInfoExtractionVisitor()
    graph.accept(ni_vstr)
    nodes = ni_vstr.node_types()
    node_values = ni_vstr.node_values()

    # Extract edges
    ee_vstr = EdgeExtractionVisitor(edge_types={'AST': 0, 'LIVE': 1})
    graph.accept(ee_vstr)
    edges = ee_vstr.edges

    graph_export = {
        utils.T.NODES: nodes,
        utils.T.NODE_VALUES: node_values,
        utils.T.EDGES: edges
    }

    return graph_export


def codegraphs_create_from_miner_output(jRoot: dict) -> list:
    """
    Creates AST graphs by parsing the output of the Clang tool.

    Args:
        jRoot: A dict containing the parsed JSON output of the Clang tool.

    Returns:
        A list of AST graphs. One graph per function.
    """
    CLANG_SCALAR_TYPES = ['int*', 'blockPtr', 'objCPtr', 'memberPtr', 'bool', 'int', 'float',
                          'complexInt', 'complexFloat', 'someComplexType']

    cgs = []
    for jFunction in jRoot['functions']:
        cg = CodeGraph()

        # Create function
        function = Function()
        function.name = jFunction['name']
        if jFunction['type'] == -1:
            function.specifics['type'] = 'void'
        else:
            function.specifics['type'] = CLANG_SCALAR_TYPES[jFunction['type']]
        cg.functions.append(function)

        # Create arguments
        if jFunction['arguments'] is not None:
            for node_obj in jFunction['arguments']:
                stmt = Statement(node_obj['name'])

                if stmt.name == 'FunctionArgument':
                    if node_obj['type'] == -1:
                        cg.has_complex_types = True
                    stmt.specifics['type'] = CLANG_SCALAR_TYPES[node_obj['type']]
                    stmt.rank = 0

                stmt_from = function
                stmt_to = stmt

                edge = Edge('AST', stmt_from, stmt_to)
                stmt_from.edges.append(edge)

        # Create statements
        for node_obj in jFunction['body']:
            stmt = Statement(node_obj['name'])

            # Specific statement information
            if stmt.name == 'DeclRefExpr' and 'function_name' in node_obj:
                stmt.specifics['function_name'] = node_obj['function_name']
            if stmt.name == 'DeclStmt':
                if node_obj['type'] == -1:
                    cg.has_complex_types = True
                stmt.specifics['type'] = CLANG_SCALAR_TYPES[node_obj['type']]
            if stmt.name == 'IntegerLiteral':
                stmt.specifics['value'] = node_obj['value']
            if stmt.name == 'UnaryOperator' or stmt.name == 'BinaryOperator' or stmt.name == 'CompoundAssignOperator':
                stmt.specifics['operator'] = node_obj['operator']

            function.all_statements.append(stmt)

            if node_obj['is_root']:
                stmt_from = function
                stmt_to = stmt

                edge = Edge('AST', stmt_from, stmt_to)
                stmt_from.edges.append(edge)

        # Create AST edges
        for node_idx, node_obj in enumerate(jFunction['body']):
            stmt_from = function.all_statements[node_idx]

            if 'ast_relations' in node_obj:
                for stmt_to_idx in node_obj['ast_relations']:
                    stmt_to = function.all_statements[stmt_to_idx]

                    edge = Edge('AST', stmt_from, stmt_to)
                    stmt_from.edges.append(edge)

        # Create Liveness edges for statements
        for node_idx, node_obj in enumerate(jFunction['body']):
            stmt_from = function.all_statements[node_idx]

            if 'liveness_relations' in node_obj:
                for stmt_to_idx in node_obj['liveness_relations']:
                    stmt_to = function.all_statements[stmt_to_idx]

                    edge = Edge('LIVE', stmt_to, stmt_from)
                    stmt_to.edges.append(edge)

        # Create Liveness edges for arguments
        if jFunction['arguments'] is not None:
            for node_idx, node_obj in enumerate(jFunction['arguments']):
                all_arguments = function.get_arguments()
                stmt_from = all_arguments[node_idx]

                if 'liveness_relations' in node_obj:
                    for stmt_to_idx in node_obj['liveness_relations']:
                        stmt_to = function.all_statements[stmt_to_idx]

                        edge = Edge('LIVE', stmt_to, stmt_from)
                        stmt_to.edges.append(edge)
        cgs.append(cg)

    return cgs
