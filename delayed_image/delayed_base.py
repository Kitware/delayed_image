"""
Abstract nodes
"""
import numpy as np
import ubelt as ub

try:
    import rich as rich_mod
except Exception:
    rich_mod = None


# from kwcoco.util.util_monkey import Reloadable  # NOQA
# @Reloadable.developing  # NOQA
class DelayedOperation(ub.NiceRepr):

    def __init__(self):
        self.meta = {}
        self._opt_logs = []

    def __nice__(self):
        """
        Returns:
            str
        """
        return '{}'.format(self.shape)

    def nesting(self):
        """
        Returns:
            Dict[str, dict]
        """
        def _child_nesting(child):
            if hasattr(child, 'nesting'):
                return child.nesting()
            elif isinstance(child, np.ndarray):
                return {
                    'type': 'ndarray',
                    'shape': self.subdata.shape,
                }
        # from kwcoco.util import ensure_json_serializable
        meta = self.meta.copy()
        try:
            meta['transform'] = meta['transform'].concise()
        except (AttributeError, KeyError):
            pass
        try:
            meta['channels'] = meta['channels'].concise().spec
        except (AttributeError, KeyError):
            pass
        item = {
            'type': self.__class__.__name__,
            'meta': meta,
        }
        child_nodes = list(self.children())
        if child_nodes:
            child_nestings = [_child_nesting(child) for child in child_nodes]
            item['children'] = child_nestings
        return item

    def as_graph(self):
        """
        Builds the underlying graph structure as a networkx graph with human
        readable labels.

        Returns:
            networkx.DiGraph
        """
        graph = self._traversed_graph()
        for node_id, node_data in graph.nodes(data=True):
            item = node_data['obj']
            sub_meta = node_data['meta']
            # Add some concise labels to the graph
            if 'transform' in sub_meta:
                sub_meta['transform'] = sub_meta['transform'].concise()
                sub_meta['transform'].pop('type')
            if 'channels' in sub_meta:
                sub_meta['channels'] = str(sub_meta['channels'].spec)
                sub_meta.pop('num_channels', None)
            sub_meta.pop('jagged', None)
            sub_meta.pop('border_value', None)
            sub_meta.pop('antialias', None)
            sub_meta.pop('interpolation', None)
            sub_meta.pop('noop_eps', None)
            if 'fpath' in sub_meta:
                sub_meta['fname'] = ub.Path(sub_meta.pop('fpath')).name
            param_key = ub.repr2(sub_meta, sort=0, compact=1, nl=0, precision=4)
            short_type = item.__class__.__name__.replace('Delayed', '')
            node_data['label'] = f'{short_type} {param_key}'
        return graph

    def _traverse(self):
        """
        A flat list of all descendent nodes and their parents

        Yields:
            Tuple[None | DelayedOperation, DelayedOperation] :
                tules of parent / child nodes. Discarding the parents
                will be a list of all nodes.
        """
        # Might be useful in _set_nested_params or other functions that
        # need to touch all descendants. This will be faster than recursion
        stack = [(None, self)]
        while stack:
            parent, item = stack.pop()
            yield parent, item
            for child in item.children():
                stack.append((item, child))

    def _leafs(self):
        """
        Iterates over all leafs in the tree.

        Yields:
            Tuple[DelayedOperation] :
        """
        # Might be useful in _set_nested_params or other functions that
        # need to touch all descendants. This will be faster than recursion
        stack = [self]
        while stack:
            item = stack.pop()
            children = list(item.children())
            if children:
                for child in item.children():
                    stack.append(child)
            else:
                yield item

    def _leaf_paths(self):
        """
        Builds all independent paths to leafs.

        Yields:
            Tuple[DelayedOperation, DelayedOperation]:
                The leaf, and the path to it,

        Example:
            >>> from delayed_image import demo
            >>> self = demo.non_aligned_leafs()
            >>> for leaf, part in list(self._leaf_paths()):
            ...     leaf.write_network_text()
            ...     part.write_network_text()

        Example:
            >>> from delayed_image import demo
            >>> import delayed_image
            >>> orig = delayed_image.DelayedLoad.demo().prepare()
            >>> part1 = orig[0:100, 0:100].scale(2, dsize=(128, 128))
            >>> part2 = delayed_image.DelayedNans(dsize=(128, 128))
            >>> self = delayed_image.DelayedChannelConcat([part2, part1])
            >>> for leaf, part in list(self._leaf_paths()):
            ...     leaf.write_network_text()
            ...     part.write_network_text()
        """
        # Might be useful in _set_nested_params or other functions that
        # need to touch all descendants. This will be faster than recursion
        import copy
        stack = [[self]]
        while stack:
            path = stack.pop()
            item = path[-1]
            children = list(item.children())
            if children:
                for child in item.children():
                    stack.append(path + [child])
            else:
                leaf = item
                # We found a path to a leaf, we now need to process it
                prev = None
                assert len(path)
                for part in path[::-1]:
                    if hasattr(part, 'parts'):
                        # Skip concats (todo assert it really is a concat and
                        # not an unhandled op)
                        part = prev
                    else:
                        if prev is not None:
                            if part.subdata is not prev:
                                # The subdata was a skipped node, we need to
                                # contract the operation edge.
                                part = copy.copy(part)
                                part.subdata = prev
                        prev = part
                yield leaf, part

    def _traversed_graph(self):
        """
        A flat list of all descendent nodes and their parents
        """
        import networkx as nx
        import itertools as it
        import math
        counter = it.count(0)
        graph = nx.DiGraph()
        ndigits = int(math.log10(max(1, len(graph.nodes)))) + 1
        # Can't reuse traverse unfortunately
        stack = [(None, self)]
        while stack:
            parent_id, item = stack.pop()

            # There might be copies of the same node in concat graphs so, we
            # cant assume the id will be unique. We can assert a forest
            # structure though.
            node_id = f'{item.__class__.__name__}_{next(counter):0{ndigits}d}'

            graph.add_node(node_id)
            if parent_id is not None:
                graph.add_edge(parent_id, node_id)

            sub_meta = {k: v for k, v in item.meta.items() if v is not None}
            node_data = graph.nodes[node_id]
            node_data['type'] = item.__class__.__name__
            node_data['meta'] = sub_meta
            node_data['obj'] = item

            for child in list(item.children())[::-1]:
                stack.append((node_id, child))
        return graph

    def write_network_text(self, with_labels=True, rich='auto'):
        from delayed_image.helpers import write_network_text
        graph = self.as_graph()
        path = None
        end = '\n'
        if rich == 'auto':
            rich = rich_mod is not None
        if rich:
            path = rich_mod.print
            end = ''
        # TODO: remove once this is merged into networkx itself
        write_network_text(graph, with_labels=with_labels, path=path, end=end)

    @property
    def shape(self):
        """
        Returns:
            None | Tuple[int | None, ...]
        """
        raise NotImplementedError

    def children(self):
        """
        Yields:
            Any:
        """
        raise NotImplementedError
        yield None

    def prepare(self):
        """
        If metadata is missing, perform minimal IO operations in order to
        prepopulate metadata that could help us better optimize the operation
        tree.

        Returns:
            DelayedOperation2
        """
        for child in self.children():
            child.prepare()
        return self

    def _finalize(self):
        """
        This is the method that new nodes should overload.

        Conceptually this works just like the finalize method with the
        exception that it happens at every node in the tree, whereas the public
        facing method only happens once, calls this, and is able to do one-time
        pre and post operations.

        Returns:
            ArrayLike
        """
        raise NotImplementedError

    def finalize(self, prepare=True, optimize=True, **kwargs):
        """
        Evaluate the operation tree in full.

        Args:
            prepare (bool):
                ensure prepare is called to ensure metadata exists if possible
                before optimizing.  Defaults to True.
            optimize (bool):
                ensure the graph is optimized before loading.  Default to True.
            **kwargs: for backwards compatibility, these will allow for
                in-place modification of select nested parameters.

        Returns:
            ArrayLike

        Notes:
            Do not overload this method. Overload
            :func:`DelayedOperation2._finalize` instead.
        """
        if kwargs:
            """
            show dep warnings

            import warnings
            for item in list(warnings.filters):
                if item[0] == 'ignore' and item[2] is DeprecationWarning:
                    warnings.filters.remove(to_remove)
            """
            # Undeprecate, I think I actually like this, but maybe not inplace.
            # It might be better to make this procedure happen in optmize.
            self._set_nested_params(**kwargs)
        if prepare:
            self = self.prepare()
        if optimize:
            self = self.optimize()
        # The protected version of this method does all the work, this function
        # just sits at the user-level and ensures correct final output whereas
        # the protected function can return optimized representations that
        # other _finalize methods can utilize.
        final = self._finalize()
        # Ensure we are array like
        final = final[:]
        # final = np.asanyarray(final) # does not work with xarray
        return final

    def optimize(self):
        """
        Returns:
            DelayedOperation2
        """
        raise NotImplementedError

    def _set_nested_params(self, **kwargs):
        """
        Hack to override nested params on all warps for things like
        interplation / antialias
        """
        graph = self.as_graph()
        for node_id, node_data in graph.nodes(data=True):
            obj = node_data['obj']
            common = ub.dict_isect(kwargs, obj.meta)
            obj.meta.update(common)


class DelayedNaryOperation(DelayedOperation):
    """
    For operations that have multiple input arrays
    """
    def __init__(self, parts):
        super().__init__()
        self.parts = parts

    def children(self):
        """
        Yields:
            Any:
        """
        yield from iter(self.parts)


class DelayedUnaryOperation(DelayedOperation):
    """
    For operations that have a single input array
    """
    def __init__(self, subdata):
        super().__init__()
        self.subdata = subdata

    def children(self):
        """
        Yields:
            Any:
        """
        if self.subdata is not None:
            yield self.subdata
