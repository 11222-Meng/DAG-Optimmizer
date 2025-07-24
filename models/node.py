class Node:
    def __init__(self, node_id, node_type, output_tensor_size):
        """
        初始化DAG节点
        :param node_id: 节点ID
        :param node_type: 节点类型 (Conv2d, ReLU等)
        :param output_tensor_size: 输出张量大小 (用于计算cost/gain)
        """
        self.id = node_id
        self.type = node_type
        self.output_tensor_size = output_tensor_size
        self.children = []
        self.parents = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def add_parent(self, parent_node):
        self.parents.append(parent_node)

    def is_bottleneck(self):
        """检查节点是否是瓶颈层"""
        bottleneck_types = ['Conv2d', 'MaxPool2d', 'BatchNorm2d']
        return self.type in bottleneck_types

    def __repr__(self):
        return f"Node(id={self.id}, type={self.type})"