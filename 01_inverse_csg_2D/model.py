import torch
import numpy as np
import uuid
import matplotlib.pyplot as plt

from utils.plot_2D_occupancy import plot_2D_occupancy

class csg_tree_full_binary(torch.nn.Module):
    def __init__(self, depth):
        super().__init__()

        # hyperparameters
        self.depth = depth
        self.num_primitives = 2**depth
        self.temperature = 1000.
        self.omega = 10.

        # primitive parameters
        self.num_params_per_primitive = 6 + 1 # "+1" is a uniform scaling factor
        self.primitive_params = torch.nn.Parameter(torch.FloatTensor(self.num_primitives, self.num_params_per_primitive).uniform_(-0.5, 0.5))
            
        # boolean parameters
        self.num_boolean = 2**depth - 1
        self.num_params_per_boolean = 4
        self.boolean_params = torch.nn.Parameter(torch.FloatTensor(self.num_boolean, self.num_params_per_boolean).uniform_(-0.5, 0.5))
        
    def activation(self, x):
        return torch.sigmoid(-x)
    
    def primitive_forward(self, p):
        """
        Inputs
        p: (b,p,dim) sample points
        primititive_params: a vector of parameters, can be reshaped as (m,num_params_per_primitive)
        
        Ouputs
        out: (b,p,m)
        """
        # get points 
        x = p[:,:,[0]] # b x p x 1
        y = p[:,:,[1]]

        # get quadric parameters
        q0 = self.primitive_params[:,[0]].T.unsqueeze(0) # 1x1xm
        q1 = self.primitive_params[:,[1]].T.unsqueeze(0)
        q2 = self.primitive_params[:,[2]].T.unsqueeze(0)
        q3 = self.primitive_params[:,[3]].T.unsqueeze(0)
        q4 = self.primitive_params[:,[4]].T.unsqueeze(0)
        q5 = self.primitive_params[:,[5]].T.unsqueeze(0)
        scaling = self.primitive_params[:,[6]].T.unsqueeze(0)

        # get quadric parameters
        out = (q0*x**2 + q1*y**2 + q2*x*y + q3*x + q4*y + q5) * scaling
        return out

    def boolean_forward(self, boolean_params, x, y):
        """
        Inputs:
        bool_params: m x 4 real number of boolean parameters 
        x: b x p x m tensor of soft sign values
        y: b x p x m tensor of soft sign values
        """
        boolean_weights = torch.nn.functional.softmax(torch.sin(self.omega * boolean_params) * self.temperature, dim=1) # m x 4
        c0, c1, c2, c3 = boolean_weights[:,0], boolean_weights[:,1], boolean_weights[:,2], boolean_weights[:,3]
        out = (c1 + c2) * x + (c1 + c3) * y +  (c0 - c1 - c2 - c3) * x*y 
        return out

    def forward(self, p):
        """
        Inputs
        p: b x p x dim

        Ouputs
        out: b x p signed values
        """
        s = self.primitive_forward(p) # b x p x m
        s = self.activation(s)

        curIdx = 0
        for d in range(self.depth):
            s_left = s[:, :, ::2] # b, p, m/2
            s_right = s[:, :, 1::2] 

            num_boolean_nodes = s_right.shape[2]
            boolean_params_ii = self.boolean_params[curIdx:curIdx+num_boolean_nodes, :] 
            curIdx = curIdx + num_boolean_nodes

            s = self.boolean_forward(boolean_params_ii, s_left, s_right)
        return s.squeeze(2)
    
    def get_booleans(self):
        boolean_weights = torch.nn.functional.softmax(torch.sin(self.omega * self.boolean_params) * self.temperature, dim=1)
        return boolean_weights
    
    def get_primitives(self, p):
        s = self.primitive_forward(p) # b x p x m
        s = self.activation(s) 
        return s
    

class csg_tree_pointers(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.root = None
        self.depth = model.depth

        self.UNION = 1
        self.INTERSECTION = 2
        self.LEFT_DIFF_RIGHT = 3
        self.RIGHT_DIFF_LEFT = 4

        # create boolean nodes
        node_list = [] # list of boolean nodes (start from root, and left->right)
        learned_boolean_weights = model.get_booleans()
        end_index = learned_boolean_weights.shape[0]
        for d in range(self.depth):
            num_nodes = 2**d
            per_layer_boolean_weights = learned_boolean_weights[end_index-num_nodes:end_index]
            for ii in range(num_nodes):
                # determine the type of boolean operator
                boolean_type = self.boolean_params_to_operation(per_layer_boolean_weights[ii,:])
                cur_node = boolean_node(boolean_type)
                if d == 0:
                    # init root node
                    self.root = cur_node
                else:
                    # init boolean nodes
                    if ii%2 == 0: # odd nodes == left child
                        node_list[0].left = cur_node
                        cur_node.parent = node_list[0]
                    else:  # odd nodes == right child
                        node_list[0].right = cur_node
                        cur_node.parent = node_list[0]
                        node_list.pop(0)
                node_list.append(cur_node)
            end_index = end_index - num_nodes

        # initialize leaf primitive nodes
        for ii in range(2**self.depth):
            prim_params = model.primitive_params[ii,:-1].detach()
            sharpness = model.primitive_params[ii,-1].detach()
            cur_node = primitive_node_quadrics(prim_params * sharpness)
            if ii%2 == 0: # odd nodes == left child
                node_list[0].left = cur_node
                cur_node.parent = node_list[0]
            else:  # odd nodes == right child
                node_list[0].right = cur_node
                cur_node.parent = node_list[0]
                node_list.pop(0)
            
            
    def boolean_params_to_operation(self, boolean_param):
        union = torch.tensor([0,1.,0,0])
        intersection = torch.tensor([1.,0,0,0])
        left_diff_right = torch.tensor([0,0,1.,0])
        right_diff_left = torch.tensor([0,0,0,1.])
        exact_boolean_params = torch.stack((union, intersection, left_diff_right, right_diff_left), dim=0)

        distances = torch.sum((exact_boolean_params - boolean_param)**2,1)
        min_idx = torch.argmin(distances)
        
        if min_idx == 0:
            return self.UNION
        elif min_idx == 1:
            return self.INTERSECTION
        elif min_idx == 2:
            return self.LEFT_DIFF_RIGHT
        elif min_idx == 3:
            return self.RIGHT_DIFF_LEFT
        
    def __print_helper(self, currPtr, indent, last, file=None): 
        # print the tree structure on the screen
        if currPtr != None:
            print(indent, end="", file=file)
            if last:
                print("R----",end="", file=file)
                indent += "     "
            else:
                print("L----", end="",file=file)
                indent += "|    "

            if currPtr.is_primitive:
                print("prim:" + str(currPtr.ID), file=file)
            else:
                print(currPtr.get_boolean_type_str(), file=file)

            self.__print_helper(currPtr.left, indent, False, file)
            self.__print_helper(currPtr.right, indent, True, file)

    def print_tree(self, filename):
        f = open(filename, 'w')
        self.__print_helper(self.root, "", True, f)
        f.close()

    def postorder_traversal(self, root = None):
        if root is None:
            root = self.root
        stack = [root]
        visited_stack = [False]
        ordered_node_list = torch.nn.ModuleList()

        while len(stack)>0:
            cur, visited = stack.pop(), visited_stack.pop()
            if cur:
                if visited:
                    ordered_node_list.append(cur)
                else:
                    stack.append(cur)
                    visited_stack.append(True)
                    stack.append(cur.right)
                    visited_stack.append(False)
                    stack.append(cur.left)
                    visited_stack.append(False)
        return ordered_node_list

    def forward(self, p):
        node_list = self.postorder_traversal()
        output_stack = []
        for ii in range(len(node_list)):
            cur = node_list[ii]

            if cur.is_primitive: # it is a primitive node
                cur_func = cur(p)
                output_stack.append(cur_func)

            else: # it is a boolean node
                right_child_func = output_stack.pop()
                left_child_func = output_stack.pop()
                cur_func = cur(left_child_func, right_child_func)
                output_stack.append(cur_func)
        return output_stack.pop()
    
    def pruning(self, p, threshold_similarity=1e-3):
        self.forward(p) # to generate results on cache
        nodes = self.postorder_traversal()
        for node in nodes:
            if not node.is_primitive: # if this is a boolean node
                node_val = node.cache_values
                node_left_val = node.left.cache_values
                node_right_val = node.right.cache_values
                difference_left = torch.mean((node_val - node_left_val)**2)
                difference_right = torch.mean((node_val - node_right_val)**2)
                difference = torch.minimum(difference_left, difference_right)
                if difference < threshold_similarity: # remove a node
                    # determine which node to keep
                    if difference_left < difference_right:
                        min_node = node.left
                    else:
                        min_node = node.right

                    # reconnect tree
                    if node.parent is None: # node is root node
                        self.root = min_node
                        min_node.parent = None
                    elif node.parent.left == node: # node is left child
                        min_node.parent = node.parent
                        node.parent.left = min_node
                    else: # node is right child
                        min_node.parent = node.parent
                        node.parent.right = min_node

    def get_leaves(self, root = None):
        if root is None:
            root = self.root
        stack = []
        cur = root
        leaf_list = torch.nn.ModuleList()
        while (cur is not None) or len(stack)>0:
            while cur:
                stack.append(cur)
                if cur.is_primitive:
                    leaf_list.append(cur)
                cur = cur.left
            cur = stack.pop()
            cur = cur.right
        return leaf_list    

    def save_all_primitives(self, filename, p):
        self.forward(p)
        leaves = self.get_leaves()
        num_images_each_side = np.ceil(np.sqrt(len(leaves))).astype(int)
        plt.figure(12345, figsize = (num_images_each_side * 3, num_images_each_side * 3))
        plt.clf()
        subplot_index = 1
        for node in leaves:
            plt.subplot(num_images_each_side,num_images_each_side,subplot_index)
            node_val = node.cache_values
            plot_2D_occupancy(node_val)
            plt.title("prim" + str(node.ID))
            subplot_index += 1
        plt.savefig(filename)
    
class primitive_node_quadrics(torch.nn.Module):
    def  __init__(self, primitive_params):
        super().__init__()
        self.ID = str(uuid.uuid4())[:8]
        self.parent = None
        self.left = None
        self.right = None
        self.is_primitive = True
        self.cache_values = None
        self.primitive_params = primitive_params
        self.is_full = False # a flag to determine whether to switch this primitive to full primtive (always output -1)
        self.is_empty = False # a flag to determine whether to switch this primitive to empty primtive (always output 1)

    def __repr__(self): # print function when calling "print(primitive_node())"
        return "Primitive Node: " + str(self.ID)
    
    def activation(self, x):
        return torch.tanh(-x) / 2. + 0.5

    def forward(self, p):
        if self.is_full: # if this is full primitive
            self.cache_values = torch.ones((p.shape[0]))
            return self.cache_values
        
        if self.is_empty: # if this is empty primitive
            self.cache_values = torch.zeros((p.shape[0]))
            return self.cache_values
        
        # get points 
        x = p[:,0] 
        y = p[:,1]

        # get quadric parameters
        q0 = self.primitive_params[0]
        q1 = self.primitive_params[1]
        q2 = self.primitive_params[2]
        q3 = self.primitive_params[3]
        q4 = self.primitive_params[4]
        q5 = self.primitive_params[5]

        # get quadric parameters
        self.cache_values = q0*x**2 + q1*y**2 + q2*x*y + q3*x + q4*y + q5
        self.cache_values = self.activation(self.cache_values)
        return self.cache_values
    
    def twin(self):
        """
        return the twin node (the other node that shares the same parent) of this node
        """
        if self.parent is None: # if this is root
            return None
        if self.parent.left == self:
            return self.parent.right
        elif self.parent.right == self:
            return self.parent.left
        
    def reset_ID(self):
        self.ID = uuid.uuid4()

class boolean_node(torch.nn.Module):
    def  __init__(self, boolean_type):
        super().__init__()
        self.ID = uuid.uuid4()
        self.parent = None
        self.left = None
        self.right = None
        self.is_primitive = False
        self.cache_values = None
        self.boolean_type = boolean_type
        self.is_full = False # this should always be a false because this is not a primitive node
        self.is_empty = False # this should always be a false because this is not a primitive node

        self.UNION = 1
        self.INTERSECTION = 2
        self.LEFT_DIFF_RIGHT = 3
        self.RIGHT_DIFF_LEFT = 4

    def __repr__(self): 
        return "Boolean Node: " + str(self.ID)
    
    def forward(self, x, y):
        if self.boolean_type == self.UNION:
            self.cache_values = x + y - x * y
        elif self.boolean_type == self.INTERSECTION:
            self.cache_values = x * y
        elif self.boolean_type ==  self.LEFT_DIFF_RIGHT:
            self.cache_values = x * (1-y)
        elif self.boolean_type == self.RIGHT_DIFF_LEFT:
            self.cache_values = (1-x) * y
        return self.cache_values
        
    def get_boolean_type_str(self):
        if self.boolean_type == self.UNION:
            return "union"
        elif self.boolean_type == self.INTERSECTION:
            return "intersection"
        elif self.boolean_type == self.LEFT_DIFF_RIGHT:
            return "left-right"
        elif self.boolean_type == self.RIGHT_DIFF_LEFT:
            return "right-left"
        
    def negate_parameters(self):
        self.boolean_type = -self.boolean_type

    def twin(self):
        """
        return the twin node (the other node that shares the same parent) of this node
        """
        if self.parent is None: # if this is root
            return None
        if self.parent.left == self:
            return self.parent.right
        elif self.parent.right == self:
            return self.parent.left
        
    def reset_ID(self):
        self.ID = uuid.uuid4()