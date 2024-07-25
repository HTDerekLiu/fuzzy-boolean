import torch

class csg_net(torch.nn.Module):
    def __init__(self, depth, max_temperature=1000, boolean_frequency=10):
        super().__init__()

        self.depth = depth
        self.num_primitives = 2**depth
        self.dim = 3
        
        # boolean parameterization
        self.max_temperature = max_temperature
        self.boolean_frequency = boolean_frequency

        self.num_params_per_primitive = 10
        self.primitive_params = torch.nn.Parameter(torch.FloatTensor(self.num_primitives, self.num_params_per_primitive).uniform_(-0.5, 0.5))
                
        # get network for boolean parameters
        self.num_boolean = 2**depth - 1
        self.num_params_per_boolean = 4
        self.boolean_params = torch.nn.Parameter(torch.FloatTensor(self.num_boolean, self.num_params_per_boolean).uniform_(-0.5, 0.5))
        
        # get sharpness parameters
        self.sharpness_params = torch.nn.Parameter(torch.FloatTensor(self.num_primitives).uniform_(0.5, 1.5))
        
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
        z = p[:,:,[2]]

        # get quadric parameters
        q0 = self.primitive_params[:,[0]].T.unsqueeze(0) # 1x1xm
        q1 = self.primitive_params[:,[1]].T.unsqueeze(0)
        q2 = self.primitive_params[:,[2]].T.unsqueeze(0)
        q3 = self.primitive_params[:,[3]].T.unsqueeze(0)
        q4 = self.primitive_params[:,[4]].T.unsqueeze(0)
        q5 = self.primitive_params[:,[5]].T.unsqueeze(0)
        q6 = self.primitive_params[:,[6]].T.unsqueeze(0)
        q7 = self.primitive_params[:,[7]].T.unsqueeze(0)
        q8 = self.primitive_params[:,[8]].T.unsqueeze(0)
        q9 = self.primitive_params[:,[9]].T.unsqueeze(0)

        # get quadric parameters
        out = q0*x**2 + q1*y**2 + q2*z**2 + q3*x*y + q4*y*z + q5*z*x + q6*x + q7*y + q8*z + q9
        return out

    def boolean_forward(self, boolean_params, x, y):
        """
        Inputs:
        bool_params: m x 4 real number of boolean parameters 
        x: b x p x m tensor of soft sign values
        y: b x p x m tensor of soft sign values
        """
        boolean_weights = torch.nn.functional.softmax(torch.sin(self.boolean_frequency * boolean_params) * self.max_temperature, dim=1) # m x 4
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
        primitive_levelset = self.primitive_forward(p) # b x p x m
        primitive_levelset = self.sharpness_params.unsqueeze(0).unsqueeze(0) * primitive_levelset
        s = self.activation(primitive_levelset) 

        curIdx = 0
        for ii in range(self.depth):
            s_left = s[:, :, ::2] # b, p, m/2
            s_right = s[:, :, 1::2] 

            num_boolean_nodes = s_right.shape[2]
            boolean_params_ii = self.boolean_params[curIdx:curIdx+num_boolean_nodes, :] # m x 4
            curIdx = curIdx + num_boolean_nodes

            s = self.boolean_forward(boolean_params_ii, s_left, s_right) # b, p, m/2
        return s.squeeze(2)