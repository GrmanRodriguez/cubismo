import numpy as np
import torch
from PIL import Image
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

class ModelWrappedForODESolver(ModelWrapper):
     def forward(self, x: torch.Tensor, t: torch.Tensor, y):
        return self.model(x, torch.full(x.shape[:1], fill_value=t, device="cuda"), y.to("cuda"))

class CFMPolicy:
    def __init__(
        self,
        model,
        output_resolution=(3,218,178),
    ):
        self.__model = model.cuda()
        self.__wrapped_model = ModelWrappedForODESolver(self.__model).cuda()
        self.__resolution = output_resolution
        self.__path = CondOTProbPath()
        self.__ode_solver = ODESolver(self.__wrapped_model)

    def train(self):
        self.__model.train()

    def eval(self):
        self.__model.eval()

    def zero_grad(self):
        self.__model.zero_grad()

    def weights(self):
        return self.__model.state_dict()

    def parameters(self):
        return self.__model.parameters()
    
    def compute_loss(
        self,
        images,
        labels,
    ):
        x_0 = torch.randn_like(images).cuda()
        t = torch.rand(images.shape[0]).cuda()

        path_sample = self.__path.sample(t=t,x_0=x_0,x_1=images)

        optimal_flow = path_sample.dx_t
        predicted_flow = self.__model(path_sample.x_t, path_sample.t, labels)

        return (predicted_flow - optimal_flow).square().mean()
    
    def generate_images(
        self,
        labels,
    ):
        labels_num = [label.value for label in labels]
        input_size = [len(labels_num)]
        input_size.extend(self.__resolution)
        ode_solution = self.__run_flow(
            torch.randn(input_size).cuda(),
            0,
            1,
            torch.tensor(labels_num).cuda()
        )
        ode_solution_list =  list(torch.unbind(ode_solution, dim=0))
        images = [solution.squeeze(0).permute(1,2,0).cpu().numpy() for solution in ode_solution_list]
        images = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
        return images
    
    
    
    @torch.no_grad()
    def __run_flow(
        self,
        starting_point,
        t_initial,
        t_final,
        labels
    ):
        times = torch.linspace(t_initial, t_final, 10)
        return self.__ode_solver.sample(
            x_init=starting_point,
            method="euler",
            time_grid=times,
            step_size=0.1,
            y=labels,
        )