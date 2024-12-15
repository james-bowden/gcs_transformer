import numpy as np
import pickle
from tqdm import tqdm

from pydrake.geometry import StartMeshcat
from pydrake.geometry import MeshcatVisualizer, Rgba, StartMeshcat
from pydrake.geometry.optimization import Point
from pydrake.planning import GcsTrajectoryOptimization

from underactuated import running_as_notebook
from underactuated.uav_environment import (
    CONVEX_GCS_OPTION,
    NONLINEAR_GCS_OPTION,
    UavEnvironment,
)

# from underactuated.uav_environment import *
from custom_uav_env import *

def reconstruct_trajectory_from_solution_path(edges, rounded_result, vertex_to_subgraph, gcs):
    bezier_curves = []
    bezier_curves_params = []
    for e in edges:
        phi_inv = 1. / rounded_result.GetSolution(e.phi())
        num_control_points = vertex_to_subgraph[e.u()].order() + 1
        # edge_path_points = 
        # print(rounded_result.GetSolution(e.xu()).shape, num_control_points)
        # total += (rounded_result.GetSolution(e.xu())[-1])
        xu_result = rounded_result.GetSolution(e.xu())
        h = phi_inv * xu_result[-1]
        edge_path_points = xu_result[:-1].reshape(num_control_points, gcs.num_positions()).T

        start_time = 0 if len(bezier_curves) == 0 else bezier_curves[-1].end_time()

        if num_control_points > 1 and h < 1e-10:
            # TODO
            pass
        elif not (num_control_points == 1): # TODO: enforce && vertex_to_subgraph_[&e->u()]->h_min_ == 0))
            bc = BezierCurve(start_time, start_time + h, edge_path_points)
            bezier_curves.append(bc)
            bezier_curves_params.append((start_time, start_time + h, edge_path_points))

    return CompositeTrajectory(bezier_curves), bezier_curves_params
        
env_shape = (6, 6)
x = np.arange(1, env_shape[0]-1)
y = np.arange(1, env_shape[1]-1)
xx, yy = np.meshgrid(x, y)
possible_goals = np.stack([xx, yy]).reshape(2, -1).T * CELL_SIZE
total = 0

for seed in tqdm(range(100000)):
    # goal_inds = np.random.choice(len(possible_goals), size=16, replace=False)
    goal_inds = np.arange(len(possible_goals))
    
    # print("Starting seed:", seed)
    for goal_ind in goal_inds:
        goal_xy = possible_goals[goal_ind]
        DEFAULT_GOAL = (*goal_xy, 0.5)
        uav_env = UavEnvironment(seed=seed, environment_shape=env_shape, DEFAULT_GOAL=DEFAULT_GOAL)
        regions, edges_between_regions = uav_env.compile()

        # The maximum velocity limits for the skydio2
        # were obtained from their website.
        qDt_max = 16.0
        # While the maximum acceleration are not publicly available, we assume
        # an estimated thrust to weight ratio of something slightly greater than 2.
        qDDt_max = 10.0

        gcs = GcsTrajectoryOptimization(3)
        main = gcs.AddRegions(regions, edges_between_regions, order=6, h_min=0, h_max=20)
        source = gcs.AddRegions(
            [Point(uav_env.DEFAULT_START)], order=0, h_min=0, h_max=0, name="source"
        )
        target = gcs.AddRegions(
            [Point(uav_env.DEFAULT_GOAL)], order=0, h_min=0, h_max=0, name="target"
        )
        source_to_main = gcs.AddEdges(source, main)
        main_to_target = gcs.AddEdges(main, target)

        source_to_main.AddZeroDerivativeConstraints(1)
        main_to_target.AddZeroDerivativeConstraints(1)
        source_to_main.AddZeroDerivativeConstraints(2)
        main_to_target.AddZeroDerivativeConstraints(2)

        gcs.AddVelocityBounds(3 * [-qDt_max], 3 * [qDt_max])

        gcs.AddTimeCost()
        gcs.AddPathLengthCost()


        CONVEX_GCS_OPTION.preprocessing = True
        CONVEX_GCS_OPTION.convex_relaxation = True


        CONVEX_GCS_OPTION.max_rounded_paths = 0
        CONVEX_GCS_OPTION.max_rounding_trials = 0

        result = gcs.graph_of_convex_sets().SolveShortestPath(source.Vertices()[0], target.Vertices()[0], CONVEX_GCS_OPTION)
        
        if not result.is_success():
            continue

        CONVEX_GCS_OPTION.max_rounded_paths = 300
        CONVEX_GCS_OPTION.max_rounding_trials = 300

        paths = gcs.graph_of_convex_sets().SamplePaths(source.Vertices()[0], target.Vertices()[0], result, CONVEX_GCS_OPTION)

        vertex_to_subgraph = {}
        for sg in gcs.GetSubgraphs():
            for v in sg.Vertices():
                vertex_to_subgraph[v] = sg

        best_rounded_result = None
        path_edges_list = []
        traj_list = []

        path_curve_params = []
        path_returns = []
        path_times = []

        for path in paths:
            rounded_result = gcs.graph_of_convex_sets().SolveConvexRestriction(path, CONVEX_GCS_OPTION, result)
            if rounded_result.is_success():
                # print(f"Success with cost {rounded_result.get_optimal_cost()}")
                path_edges = gcs.graph_of_convex_sets().GetSolutionPath(source.Vertices()[0], target.Vertices()[0], rounded_result, 1.)

                traj, bz_curve_params = reconstruct_trajectory_from_solution_path(path_edges, rounded_result, vertex_to_subgraph, gcs)
                
                path_edges_list.append(path_edges)
                traj_list.append(traj)

                path_curve_params.append(bz_curve_params)
                path_returns.append(rounded_result.get_optimal_cost())
                path_times.append(traj.end_time() - traj.start_time())

                # print(
                #     f"Total trajectory time: {round(traj.end_time() - traj.start_time(), 3)} seconds."
                # )

            else:
                # print("Failed")
                continue

        data_dict = {
            "metadata": {
                "seed": seed,
                "goal_ind": goal_ind,
                "goal": np.array(DEFAULT_GOAL),
                "map": uav_env.map_array,
            }, 
            "trajs": {
                "curve_params": path_curve_params,
                "returns": path_returns,
                "times": path_times
            }
        }

        with open(f"/mnt/data_dir/{seed}_{goal_ind}.pkl", "wb") as f:
            pickle.dump(data_dict, f)

        total += len(traj_list)
        print(total)