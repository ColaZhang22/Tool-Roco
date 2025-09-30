
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768))
display.start()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
import os
import pickle
import json
import numpy as np
import logging
from datetime import datetime
from glob import glob
from natsort import natsorted
from copy import deepcopy
import argparse
from typing import List, Tuple, Dict, Union, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt

from rocobench.envs import SortOneBlockTask, CabinetTask, MoveRopeTask, SweepTask, MakeSandwichTask, PackGroceryTask, MujocoSimEnv, SimRobot, visualize_voxel_scene
from rocobench import PlannedPathPolicy, LLMPathPlan, MultiArmRRT
from prompting import LLMResponseParser, FeedbackManager, DialogPrompter, CentralizedLLMResponseParser ,CentralziedPrompter,SingleThreadPrompter, save_episode_html
from prompting import DecentralziedPrompter, DecentralizedLLMResponseParser
from prompting import Selforganization_Prompter, SelfOrganizationLLMResponseParser
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# print out logging.info
logging.basicConfig(level=logging.INFO)
logging.root.setLevel(logging.INFO)

TASK_NAME_MAP = {
    "sort": SortOneBlockTask,
    "cabinet": CabinetTask,
    "rope": MoveRopeTask,
    "sweep": SweepTask,
    "sandwich": MakeSandwichTask,
    "pack": PackGroceryTask,
}

class LLMRunner:
    def __init__(
        self,
        env: MujocoSimEnv,
        robots: Dict[str, SimRobot],
        max_runner_steps: int = 50,
        video_format: str = "mp4",
        num_runs: int = 1,
        verbose: bool =False,
        np_seed: int = 0,
        start_seed: int = 0,
        run_name: str = "run",
        data_dir: str = "data",
        overwrite: bool = False,
        llm_output_mode="action_only", # "action_only" or "action_and_path"
        llm_comm_mode="chat",
        llm_num_replans=1,
        give_env_feedback=True,
        skip_display=False,
        policy_kwargs: Dict[str, Any] = dict(control_freq=50),
        direct_waypoints: int = 0,
        max_failed_waypoints: int = 0,
        debug_mode: bool = False,
        split_parsed_plans: bool = False,
        use_history: bool = False,
        use_feedback: bool = False,
        temperature: float = 0.0,
        llm_source: str = "gpt4",
        ):
        self.env = env
        self.env.reset()
        self.robots = robots
        self.robot_agent_names = list(robots.keys()) # ['Alice', etc.]
        self.data_dir = data_dir
        self.run_name = run_name
        run_dir = os.path.join(self.data_dir, self.run_name)
        os.makedirs(run_dir, exist_ok=overwrite)
        self.run_dir = run_dir
        self.verbose = verbose
        self.np_seed = np_seed
        self.start_seed = start_seed
        self.num_runs = num_runs
        self.overwrite = overwrite
        self.direct_waypoints = direct_waypoints
        self.max_failed_waypoints = max_failed_waypoints
        self.max_runner_steps = max_runner_steps
        self.give_env_feedback = give_env_feedback
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.model_path = "/home/ningjiahong/LLM/opt-13b"
        self.device = "auto"
        self.open_model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=torch.float16, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.llm_output_mode = llm_output_mode
        self.debug_mode = debug_mode # useful for debug


        self.llm_num_replans = llm_num_replans
        self.llm_comm_mode = llm_comm_mode
        self.response_keywords = ['NAME', 'ACTION']
        if llm_output_mode == "action_and_path":
            self.response_keywords.append('PATH')
        self.planner = MultiArmRRT(
            self.env.physics,
            robots=robots,
            graspable_object_names=self.env.get_graspable_objects(),
            allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
        )
        self.policy_kwargs = policy_kwargs
        self.video_format = video_format
        self.skip_display = skip_display
        self.split_parsed_plans = split_parsed_plans
        self.temperature = temperature
        if self.llm_comm_mode == "decentralized":
            self.parser = DecentralizedLLMResponseParser(
                self.env,
                llm_output_mode,
                self.env.robot_name_map,
                self.response_keywords,
                self.direct_waypoints,
                use_prepick=self.env.use_prepick,
                use_preplace=self.env.use_preplace, # NOTE: should be custom defined in each task env
                split_parsed_plans=False, # self.split_parsed_plans,
            )
        elif self.llm_comm_mode == "centralized":
            self.parser = CentralizedLLMResponseParser(
                self.env,
                llm_output_mode,
                self.env.robot_name_map,
                self.response_keywords,
                self.direct_waypoints,
                use_prepick=self.env.use_prepick,
                use_preplace=self.env.use_preplace, # NOTE: should be custom defined in each task env
                split_parsed_plans=False, # self.split_parsed_plans,
            )
        elif self.llm_comm_mode == "auto_organization":
            self.parser = SelfOrganizationLLMResponseParser(
                self.env,
                llm_output_mode,
                self.env.robot_name_map,
                self.response_keywords,
                self.direct_waypoints,
                use_prepick=self.env.use_prepick,
                use_preplace=self.env.use_preplace, # NOTE: should be custom defined in each task env
                split_parsed_plans=False, # self.split_parsed_plans,
            )

        self.feedback_manager = FeedbackManager(
            env=self.env,
            planner=self.planner,
            llm_output_mode=self.llm_output_mode,
            robot_name_map=self.env.robot_name_map,
            step_std_threshold=self.env.waypoint_std_threshold,
            max_failed_waypoints=self.max_failed_waypoints,)
        
        if llm_comm_mode == "decentralized":
            self.prompter = DecentralziedPrompter(
            env=self.env,
            parser=self.parser,
            feedback_manager=self.feedback_manager,
            max_tokens=1024,
            debug_mode=self.debug_mode,
            robot_name_map=self.env.robot_name_map,
            max_calls_per_round=10,
            use_waypoints=(self.llm_output_mode == "action_and_path"),
            use_history=self.use_history,
            use_feedback=self.use_feedback,
            num_replans=self.llm_num_replans,
            temperature=self.temperature,
            llm_source=llm_source,
            open_model = self.open_model,
            tokenizer = self.tokenizer 
            )

        elif llm_comm_mode == "centralized":
            self.prompter = CentralziedPrompter(
                env=self.env,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=1024,
                debug_mode=self.debug_mode,
                robot_name_map=self.env.robot_name_map,
                max_calls_per_round=10,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                use_feedback=self.use_feedback,
                num_replans=self.llm_num_replans,
                temperature=self.temperature,
                llm_source=llm_source,
            )
        elif llm_comm_mode == "auto_organization":
            self.prompter = Selforganization_Prompter(
                env=self.env,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=1024,
                debug_mode=self.debug_mode,
                robot_name_map=self.env.robot_name_map,
                max_calls_per_round=10,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                use_feedback=self.use_feedback,
                num_replans=self.llm_num_replans,
                temperature=self.temperature,
                llm_source=llm_source,
            )


    def display_plan(self, plan: LLMPathPlan, save_name = "vis_plan", save_dir = None):
        """ Display the plan in the open3d viewer """ 
        env = deepcopy(self.env)
        env.physics.data.qpos[:] = self.env.physics.data.qpos[:].copy()
        env.physics.forward()
        env.render_point_cloud = True
        obs = env.get_obs()
        path_ls = plan.path_3d_list
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{save_name}.jpg")
        visualize_voxel_scene(
            obs.scene,
            path_pts=path_ls,
            save_img=(save_dir is not None),
            img_path=save_path
            )
    
    def save_intermediate_env_state(self, step_dir, env):
        data_fname = f"{step_dir}/env_init.pkl"
        sim_data = env.save_intermediate_state()
        f = open(data_fname, "wb")
        pickle.dump(sim_data, f)

    def save_intermedinate_plan(self, step_dir, current_llm_plan):
        for i, plan in enumerate(current_llm_plan):
                save_fname = os.path.join(step_dir, f"llm_plan_{i}.pkl")
                with open(save_fname, "wb") as f:
                    pickle.dump(plan, f)

    def save_intermedinate_rrt_plan_action(self, step_dir, policy, i):
        plan_fname = os.path.join(step_dir, f"rrt_plan_{i}.pkl")
        rrt_plans = policy.rrt_plan_results
        with open(plan_fname, "wb") as f:
            pickle.dump(rrt_plans, f)
        actions_fname = f"{step_dir}/actions_{i}.pkl"
        with open(actions_fname, "wb") as f:
            pickle.dump(policy.action_buffer, f)

    def initial_save_dir(self, save_dir, step):
        step_dir = os.path.join(save_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=self.overwrite)
        prompt_dir = os.path.join(step_dir, "prompts")
        os.makedirs(prompt_dir, exist_ok=self.overwrite)
        result_dir = os.path.join(step_dir,"log")
        os.makedirs(result_dir, exist_ok=self.overwrite)

        return step_dir, prompt_dir, result_dir
        
    def one_run(self, run_id: int = 0, start_step: int = 0, skip_reset = False, prev_llm_plans = [], prev_response = None, prev_actions = None):
        """ uses planner """
        self.env.seed(np_seed=run_id)
        if not skip_reset:
            self.env.reset(reload=True) # NOTE: need to do this to reset the model.eq_active vals

        env = self.env
        physics = env.physics
        success = False
        save_dir = os.path.join(self.run_dir, f"run_{run_id}")
        os.makedirs(save_dir, exist_ok=self.overwrite)

        done = False
        reward = 0
        obs = env.get_obs()

        # Initialize History, history  consists of three parts:
        # 1. state history, which is the previous state of the environment
        # 2. action history, which is the previous actions taken by the agents
        # 3. feedback history, which is the feedback from the environment after executing the actions
        history = {}
        for step in range(start_step, start_step + self.max_runner_steps):
            # Turn level 每一次重制system prompt
            task_name = self.env.env_name
            system_prompt = self.prompter.compose_task_prompt(task_name)
            step_dir, prompt_dir, result_dir = self.initial_save_dir(save_dir, step)
            # Save current env
            self.save_intermediate_env_state(step_dir, env)

            # Start from middle point
            if step == start_step and (len(prev_llm_plans) > 0 or prev_response is not None):
                ready_to_execute = 1
                current_llm_plan = prev_llm_plans
                history =self.prompter.load_prompt_response(prompt_dir)
                log =[]
            else:
                ready_to_execute, current_llm_plan, history, result_log, replan_idx = self.prompter.prompt_one_round(system_prompt, obs, save_path=prompt_dir, result_dir=result_dir, step=step, history=history)
                if not ready_to_execute or current_llm_plan is None:
                    print(f"Run {run_id}: Step {step} failed to get a plan from LLM. Move on to next step.")
                    continue
                # show the path plans in figure, まだしてないです。
                if not self.skip_display:
                    for i, plan in enumerate(current_llm_plan):
                        self.display_plan(plan, save_name=f"vis_llm_plan_{i}", save_dir=step_dir)

            # save current plan
            self.save_intermedinate_plan(step_dir,current_llm_plan)
            logging.info(f"Step: {step} LLM plan parsed, begin RRT planning ")
            # try execute this plan, if one of the plan failed, rewind the env to before the first plan was executed!
            rewind_env = False
            # we have three kinds of plans: centralized multi-agent, decentralized multi-agent and autonomous single-agent
            # for centralized multi-agent plans, the length of the plan should be the one. Each agent calls one tools.
            for i, plan in enumerate(current_llm_plan):

                policy = PlannedPathPolicy(
                    physics=env.physics,
                    robots=self.robots,
                    path_plan=plan,
                    graspable_object_names=self.env.get_graspable_objects(),
                    allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                    plan_splitted=self.split_parsed_plans,
                    **self.policy_kwargs,
                )

                num_sim_steps = 0
                if prev_actions is not None:
                    for sim_action in prev_actions:
                        # env.physics.model.eq_active[52:] = 0
                        obs, reward, done, info = env.step(sim_action, verbose=False)
                        num_sim_steps += 1
                else:
                    plan_success, reason = policy.plan(env)               
                    logging.info(f"Stesp: {step} Plan success: {plan_success}, reason: {reason}")
                    if plan_success:
                        for agent_name in self.prompter.robot_agent_names:
                            history[agent_name].append({"role": "system", "content": f"[REWARD] At step{step}, Successfully parse the plan and successfully execute the planned path."})
                        
                        # save the log for evaluation.
                        result_log["Execution"] = 1
                        result_log["Reason"] = "Success"
                        self.prompter.save_result(result_dir, step, replan_idx, result_log)
                        logging.info(f"Execute the plan for {len(policy.action_buffer)} steps")
                        self.save_intermedinate_rrt_plan_action(step_dir, policy, i)

                        while not policy.plan_exhausted:
                            self.env.physics.forward() 
                            sim_action = policy.act(obs, env.physics)
                            obs, reward, done, info = env.step(sim_action, verbose=False)
                            num_sim_steps += 1
                    
                    else:
                        for agent_name in self.prompter.robot_agent_names:
                            history[agent_name].append({"role": "system", "content": f"[REWARD] At step{step}, Plan failed to execute:{reason}, you need to replan a different path parameters and avoid Collision with any objective."})
                        # save the log for evaluation
                        result_log["Execution"] = 0
                        result_log["Reason"] = reason
                        self.prompter.save_result(result_dir, step, replan_idx, result_log)
                        # log.append({"role":"ENV","Tools Calling":1, "Valid Parameter":1,"Execution": 0})
                
                #Save demo video
                if num_sim_steps > 0:
                    # env.physics.model.vis.geomgroup[:] = 1 
                    # env.physics.forward() 
                    env.export_render_to_video(f"{step_dir}/execute.mp4", out_type=self.video_format,  fps=20)
                    logging.info(f'Plans all executed! Video sample saved to {step_dir}/execute.mp4')
                else:
                    if start_step == 0:
                        rewind_env = False
                    else:
                        rewind_env = True
                    logging.info(f"Plan {i} failed to execute. Rollback to last state")
                    break

            if rewind_env:
                print("Rewinding the environment to before the first plan was executed.")
                env.load_saved_state(sim_data)
            else:
                sim_data = env.save_intermediate_state()

            data_fname = f"{step_dir}/env_end.pkl"
            with open(data_fname, "wb") as f:
                pickle.dump(sim_data, f)

            self.prompter.post_execute_update(
                obs_desp="", # TODO
                execute_success=(not rewind_env),
                parsed_plan=current_llm_plan[0].get_action_desp())

            if done:
                break

        success = reward > 0
        json.dump(
            dict(step=step, success=success),
            open(f"{save_dir}/steps{step}_success_{success}.json", "w"),
        )
        print("Run finished after {} timesteps".format(step))
        self.prompter.post_episode_update()
        save_episode_html(
            save_dir,
            html_fname=f"steps{step}_success_{success}",
            video_fname="execute.mp4",
            sender_keys=["Alice", "Bob", "Chad", "Dave", "Planner", "Feedback", "Action"],
            )
        print(f"Episode html saved to {save_dir}")


    def run(self, args):
        start_id = 0 if args.start_id == -1 else args.start_id
        if args.cont:
            logging.info("Continuing from previous run")
            load_run = glob(os.path.join(self.data_dir, args.load_run_name, f"run_{args.load_run_id}"))
            if len(load_run) == 0:
                raise ValueError(f"Cannot find run {args.load_run_id} in {args.load_run_name}")
                exit()
            load_run = load_run[0]
            # find the latest steps
            # TODO: zhangke need to re-write load part.
            step_dirs = natsorted(glob(os.path.join(load_run, "step_*")))
            if len(step_dirs) == 0:
                raise ValueError(f"Cannot find any steps in {load_run}")
                exit()
            latest_step = step_dirs[-1]
            env_init_fname = os.path.join(latest_step, "env_init.pkl")
            with open(env_init_fname, "rb") as f:
                saved_data = pickle.load(f)
                self.env.load_saved_state(saved_data)

          
            print(f"==== Loading back Run {args.load_run_id} ====")
            next_step = int(latest_step.split("/")[-1].split("_")[-1])
            prev_llm_plans = []
            prev_plans = natsorted(glob(os.path.join(latest_step, "llm_plan_*pkl")))
            if len(prev_plans) > 0:
                prev_llm_plans = [pickle.load(open(fname, "rb")) for fname in prev_plans]

            prev_response = None
            prev_responses = natsorted(glob(os.path.join(latest_step, "prompts", "*response.json")))
            if len(prev_responses) > 0:
                prev_response = json.load(open(prev_responses[-1], "rb"))

            prev_actions = None
            fname = os.path.join(latest_step, "actions.pkl")
            if os.path.exists(fname):
                prev_actions = pickle.load(open(fname, "rb"))

            self.one_run(
                args.load_run_id,
                start_step=next_step,
                skip_reset=True,
                prev_llm_plans=prev_llm_plans,
                prev_response=prev_response,
                prev_actions=prev_actions
                )
            start_id = args.load_run_id + 1
        existing_runs = glob(os.path.join(self.data_dir, args.run_name, "run_*"))
        if args.start_id == -1 and len(existing_runs) > 0:
            existing_run_ids = [int(run.split("_")[-1]) for run in existing_runs]
            start_id = max(existing_run_ids) + 1
        for run_id in range(start_id, start_id + self.num_runs):
            print(f"==== Run {run_id} starts ====")
            self.one_run(run_id)

def main(args):
    assert args.task in TASK_NAME_MAP.keys(), f"Task {args.task} not supported"
    env_cl = TASK_NAME_MAP[args.task]
    if args.task == 'rope':
        args.output_mode = 'action_and_path'
        args.split_parsed_plans = True
        logging.warning("MoveRopeTask requires split parsed plans\n")

        args.control_freq = 20
        args.max_failed_waypoints = 0
        logging.warning("MopeRope requires max failed waypoints 0\n")
        if not args.no_feedback:
            args.tstep = 5
            logging.warning("MoveRope needs only 5 tsteps\n")

    # elif args.task == 'pack':
    #     args.output_mode = 'action_and_path'
    #     args.control_freq = 10
    #     args.split_parsed_plans = True
    #     args.max_failed_waypoints = 0
    #     args.direct_waypoints = 3
    #     logging.warning("PackGroceryTask requires split parsed plans, and no failed waypoints, no direct waypoints\n")

    render_freq = 600
    if args.control_freq == 15:
        render_freq = 1200
    elif args.control_freq == 10:
        render_freq = 2000
    elif args.control_freq == 5:
        render_freq = 3000
    env = env_cl(
        render_freq=render_freq,
        image_hw=(400,400),
        sim_forward_steps=300,
        error_freq=30,
        error_threshold=1e-5,
        randomize_init=True,
        render_point_cloud=0,
        render_cameras=["face_panda","face_ur5e","teaser",],
        one_obj_each=True,
    )
    robots = env.get_sim_robots()
    if args.no_feedback:
        assert args.num_replans == 1, "no feedback mode requires num_replans=1 but longer -tsteps"


    # save args into a json file
    args_dict = vars(args)
    args_dict["env"] = env.__class__.__name__
    timestamp = datetime.now().strftime("%Y%m_%H%M")
    fname = os.path.join(args.data_dir, args.run_name, f"args_{timestamp}.json")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    json.dump(args_dict, open(fname, "w"), indent=2)
    runner = LLMRunner(
        env=env,
        data_dir=args.data_dir,
        robots=robots,
        max_runner_steps=args.tsteps,
        num_runs=args.num_runs,
        run_name=args.run_name,
        overwrite=True,
        skip_display=args.skip_display,
        llm_output_mode=args.output_mode, # "action_only" or "action_and_path"
        llm_comm_mode=args.comm_mode, # "chat" or "plan"
        llm_num_replans=args.num_replans,
        policy_kwargs=dict(
            control_freq=args.control_freq,
            use_weld=args.use_weld,
            skip_direct_path=0,
            skip_smooth_path=0,
            check_relative_pose=args.rel_pose,
        ),
        direct_waypoints=args.direct_waypoints,
        max_failed_waypoints=args.max_failed_waypoints,
        debug_mode=args.debug_mode,
        split_parsed_plans=args.split_parsed_plans,
        use_history=(not args.no_history),
        use_feedback=(not args.no_feedback),
        temperature=args.temperature,
        llm_source=args.llm_source,
    )
    runner.run(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="Experiment")
    parser.add_argument("--temperature", "-temp", type=float, default=0)
    parser.add_argument("--start_id", "-sid", type=int, default=-1)
    parser.add_argument("--num_runs", '-nruns', type=int, default=2)
    parser.add_argument("--run_name", "-rn", type=str, default="cabinet_gpt-5_dec")
    parser.add_argument("--tsteps", "-t", type=int, default=10)
    parser.add_argument("--task", type=str, default="cabinet")
    parser.add_argument("--output_mode", type=str, default="action_and_path", choices=["action_only", "action_and_path"])
    parser.add_argument("--comm_mode", type=str, default="decentralized", choices=["decentralized", "auto_organization"])
    parser.add_argument("--control_freq", "-cf", type=int, default=15)
    parser.add_argument("--skip_display", "-sd", action="store_false")
    parser.add_argument("--direct_waypoints", "-dw", type=int, default=3)
    parser.add_argument("--num_replans", "-nr", type=int, default=5)
    parser.add_argument("--cont", "-c", action="store_true")
    parser.add_argument("--load_run_name", "-lr", type=str, default="cabinet")
    parser.add_argument("--load_run_id", "-ld", type=int, default=0)
    parser.add_argument("--max_failed_waypoints", "-max", type=int, default=1)
    parser.add_argument("--debug_mode", "-i", action="store_true")
    parser.add_argument("--use_weld", "-w", type=int, default=1)
    parser.add_argument("--rel_pose", "-rp", action="store_true")
    parser.add_argument("--split_parsed_plans", "-sp", action="store_true")
    parser.add_argument("--no_history", "-nh", action="store_true")
    parser.add_argument("--no_feedback", "-nf", action="store_true")
    parser.add_argument("--llm_source", "-llm", type=str, default="gpt-4")
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    main(args)
