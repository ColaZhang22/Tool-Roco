import os 
import json
import pickle 
from openai import OpenAI
import numpy as np
from datetime import datetime
from os.path import join
from typing import List, Tuple, Dict, Union, Optional, Any
from glob import glob
import random
from rocobench.subtask_plan import LLMPathPlan
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.envs import MujocoSimEnv, EnvState 
from prompting.feedback import FeedbackManager
from prompting.parser import LLMResponseParser

# assert os.path.exists("openai_key.json"), "Please put your OpenAI API key in a string in robot-collab/openai_key.json"
# OPENAI_KEY = str(json.load(open("openai_key.json")))
# openai.api_key = OPENAI_KEY

PATH_PLAN_INSTRUCTION="""
[Path Plan Instruction]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""

class SelforganizationCentral_Prompter:
    """
    Each round contains multiple prompts, query LLM once per each agent 
    """
    def __init__(
        self,
        env: MujocoSimEnv,
        parser: LLMResponseParser,
        feedback_manager: FeedbackManager, 
        max_tokens: int = 512,
        debug_mode: bool = False,
        use_waypoints: bool = False,
        robot_name_map: Dict[str, str] = {"panda": "Bob"},
        num_replans: int = 3, 
        max_calls_per_round: int = 10,
        use_history: bool = True,  
        use_feedback: bool = True,
        temperature: float = 0,
        llm_source: str = "gpt-4",
        open_model = None,
        tokenizer = None 
    ):
        self.open_model = open_model
        self.tokenizer = tokenizer 
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.robot_name_map = robot_name_map
        self.robot_agent_names = list(robot_name_map.values())
        self.num_replans = num_replans
        self.env = env
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.round_history = []
        self.failed_plans = [] 
        self.latest_chat_history = []
        self.max_calls_per_round = max_calls_per_round 
        self.temperature = temperature
        self.llm_source = llm_source
        self.active_agents = [random.choice(self.robot_agent_names)]
        
        # assert llm_source in ["gpt-4", "gpt-3.5-turbo", "claude"], f"llm_source must be one of [gpt4, gpt-3.5-turbo, claude], got {llm_source}"

    def get_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}\n"
        ret += f"== Current Round ==\n"
        return ret
    
    def compose_task_prompt(self,task_name) -> List[str]:
        """
        Compose the task prompt for the agents.
        This is a static prompt that describes the task and the agents.
        """
        system_prompt_pool =  json.load(open(f"prompt_template/task/{task_name}/system_prompt.json","r", encoding="utf-8"))

        Task_desc = system_prompt_pool["TASK_CONTEXT"]
        Cooperation_decs = system_prompt_pool["TASK_CHAT_PROMPT"]

        system_prompt = [{"role": "system", "content": Task_desc + Cooperation_decs},]
        return system_prompt

    def compose_next_prompt(self,system_prompt, centralized_response, parsed_str):
        """
        Compose the next prompt for the agents.
        """
        system_prompt = self.compose_task_prompt(self.env.env_name)
        system_prompt.append({"role": "assistant", "content": "Last turn, your answer is:" + centralized_response})
        if not (parsed_str is None or parsed_str == ""):
            system_prompt.append({"role": "system", "content": f"Parse this tool calling response failure. The reason is {parsed_str}"})
        return system_prompt
    

    def save_result(self, result_dir, replan_idx, n_calls, result):
        fname = f'{result_dir}/replan{replan_idx}_call{n_calls}_centralized_agent_result.json'
        json.dump(result, open(fname, 'w'))  

    def prompt_one_round(self, system_prompt, obs: EnvState, save_path: str = "", result_dir: str="", step: int = 0, history: List=[]) -> Tuple[bool, Optional[List[LLMPathPlan]], List[Dict]]: 


        for i in range(self.num_replans):
            # extend previous history
            if history !=[]:
                system_prompt.extend(history)
            print(f"Active Agent list--------------------------------{self.active_agents}")
            system_prompt, centralized_response, current_state, prompt_tokens, completion_tokens = self.prompt_one_centralized_round(system_prompt, obs, replan_idx=i,save_path=save_path, step=step)
            parse_succ, parsed_str, llm_plans = self.parser.central_parse(obs, centralized_response, self.active_agents) 
            
            history.append({"role": "system", "content": f"[STATE] At step{step}, replan_number{i} the agent states are: {current_state}"})
            history.append({"role": "assistant", "content": f"[ACTION] At step{step}, replan_number{i}, the agent actions are: {centralized_response}"})
          
            # 判断是否parse成功
            if parse_succ:  
                ready_to_execute = True
                # 判断所有的plan是否能在环境中执行, 主要检测碰撞，可达性，规划路径
                if parsed_str == "Interior Tool":
                    result_log = {"Env":f"{self.env.env_name}", "active_agents": self.active_agents,  "prompt_tokens":prompt_tokens, "completion_tokens":completion_tokens, "Step": step, "Replan": i, "Prompt":system_prompt, "Response":centralized_response ,"Tools Calling":1, "Valid Parameter":1,"Execution": 1, "Reason":"Interiror Tool Success"}
                    system_prompt = self.compose_task_prompt(self.env.env_name)
                    self.save_result(result_dir, step, i, result_log)
                    continue
                for j, llm_plan in enumerate(llm_plans): 
                    ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)        
                    if not ready_to_execute:
                        # 如果feedback失败，记录失败的原因，跳出循环
                        history.append({"role": "system", "content": f"[REWARD] At step{step}, replan_number{i} , your Plan is successfully parsed but environment feedback is failure, failure history: {env_feedback}"})
                        result_log = {"Env":f"{self.env.env_name}", "active_agents": self.active_agents,   "prompt_tokens":prompt_tokens, "completion_tokens":completion_tokens, "Step": step, "Replan": i, "Prompt":system_prompt, "Response":centralized_response ,"Tools Calling":1, "Valid Parameter":0,"Execution": 0, "Reason":env_feedback}
                        self.save_result(result_dir, step, i, result_log)
                        system_prompt = self.compose_task_prompt(self.env.env_name)
                        ready_to_execute = False
                        break
            else:
                # print("central_parse failure")
                # 如果parse失败，记录原因，并重新规划，
                history.append({"role": "system", "content": f"[REWARD] At step{step}, replan_number{i} , your Plan is FAIL to parsed and the reason is  {parsed_str}"})
                result_log = {"Env":f"{self.env.env_name}", "active_agents": self.active_agents, "prompt_tokens":prompt_tokens, "completion_tokens":completion_tokens, "Step": step, "Replan": i, "Prompt":system_prompt, "Response":centralized_response,"Tools Calling":0, "Valid Parameter":0,  "Execution": 0 ,"Reason":parsed_str}
                self.save_result(result_dir, step, i, result_log)
                system_prompt = self.compose_task_prompt(self.env.env_name)
                ready_to_execute = False

            if ready_to_execute: 
                # print("feedback succes")
                result_log = {"Env":f"{self.env.env_name}", "active_agents": self.active_agents, "prompt_tokens":prompt_tokens, "completion_tokens":completion_tokens, "Step": step, "Replan": i, "Prompt":system_prompt, "Response":centralized_response, "Tools Calling":1, "Valid Parameter":1,  "Execution": 0 ,"Reason":""}
                break  
            else:
                print(f"Replan {i} failed, Replan again")
        return ready_to_execute, llm_plans, history, result_log, i
    
    def compose_all_agent_state(self, obs, system_prompt: str, n_calls: int) -> List[Dict]:
        """
        Create all agents prompt.
        """
        # agent_prompt = f"You are {agent_name}, your response is:"
        # if n_calls == self.max_calls_per_round - 1:
        #         agent_prompt = f""" You are {agent_name}, this is the last call, you must end your response by incoporating all previous discussions and output the best plan via EXECUTE.  Your response is:"""
        for agent_name in self.robot_agent_names:
            # assert agent_name in self.robot_name_map, f"{agent_name} not found in robot_name_map"
            system_prompt.append({"role": "system", "content": self.env.get_agent_objective_prompt(obs, agent_name)})
        return system_prompt
    
    def compose_all_agent_tools(self, system_prompt: List[Dict]) -> List[Dict]:
        # Compose all agents tool list.
        for agent_name in self.active_agents:
            tools =  json.load(open(f"prompt_template/agent/{agent_name}/tools.json","r", encoding="utf-8"))
            system_prompt.append({"role": "system", "content": f"For {agent_name}, it has a tool list and are:" + json.dumps(tools)})
        return system_prompt

    def compose_centralized_answer_format(self, system_prompt) -> List[Dict]:
        system_prompt_pool =  json.load(open(f"prompt_template/task/{self.env.env_name}/system_prompt.json","r", encoding="utf-8"))
        Answer_format = system_prompt_pool["CENTRALIZED_ANSWER_FORMAT"]
        system_prompt.append({"role": "user", "content": Answer_format})
        return system_prompt

    def compose_selfo_centralized_answer_format(self, system_prompt) -> List[Dict]:
        system_prompt_pool =  json.load(open(f"prompt_template/task/{self.env.env_name}/system_prompt.json","r", encoding="utf-8"))
        Answer_format = system_prompt_pool["SELFORGANIZATION_CENTRALIZED_ANSWER_FORMAT"]
        system_prompt.append({"role": "user", "content": Answer_format})
        return system_prompt
    
    def compose_current_agent_pool(self, system_prompt: List[Dict]):
        system_prompt_pool =  json.load(open(f"prompt_template/task/{self.env.env_name}/system_prompt.json","r", encoding="utf-8"))
        agentgroup_format = system_prompt_pool["SELFORGANIZATION_CENTRALIZED_AGENT_TEAM_STATE"]
        inactive_agents =[]
        # detect active agents
        for id in self.robot_agent_names:
            if id not in self.active_agents:
                inactive_agents.append(id)
        agentgroup_format = agentgroup_format.format(active_robots=', '.join(self.active_agents), inactive_robots=', '.join(inactive_agents))

        system_prompt.append({"role": "user", "content": agentgroup_format})
        return system_prompt
    
    def save_prompt_response(self, save_path, replan_idx, n_calls, system_prompt):
        timestamp = datetime.now().strftime("%m%d-%H%M")
        fname = f'{save_path}/replan{replan_idx}_call{n_calls}_centralized_agent_{timestamp}.json'
        json.dump(system_prompt, open(fname, 'w'))  

    def load_prompt_response(self, save_path):
        files = glob(os.path.join(save_path, "replan*_call*_centralized_agent_*.json"))
        if not files:
            raise FileNotFoundError("No JSON files found in the directory.")
        latest_file = max(files, key=os.path.getmtime)
        with open(latest_file, "r") as f:
            latest_history = json.load(f)
        return latest_history

    def prompt_one_centralized_round(self, system_prompt, obs, replan_idx: int = 0, save_path: str ='data/', step: int = 0):
        """
        Prompt one round of centralized with all agents. Centralized means all agents are prompted in one round, and there exists one agent to plan the policy for all agent. In this case, we let llm to separately decide one action for each agent, and then execute the actions in parallel.
        """
        n_calls = 0
        # compose all agent prompt

        # current_state will be add into system_prompt, length of system_prompt add 3
        system_prompt = self.compose_all_agent_state(obs, system_prompt, n_calls)  
        # set the current state of the environment as current_state
        current_state = "".join(d["content"] for d in system_prompt[-3:-1])
        # curent_tools will be add into system_prompt, length of system_prompt add 3
        system_prompt = self.compose_current_agent_pool(system_prompt)
        # system_prompt = self.compose_decentralized_answer_format(system_prompt, agent_name)
        system_prompt = self.compose_all_agent_tools(system_prompt)  
        system_prompt = self.compose_selfo_centralized_answer_format(system_prompt)
        while n_calls < self.max_calls_per_round:
            response, prompt_tokens, completion_tokens = self.query_once(system_prompt, max_query=3,)
            system_prompt.append({"role": "assistant", "content": response})
            # save current prompt and answer
            self.save_prompt_response(save_path, replan_idx, n_calls, system_prompt)
            if response != "" or response is not None:
                break
        return system_prompt, response, current_state, prompt_tokens, completion_tokens

    def query_once(self, system_prompt,  max_query):
        response = None
        prompt_tokens = 0
        completion_tokens = 0
        usage = None   
        for n in range(max_query):
            print('querying {}th time'.format(n))

            # text = self.tokenizer.apply_chat_template(system_prompt, tokenize=False,add_generation_prompt=True,)
            # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.open_model.device)
            # generated_ids = self.open_model.generate(**model_inputs, max_new_tokens=1024, )
            # # generated_ids = self.open_model.generate(
            # #     **model_inputs,
            # #     max_new_tokens=1024,       
            # #     do_sample=True,          
            # #     temperature=0.1,         
            # #     top_p=0.9,               
            # #     repetition_penalty=1.1,   
            # #     num_return_sequences=1,   
            # #     no_repeat_ngram_size=3    
            # # )
            # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            # response = self.tokenizer.decode(output_ids, skip_special_tokens=True)  
            # print(f'prompt_token_num:{len(model_inputs.input_ids[0])}')
            try:
                response = client.chat.completions.create(model="gpt-4.1", messages=system_prompt, max_tokens=self.max_tokens, temperature=self.temperature)
                # usage = response['usage']
                # response = client.chat.completions.create(model="gpt-5", messages=system_prompt)
            except:
                print("API error, try again")
                continue
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            response = response.choices[0].message.content
            print(f'prompt_token_num:{prompt_tokens},complection_token_nume:{completion_tokens}completion_tokens ======= response ======= \n ', response)
            break
        return response, prompt_tokens, completion_tokens
    
    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success: 
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            chats = "\n".join(self.latest_chat_history)
            self.round_history.append(
                f"[Chat History]\n{chats}\n[Executed Action]\n{parsed_plan}"
            )
        else:
            self.failed_plans.append(
                parsed_plan
            )
        return 

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.failed_plans = [] 
        self.latest_chat_history = []