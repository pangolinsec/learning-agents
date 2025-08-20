# learning-agents
A repo for learning llamaindex and langchain, loosely following the Huggingface Agents course


## LlamaIndex

The main.py and llama_index_agent.py programs focus on llamaindex.

### Agents
Agents execute tools or other agents.

#### Agents that use tools
Agents that use tools can be created with functions like `ReActAgent()` or `FunctionAgent()`, and then called with the `.run()` method. They can also be created with the `AgentWorkflow.from_tools_or_functions()` method.

#### Agents that use other agents
Agents can also be created with the `AgentWorkflow()` function, and be called with the `.run()` method.
But if this approach is taken, then that agent needs to be given a list of agents it uses (each agent of which in turn needs to use tools).
