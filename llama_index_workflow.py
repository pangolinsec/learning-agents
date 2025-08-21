from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Event
from llama_index.utils.workflow import draw_all_possible_flows
import asyncio
import random

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")


#w = MyWorkflow(timeout=10, verbose=False)



class ProcessingEvent(Event):
    intermediate_result: str

class LoopEvent(Event):
    loop_output: str

'''
class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        # Process initial data
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)
'''

class MultiStepWorkflow(Workflow):

    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

w = MultiStepWorkflow(timeout=10, verbose=False)

async def result():
    r = await w.run()
    print(r)

#result = await w.run()
#result

asyncio.run(result())
#draw_all_possible_flows(w, "flow.html")