# python -m vllm.entrypoints.openai.api_server --model microsoft/phi-3-mini-128k-instruct --max-model-len=8192
import textgrad as tg
from textgrad.engine import get_engine
eng = get_engine("vllm-microsoft/phi-3-mini-128k-instruct")
tg.set_backward_engine(eng, override=True)

# Step 1: Get an initial response from an LLM.
model = tg.BlackboxLLM(engine=eng)
question_string = ("If it takes 1 hour to dry 25 shirts under the sun, "
                   "how long will it take to dry 30 shirts under the sun? "
                   "Reason step by step")

question = tg.Variable(question_string, 
                       role_description="question to the LLM", 
                       requires_grad=False)

answer = model(question)
print(f'{answer=}')


initial_solution = """To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
x = (-b ± √(b^2 - 4ac)) / 2a
a = 3, b = -7, c = 2
x = (7 ± √((-7)^2 - 4 * 3(2))) / 6
x = (7 ± √(7^3) / 6
The solutions are:
x1 = (7 + √73)
x2 = (7 - √73)"""

# Define the variable to optimize, let requires_grad=True to enable gradient computation
solution = tg.Variable(initial_solution,
                       requires_grad=True,
                       role_description="solution to the math question")

# Define the optimizer, let the optimizer know which variables to optimize, and run the loss function

loss_fn = tg.TextLoss("You will evaluate a solution to a math question. Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.")

optimizer = tg.TGD(parameters=[solution])
loss = loss_fn(solution)

loss.backward()
optimizer.step()
print(solution.value)
