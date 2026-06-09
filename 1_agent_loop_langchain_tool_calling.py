from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MODEL = "openai:gpt-4.1-nano"
MAX_ITERATIONS = 10
llm = init_chat_model(MODEL, temperature=0)

# -- Tools (LangChain @tool decorator) --

@tool
def get_product_price(product_name: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"      >> Executing get_product_price(product='{product_name}')")
    product_catalog = {"laptop": 1299.99, "keyboard": 799.50, "headphones": 199.20}
    return product_catalog.get(product_name, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"      >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 15}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)
    
# -- Agent Loop --

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool (get_product_price) "
                "and a discount tool (apply_discount).\n\n"
                "STRICT RULES - follow these exactly:\n"
                "1. Never guess or assume any product price. "
                "You must call get_product_price first to get the real price. Also, there are no different model of a product what ever is present in that product catalog is all we have.\n"
                "2. Only call apply_discount After you have received a price from get_product_price. "
                "Pass the exact price returned by get_product_price; do not invent numbers.\n"
                "3. Never calculate discounts yourself using math. Always use the apply discount tool.\n"
                "4. If the user does not specify a discount tier, ask them which tier to use- do not assume one.\n"
            )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n --- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)

        # Debug: show the model's text and any detected tool calls
        print("\nai_message content:\n", ai_message.content)
        tool_calls = ai_message.tool_calls
        print("ai_message.tool_calls:", tool_calls)

        # If no tool calls, this is the final Answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f" [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max Iterations reached without a final Answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")