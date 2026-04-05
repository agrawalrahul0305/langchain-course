from dotenv import load_dotenv
import os
import dotenv
#from langchain.chat_models import init_chat_model
#from langchain.tools import tool
#from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import ollama
from langsmith import traceable
load_dotenv()
MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"
#MODEL = "functiongemma"

# --- Tools (Langchain @tool decorator) ---

#@tool Not available with raw ollama sdk
@traceable(name="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": "129.99", "keyboard": 49.99}
    return prices.get(product, 0)

@traceable(name="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools_to_use = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product e.g. laptop, headphones or keyboard"
                    }
                },
                "required": ["product"]
            }

        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price"
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'gold, 'silver or 'bronze'"
                    }
                },
                "required": ["price", "discount_tier"]
            }

        }
    }
]
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat(messages) -> ollama.ChatResponse:
    response = ollama.chat(model=MODEL, tools= tools_to_use , messages=messages)
    return response
# --- Agent Loop ---

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    #tools = [get_product_price, apply_discount]
    tools_dict = {
        "get_product_price" : get_product_price,
        "apply_discount": apply_discount
    }
    #tools_dict = {t.name: t for t in tools}
    #llm = chat(f"ollama:{MODEL}", temperature=0)
    #llm_with_tools = llm.bind_tools(tools)
    messages = [ 
        {
            "role": "system",
            "content": "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one.\n"
                "5. Respond only with the tool call. Do not provide any conversational text."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        response = ollama_chat(messages)
        ai_message = response.message
        #ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content
        print(f"Number of tool calls: {len(tool_calls)}")
        #tool_call = tool_calls[0]
        tool_name = tool_calls[0].function.name
        tool_args = tool_calls[0].function.arguments
        #tool_call_id = tool_call.get("id")
        print(f" [Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f" Tool: {tool_name} Not Found")
        observation = tool_to_use(**tool_args)
        #observation = tool_to_use
        print(f" [Tool Result] {observation}")
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation)
            }
        )

    print("Error Max Iterations reached")
    return None


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a headphones after applying a gold discount?")