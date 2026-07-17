import logging
from typing import Dict, List, Optional
from .chatbot_intent import ChatbotIntentParser

logger = logging.getLogger(__name__)

class CookingCoach:
    """
    Hands-Free AI Cooking Coach.
    Tracks recipe state and provides contextual help via an LLM.
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
        self.intent_parser = ChatbotIntentParser(api_key=api_key, provider=provider)
        
    def parse_instructions(self, instructions_text: str) -> List[str]:
        """Convert a raw instruction string into a list of steps."""
        if not instructions_text:
            return []
        
        # Split by typical delimiters (newlines or numbered lists)
        raw_steps = [s.strip() for s in instructions_text.replace("\r", "").split("\n") if s.strip()]
        
        # Simple cleanup
        steps = []
        for step in raw_steps:
            # Strip leading numbers/bullets if any, or just keep as is.
            # Keeping as is for simplicity, assuming they are somewhat clean.
            if len(step) > 3:
                steps.append(step)
                
        return steps if steps else [instructions_text]

    def handle_command(self, user_input: str, recipe: Dict, current_step_index: int) -> Dict:
        """
        Process a user command (e.g., "next step", "how to fix thick sauce?").
        Returns the new state and the response text.
        """
        instructions_text = recipe.get("instructions", "")
        steps = self.parse_instructions(instructions_text)
        
        if not steps:
            return {
                "response": "I don't have step-by-step instructions for this recipe.",
                "new_index": current_step_index,
                "is_step_change": False
            }
            
        max_idx = len(steps) - 1
        
        # Basic intent matching for navigation
        lower_input = user_input.lower()
        if any(w in lower_input for w in ["next", "forward", "continue"]):
            new_idx = min(current_step_index + 1, max_idx)
            return {
                "response": f"Step {new_idx + 1}: {steps[new_idx]}",
                "new_index": new_idx,
                "is_step_change": True
            }
            
        if any(w in lower_input for w in ["previous", "back", "last step"]):
            new_idx = max(current_step_index - 1, 0)
            return {
                "response": f"Going back. Step {new_idx + 1}: {steps[new_idx]}",
                "new_index": new_idx,
                "is_step_change": True
            }
            
        if any(w in lower_input for w in ["repeat", "what was that", "say again"]):
            return {
                "response": f"Step {current_step_index + 1}: {steps[current_step_index]}",
                "new_index": current_step_index,
                "is_step_change": False
            }
            
        # If it's not a simple navigation command, use the LLM to answer the cooking question contextually
        current_step_text = steps[current_step_index] if 0 <= current_step_index <= max_idx else ""
        
        context_prompt = f"""
You are an expert AI cooking coach. The user is currently cooking '{recipe.get("title")}'.
They are currently on this step: "{current_step_text}".
The user just asked: "{user_input}"
Provide a brief, helpful, and encouraging answer to their question so they can continue cooking.
Do not lose track of the recipe.
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful culinary AI assistant."},
            {"role": "user", "content": context_prompt}
        ]
        
        try:
            llm_response = self.intent_parser.chat_with_chef(messages)
        except Exception as e:
            logger.error(f"LLM Error in Cooking Coach: {e}")
            llm_response = "I'm sorry, I couldn't process that question right now. You are still on step " + str(current_step_index + 1) + "."

        return {
            "response": llm_response,
            "new_index": current_step_index,
            "is_step_change": False
        }
