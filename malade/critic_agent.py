import langroid as lr
from langroid.utils.constants import PASS
from malade.utils.constants import DONE
from malade.utils.formatting import format_list

class FinalAnswerTool(lr.ToolMessage):
    request: str = "final_answer"
    purpose: str = "To express your answer to a question and the reasoning steps used to derive it"
    question: str
    steps: list[str]
    answer: str

    def handle(self) -> str:
        reasoning_steps = format_list(self.steps, numbered=True)
            
        reasoning = f"""
        Question: {self.question}
        -----
        Reasoning:
        {reasoning_steps}
        -----
        Final answer: {self.answer}
        """

        return reasoning 

class FeedbackTool(lr.ToolMessage):
    request: str = "feedback"
    purpose: str = "To express your feedback of the reasoning process"
    critique: str

    def handle(self) -> str:
        if self.critique == "":
            return "Your reasoning is valid, no feedback was provided."

        return f"""
        Feedback: {self.critique}

        If any flaws in the reasoning used to produce your
        answer were identified, you must try again.
        """

class CriticConfig(lr.ChatAgentConfig):
    system_message: str = """
    You are an expert in logical reasoning whose task is to act as a
    critic, who identifies any flaws in the user's reasoning.

    {user_message}

    First, you will take time to think and determine the flaws in the
    reasoning; explain your thought process. Once you are done, you
    will use the `feedback` tool/function to respond with your
    criticism, if any. Do not begin composing the `feedback`
    message until you are certain whether or not you want the
    user to try again due to flawed reasoning.

    If the reasoning provided is correct and/or you cannot identify
    any specific flaws, the `critique` provided MUST be
    "". Otherwise, clearly describe any issues you have identified so
    the user can correct them.
    """
    user_message: str = ""
    criticized: type[lr.ToolMessage] = FinalAnswerTool

class Critic(lr.ChatAgent):
    def __init__(self, config: CriticConfig):
        super().__init__(config)
        self.system_message = config.system_message.format(
            user_message=config.user_message,
        )
        self.enable_message(
            config.criticized,
            handle=True,
            use=False,
        )
        self.enable_message(FeedbackTool, handle=True, use=True)

    def handle_message_fallback(
        self, msg: str | lr.ChatDocument
    ) -> str | lr.ChatDocument | None:
        if isinstance(msg, lr.ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            return """
            You MUST use the `feedback` tool/function in your next
            message.  If possible, send the critique you intended for
            the user's provided reasoning. If that is not possible or
            if you would simply acknowledge this message, you must
            instead use the `feedback` tool with `critique` set to "".
            """

    def feedback(self, _: FeedbackTool) -> str:
            return DONE + " " + PASS
