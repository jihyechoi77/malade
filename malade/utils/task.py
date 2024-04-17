import langroid as lr
from typing import Optional
from malade.utils.constants import DONE
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from typing import Type
Responder = lr.Entity | Type["Task"]

# Modified from langroid.agent.Task source
class Task(lr.Task):
    def __init__(
            self,
            agent: Optional[lr.Agent] = None,
            recipient: Optional[str] = None,
            max_cost=10, # catches potential infinite loops
            error_on_over_cost=True,
            **kwargs
    ):
        super().__init__(agent=agent, **kwargs)
        self.recipient = recipient # If set, forces a specific recipient for all results
        self.max_cost = max_cost
        self.error_on_over_cost = error_on_over_cost

    """Replaces DONE with <DONE> to avoid conflicts with drug names."""
    def result(self):
        result_msg = self.pending_message

        content = result_msg.content if result_msg else ""
        if content:
            # assuming it is of the form "DONE: <content>"
            content = content.replace(DONE, "").strip()
        fun_call = result_msg.function_call if result_msg else None
        tool_messages = result_msg.tool_messages if result_msg else []
        block = result_msg.metadata.block if result_msg else None
        recipient = result_msg.metadata.recipient if result_msg else None
        responder = result_msg.metadata.parent_responder if result_msg else None
        tool_ids = result_msg.metadata.tool_ids if result_msg else []

        # regardless of which entity actually produced the result,
        # when we return the result, we set entity to USER
        # since to the "parent" task, this result is equivalent to a response from USER
        return lr.ChatDocument(
            content=content,
            function_call=fun_call,
            tool_messages=tool_messages,
            metadata=lr.ChatDocMetaData(
                source=lr.Entity.USER,
                sender=lr.Entity.USER,
                block=block,
                parent_responder=responder,
                sender_name=self.name,
                recipient=self.recipient or recipient,
                tool_ids=tool_ids,
            ),
        )

    def done(
        self, result: lr.ChatDocument | None = None, r: lr.agent.task.Responder | None = None
    ) -> bool:
        """
        Check if task is done. This is the default behavior.
        Derived classes can override this.
        Args:
            result (ChatDocument|None): result from a responder
            r (Responder|None): responder that produced the result
                Not used here, but could be used by derived classes.
        Returns:
            bool: True if task is done, False otherwise
        """
        result = result or self.pending_message
        user_quit = (
            result is not None
            and result.content in lr.utils.constants.USER_QUIT
            and result.metadata.sender == lr.Entity.USER
        )
        if self._level == 0 and self.only_user_quits_root:
            # for top-level task, only user can quit out
            return user_quit

        if self.is_done:
            return True

        if self.n_stalled_steps >= self.max_stalled_steps:
            # we are stuck, so bail to avoid infinite loop
            logger.warning(
                f"Task {self.name} stuck for {self.max_stalled_steps} steps; exiting."
            )
            return True

        if self.agent.total_llm_token_cost > self.max_cost:
            logger.warning(f"Task {self.name}: total cost {self.agent.total_llm_token_cost} exceeds maximum of {self.max_cost}")
            if self.error_on_over_cost:
                raise RuntimeError("Exceeded maximum token cost for task")

        return (
            # no valid response from any entity/agent in current turn
            result is None
            # An entity decided task is done
            or DONE in result.content
            or self.agent.total_llm_token_cost > self.max_cost
            or (  # current task is addressing message to caller task
                self.caller is not None
                and self.caller.name != ""
                and result.metadata.recipient == self.caller.name
            )
            # or (
            #     # Task controller is "stuck", has nothing to say
            #     NO_ANSWER in result.content
            #     and result.metadata.sender == self.controller
            # )
            or user_quit
        )


    def _is_done_response(
        self, result: str | None | lr.ChatDocument, responder: lr.agent.task.Responder
    ) -> bool:
        """Is the task done based on the response from the given responder?"""

        response_says_done = result is not None and (
            (isinstance(result, str) and DONE in result)
            or (isinstance(result, lr.ChatDocument) and DONE in result.content)
        )
        return (
            (
                responder.value in self.done_if_response
                and not self._is_empty_message(result)
            )
            or (
                responder.value in self.done_if_no_response
                and self._is_empty_message(result)
            )
            or (not self._is_empty_message(result) and response_says_done)
        )
