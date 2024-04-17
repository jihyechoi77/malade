import langroid as lr
from langroid.utils.constants import PASS, PASS_TO
from langroid.agent.tools import RecipientTool, AddRecipientTool
from malade.tools.handling import handle_with

const = lambda x: lambda _: x
pass_to_handler = lambda x: const(PASS_TO + x)

def handle_with_pass_to(
        agent: lr.Agent,
        tool: type[lr.ToolMessage],
        destination_name: str,
) -> None:
    """Handles a `ToolMessage` by passing it to a certain recipient."""
    handle_with(agent, tool, pass_to_handler(destination_name))

def handle_with_pass(
        agent: lr.Agent,
        tool: type[lr.ToolMessage],
) -> None:
    """Handles a `ToolMessage` by passing."""
    handle_with(agent, tool, const(PASS))


def override_fallback_recipient_tool(fallback_msg: str) -> type[lr.ToolMessage]:
    class RecipientToolOverride(RecipientTool):
        # Modified from langroid.agent.tools.RecipientTool source
        @staticmethod
        def handle_message_fallback(
            agent: lr.ChatAgent, msg: str | lr.ChatDocument
        ) -> str | lr.ChatDocument | None:
            """
            Response of agent if this tool is not used, e.g.
            the LLM simply sends a message without using this tool.
            This method has two purposes:
            (a) Alert the LLM that it has forgotten to specify a recipient, and prod it
                to use the `add_recipient` tool to specify just the recipient
                (and not re-generate the entire message).
            (b) Save the content of the message in the agent's `content` field,
                so the agent can construct a ChatDocument with this content once LLM
                later specifies a recipient using the `add_recipient` tool.

            This method is used to set the agent's handle_message_fallback() method.

            Returns:
                (str): reminder to LLM to use the `add_recipient` tool.
            """
            # Note: once the LLM specifies a missing recipient, the task loop
            # mechanism will not allow any of the "native" responders to respond,
            # since the recipient will differ from the task name.
            # So if this method is called, we can be sure that the recipient has not
            # been specified.
            if isinstance(msg, str):
                return None
            if msg.metadata.sender != lr.Entity.LLM:
                return None
            content = msg if isinstance(msg, str) else msg.content
            # save the content as a class-variable, so that
            # we can construct the ChatDocument once the LLM specifies a recipient.
            # This avoids having to re-generate the entire message, saving time + cost.
            AddRecipientTool.saved_content = content
            agent.enable_message(AddRecipientTool)
            print("[red]RecipientTool: Recipient not specified, asking LLM to clarify.")
            return lr.ChatDocument(
                content=fallback_msg,
                metadata=lr.ChatDocMetaData(
                    sender=lr.Entity.AGENT,
                    recipient=lr.Entity.LLM,
                ),
            )

    return RecipientToolOverride
