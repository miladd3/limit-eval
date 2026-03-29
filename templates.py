EVAL_TEMPLATE = """
You are evaluating a banking assistant that helps users check and manage debit card limits.

[Conversation history before the current user turn]
{conversation_history}

[Current user message]
{latest_user_input}

[Optional evaluation focus]
{evaluation_focus}

[Optional expected outcome]
{expected_outcome}

[Agent response]
{output}

[Evaluation criteria]
- "correct"   -> Response correctly addresses the current user message using the prior conversation context. It uses card data instead of hallucinating, and it follows any provided expected outcome.
- "partial"   -> Response is on topic but incomplete, vague, or asks for information that should already be inferable from the conversation context. It may only partially satisfy the expected outcome.
- "incorrect" -> Response fails to answer the current turn, ignores conversation context, contains errors or hallucinations, or contradicts the expected outcome.

Respond with exactly one word: correct, partial, or incorrect.
"""

CONVERSATION_EVAL_TEMPLATE = """
You are evaluating the overall quality of a banking assistant conversation about debit card limits.

[Case description]
{case_description}

[Conversation objectives]
{case_objectives}

[Full conversation transcript]
{transcript}

[Evaluation criteria]
- "correct"   -> The overall conversation is coherent, efficient, and accurate. The assistant tracks state across turns, avoids unnecessary repetition, and achieves the user's goal using real card data.
- "partial"   -> The conversation is somewhat helpful but has avoidable friction, repeated questions, incomplete progress, or weak state tracking.
- "incorrect" -> The conversation fails overall because it loses state, gives wrong information, loops unnecessarily, or does not complete the user's task.

Respond with exactly one word: correct, partial, or incorrect.
"""

PAIRWISE_EVAL_TEMPLATE = """
You are comparing two banking assistant conversations for the same task.

[Case description]
{case_description}

[Conversation objectives]
{case_objectives}

[Tools agent conversation]
{tools_transcript}

[MCP agent conversation]
{mcp_transcript}

[Comparison criteria]
- "tools_better" -> The tools conversation is clearly better overall because it is more accurate, coherent, efficient, or context-aware.
- "mcp_better"   -> The MCP conversation is clearly better overall because it is more accurate, coherent, efficient, or context-aware.
- "tie"          -> Both conversations are effectively similar in quality overall.

Respond with exactly one word: tools_better, mcp_better, or tie.
"""
