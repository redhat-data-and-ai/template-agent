package agent.authz

import rego.v1

# Additional banned words for local development.
# Git policy banned-words.rego defines base banned_words and allow rules;
# this file extends the same package without redefining banned_words (OPA conflict).
# Git allow/deny rules automatically pick up these supplementary checks via
# shared rule names (contains_banned_word_in_agent_message, etc.).

additional_banned_words := {
	"prescription",
	"nurse",
}

contains_banned_word_in_agent_message if {
	input.current_intent.action == "llm_response"
	agent_msg := lower(input.current_intent.agent_message)
	word := additional_banned_words[_]
	contains(agent_msg, lower(word))
}

contains_banned_word_in_tool_result if {
	input.current_intent.action == "tool_response"
	tool_result := lower(input.current_intent.result)
	word := additional_banned_words[_]
	contains(tool_result, lower(word))
}

deny_reasons contains msg if {
	input.current_intent.action == "llm_response"
	agent_msg := lower(input.current_intent.agent_message)
	word := additional_banned_words[_]
	contains(agent_msg, lower(word))
	msg := sprintf("Banned word '%s' found in agent response", [word])
}

deny_reasons contains msg if {
	input.current_intent.action == "tool_response"
	tool_result := lower(input.current_intent.result)
	word := additional_banned_words[_]
	contains(tool_result, lower(word))
	msg := sprintf("Banned word '%s' found in tool result", [word])
}
