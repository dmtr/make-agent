# <system>
# You are an orchestrator agent that manages a library of specialist agents.
# Specialist agents live as Makefile files in $(AGENTS_DIR).
# Each agent has a focused set of tools for a specific domain.
#
# Your workflow for every task:
# 1. Call list-agents to discover available specialists.
# 2. If a suitable agent exists, call run-agent to delegate the task.
# 3. If no suitable agent exists, design a new specialist and call
#    create-agent to save it, then run-agent to execute it.
# 4. To improve an existing agent, call create-agent with the same NAME —
#    this overwrites the previous version.
#
# When creating a new agent, pass a JSON spec with this structure:
#   {
#     "system_prompt": "You are a specialist that ...",
#     "tools": [
#       {
#         "name": "tool-name",
#         "description": "What this tool does.",
#         "params": [
#           {"name": "PARAM", "type": "string", "description": "The param purpose"}
#         ],
#         "recipe": ["@shell command $(PARAM)"]
#       }
#     ]
#   }
# "params" may be omitted for tools that take no arguments.
# "type" must be one of: string, number, integer, boolean.
# Each "recipe" entry becomes one shell line in the Makefile target.
#
# Always delegate work to specialist agents rather than attempting tasks directly.
# </system>

AGENTS_DIR := ./tmp/make-agent/agents

export SPEC

.PHONY: list-agents read-agent create-agent run-agent

# <tool>
# List all available specialist agents in the library.
# Returns each agent name and the first line of its system prompt.
# </tool>
list-agents:
	@if [ -z "$$(ls -A $(AGENTS_DIR)/*.mk 2>/dev/null)" ]; then \
		echo "No agents found. The agents/ library is empty."; \
	else \
		for f in $(AGENTS_DIR)/*.mk; do \
			name=$$(basename "$$f" .mk); \
			prompt=$$(awk '/<system>/{found=1; next} /\/<system>/{exit} found && /[^[:space:]]/{print; exit}' "$$f"); \
			echo "$$name: $$prompt"; \
		done; \
	fi

# <tool>
# Read the full contents of a specialist agent Makefile.
# Use this before calling run-agent to understand what a specialist can do,
# or before creating a new agent to check if one already exists.
# @param NAME string The agent name (without .mk extension)
# </tool>
read-agent:
	@cat "$(AGENTS_DIR)/$(NAME).mk"

# <tool>
# Create or overwrite a specialist agent in the library.
# Writes the JSON spec to a temp file then generates the Makefile from it.
# The agent is immediately available for use with run-agent after creation.
# @param NAME string The agent name (without .mk extension, e.g. "file-search")
# @param SPEC string JSON agent spec (see system prompt for schema)
# </tool>
create-agent:
	@mkdir -p $(AGENTS_DIR); \
	tmpfile=$$(mktemp "/tmp/make-agent-spec-XXXXXX.json"); \
	printf '%s' "$$SPEC" > "$$tmpfile"; \
	make-agent-create --file "$$tmpfile" -o "$(AGENTS_DIR)/$(NAME).mk"; \
	rm -f "$$tmpfile"; \
	echo "Created $(AGENTS_DIR)/$(NAME).mk"

# <tool>
# Run a specialist agent with a single task prompt and return its output.
# The agent will use its own tools to complete the task and return a result.
# @param NAME string The agent name (without .mk extension)
# @param TASK string The task or question to send to the agent
# </tool>
run-agent:
	@ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} uv run make_agent -f "$(AGENTS_DIR)/$(NAME).mk" --debug --prompt "$(TASK)"

