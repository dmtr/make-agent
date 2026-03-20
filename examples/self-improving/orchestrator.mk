# <system>
# You are an orchestrator agent that manages a library of specialist agents.
# Specialist agents live as Makefile files in the ./agents/ directory.
# Each agent has a focused set of tools for a specific domain.
#
# Your workflow for every task:
# 1. Call list-agents to discover available specialists.
# 2. If a suitable agent exists, call run-agent to delegate the task.
# 3. If no suitable agent exists, design a new specialist Makefile and call
#    create-agent to save it, then run-agent to execute it.
# 4. To improve an existing agent, create a new one with the same NAME —
#    this overwrites the previous version.
#
# When writing a new agent Makefile, follow this exact format:
#   - A # <system> ... # </system> block with the agent's purpose
#   - One or more # <tool> ... # </tool> blocks, each above a Make target
#   - Use @param NAME type description inside <tool> blocks to declare parameters
#   - Separate lines with the literal \n in the CONTENT string
#
# Always delegate work to specialist agents rather than attempting tasks directly.
# </system>

.PHONY: list-agents read-agent create-agent run-agent

# <tool>
# List all available specialist agents in the agents/ library.
# Returns each agent filename and the first line of its system prompt.
# </tool>
list-agents:
	@if [ -z "$$(ls -A agents/*.mk 2>/dev/null)" ]; then \
		echo "No agents found. The agents/ library is empty."; \
	else \
		for f in agents/*.mk; do \
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
	@cat "agents/$(NAME).mk"

# <tool>
# Create or overwrite a specialist agent Makefile in the agents/ library.
# The CONTENT must be the full Makefile text with \n as line separators.
# The agent is immediately available for use with run-agent after creation.
# @param NAME string The agent name (without .mk extension, e.g. "file-search")
# @param CONTENT string The full Makefile content with \n as line separators
# </tool>
create-agent:
	@mkdir -p agents
	@printf '%b\n' "$(CONTENT)" > "agents/$(NAME).mk"
	@echo "Created agents/$(NAME).mk"

# <tool>
# Run a specialist agent with a single task prompt and return its output.
# The agent will use its own tools to complete the task and return a result.
# @param NAME string The agent name (without .mk extension)
# @param TASK string The task or question to send to the agent
# </tool>
run-agent:
	@make-agent -f "agents/$(NAME).mk" --prompt "$(TASK)"
